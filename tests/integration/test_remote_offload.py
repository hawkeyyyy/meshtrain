"""V2.5 integration: REMOTE_RAM as a backing tier, against static and local offload.

The remote RAM worker is a TensorServer on a loopback thread (same protocol and
code path as a remote machine). CPU runs are bit-exact against static training;
CUDA runs match local CPU-optimizer offload (same CPU AdamW kernels).
"""

import threading
import time

import pytest
import torch

from meshtrain.experiments.offload_correctness import DEFAULT_MODEL, build_single_stage, compare_traces, train_trace
from meshtrain.networking.tensor_server import RemoteMemoryError, start_local_server
from meshtrain.runtime.checkpoint import load_stage_checkpoint, save_stage_checkpoint
from meshtrain.runtime.offload import HostState, ResidencyPolicy, RemoteSpec
from meshtrain.runtime.pipeline import PipelineError, PipelineSettings, run_stage
from meshtrain.runtime.tensor_store import MemoryBudgetError, MemoryTier, TensorRole

MB = 1024**2
REMOTE_LAYERS = ("model.layers.2", "model.layers.3", "model.layers.5")


@pytest.fixture(scope="module")
def ram_server():
    srv, port, stop = start_local_server(512 * MB, lease_s=60)
    yield srv, port
    stop.set()


def _remote_policy(port, job, *, reuse=True, prefetch=1, groups=REMOTE_LAYERS, resident=("model.layers.0",),
                   checksum="crc32", **kw):
    spec = RemoteSpec(address=f"127.0.0.1:{port}", worker="loopback-ram", job_id=job, budget_bytes=64 * MB,
                      checksum=checksum, timeout_s=20, **kw)
    return ResidencyPolicy(strategy="remote_offload", optimizer_execution="cpu_offload", resident_groups=resident,
                           remote_groups=groups, remote=spec, reuse=reuse, prefetch_distance=prefetch)


def _optimizer_state(stage):
    """Logical AdamW state, fetching remote layers' state from their remote copy."""
    out = {}
    res = stage.residency
    remote_vals = {}
    if res is not None:
        for g in res.remote_groups:
            remote_vals.update(res.fetch_logical(g))
    for n, p in stage.module.named_parameters():
        tid = stage.global_name(n)
        for k in ("exp_avg", "exp_avg_sq"):
            key = f"{tid}.optim.{k}"
            if key in remote_vals:
                out[key] = remote_vals[key]
            elif k in stage.optimizer.state.get(p, {}):
                out[key] = stage.optimizer.state[p][k].detach().cpu().clone()
    return out


@pytest.mark.parametrize("reuse", [False, True])
def test_remote_matches_static_and_local_every_step(ram_server, reuse):
    srv, port = ram_server
    static = train_trace(DEFAULT_MODEL, steps=4)
    local = train_trace(DEFAULT_MODEL, steps=4, policy=ResidencyPolicy(
        strategy="manual_offload", optimizer_execution="cpu_offload", keep_resident=("layers.0",)))
    remote = train_trace(DEFAULT_MODEL, steps=4, policy=_remote_policy(port, f"eq-{reuse}", reuse=reuse))
    for other in (local, remote):
        for r in compare_traces(static, other):
            assert r["loss_abs_diff"] == 0 and r["grad_max_abs_diff"] == 0
            assert r["param_max_abs_diff"] == 0 and r["logits_max_abs_diff"] == 0
    st = remote.stats[-1]["residency"]
    assert st["groups_remote"] == 3 and st["remote_fetch_count"] > 0 and st["remote_writeback_count"] == 3
    assert srv.used == 0 and srv.reserved == 0          # released at close: no leaked remote memory


def test_optimizer_state_lives_remotely_and_matches(ram_server):
    _, port = ram_server
    a, spec = build_single_stage(DEFAULT_MODEL)
    b, _ = build_single_stage(DEFAULT_MODEL, policy=_remote_policy(port, "opt-state"))
    for i in range(3):
        s = PipelineSettings("x", steps=1, batch_size=8, num_microbatches=4, step_offset=i, trace=False, log_every=99)
        run_stage(a, spec, s, upstream=None, downstream=None)
        run_stage(b, spec, s, upstream=None, downstream=None)
    sa, sb = _optimizer_state(a), _optimizer_state(b)
    assert set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    res = b.residency
    for g in res.remote_groups:
        assert g.host_state == HostState.REMOTE_ONLY          # nothing staged between steps
        for tid, p in g.params:
            meta = b.tensor_store.locate(tid)
            assert meta.tier == MemoryTier.REMOTE_RAM and not meta.dirty and not meta.cached
            assert g.versions[tid] == 4 == meta.version       # v1 placement + 3 committed updates
            assert p.numel() == 0                             # placeholder: no local copy kept
            assert "exp_avg" not in b.optimizer.state[p]      # AdamW moments live remotely
            assert b.tensor_store.locate(f"{tid}.optim.exp_avg").tier == MemoryTier.REMOTE_RAM
    assert res.audit()["problems"] == []
    a.close()
    b.close()


def test_reuse_reduces_remote_fetches(ram_server):
    _, port = ram_server
    off = train_trace(DEFAULT_MODEL, steps=2, policy=_remote_policy(port, "r-off", reuse=False), capture=False)
    on = train_trace(DEFAULT_MODEL, steps=2, policy=_remote_policy(port, "r-on", reuse=True), capture=False)
    f_off = off.stats[-1]["residency"]["remote_fetch_count"]
    f_on = on.stats[-1]["residency"]["remote_fetch_count"]
    assert f_on * 3 <= f_off                             # 4 microbatches: 10 -> 2 fetches per layer
    assert on.stats[-1]["residency"]["remote_reuse_ratio"] >= 0.75
    assert off.losses == on.losses


def test_local_ram_budget_forces_remote_and_is_enforced(ram_server):
    """Local offload does not fit the local RAM budget; remote offload of the same model does."""
    from meshtrain.models import build_model_spec
    from meshtrain.planner.residency import plan_stage_residency

    _, port = ram_server
    model = {**DEFAULT_MODEL, "layers": 6}
    spec = build_model_spec(model)
    kw = dict(microbatch_size=2, optimizer="adamw", backend="cpu", num_microbatches=4)
    unconstrained = plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=ResidencyPolicy(
        strategy="auto_offload", optimizer_execution="cpu_offload"), **kw)
    local_budget_mb = None
    for frac in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):   # tightest budget the remote plan still fits
        b_mb = unconstrained.ram_bytes * frac / MB
        local_ok = plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=ResidencyPolicy(
            strategy="auto_offload", optimizer_execution="cpu_offload", local_ram_budget_mb=b_mb), **kw).feasible
        rp = _remote_policy(port, "probe", groups=None, resident=None)
        rp.local_ram_budget_mb = b_mb
        if not local_ok and plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=rp, **kw).feasible:
            local_budget_mb = b_mb
    assert local_budget_mb is not None
    local_pol = ResidencyPolicy(strategy="auto_offload", optimizer_execution="cpu_offload",
                                local_ram_budget_mb=local_budget_mb)
    local_plan = plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=local_pol, **kw)
    assert not local_plan.feasible and "local RAM" in local_plan.reason
    with pytest.raises(MemoryBudgetError) as e:          # runtime enforces the same budget
        build_single_stage(model, policy=ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload",
                                                         keep_resident=(), local_ram_budget_mb=local_budget_mb))
    assert e.value.destination == MemoryTier.LOCAL_RAM
    pol = _remote_policy(port, "budget", groups=None, resident=None)
    pol.local_ram_budget_mb = local_budget_mb
    plan = plan_stage_residency(spec, 0, spec.num_layers, budget=None, policy=pol, **kw)
    assert plan.feasible and plan.remote_groups
    static = train_trace(model, steps=3)
    tr = train_trace(model, steps=3, policy=pol)
    assert all(r["param_max_abs_diff"] == 0 for r in compare_traces(static, tr))
    for st in tr.stats:
        assert st["tensor_store"]["local_peak"] <= local_budget_mb * MB
        assert st["residency"]["budget_violations"] == 0


@pytest.mark.parametrize("load_remote", [False, True])
def test_checkpoint_with_remote_tensors(ram_server, tmp_path, load_remote):
    _, port = ram_server
    ref, spec = build_single_stage(DEFAULT_MODEL)
    for i in range(4):
        run_stage(ref, spec, PipelineSettings("c", steps=1, batch_size=8, num_microbatches=4, step_offset=i,
                                              trace=False, log_every=99), upstream=None, downstream=None)
    want = ref.named_parameters_cpu()
    a, _ = build_single_stage(DEFAULT_MODEL, policy=_remote_policy(port, f"ck-save-{load_remote}"))
    for i in range(2):
        run_stage(a, spec, PipelineSettings("c", steps=1, batch_size=8, num_microbatches=4, step_offset=i,
                                            trace=False, log_every=99), upstream=None, downstream=None)
    save_stage_checkpoint(a, tmp_path / "r.pt", step=2)   # remote layers fetched for the logical state
    a.close()
    pol = _remote_policy(port, f"ck-load-{load_remote}") if load_remote else None
    b, _ = build_single_stage(DEFAULT_MODEL, policy=pol)
    load_stage_checkpoint(b, tmp_path / "r.pt")
    for i in range(2, 4):
        run_stage(b, spec, PipelineSettings("c", steps=1, batch_size=8, num_microbatches=4, step_offset=i,
                                            trace=False, log_every=99), upstream=None, downstream=None)
    got = b.named_parameters_cpu()
    assert all(torch.equal(want[k], got[k]) for k in want)
    b.close()


def test_remote_worker_disappearing_stops_training_cleanly():
    srv, port, stop = start_local_server(256 * MB)
    stage, spec = build_single_stage(DEFAULT_MODEL, policy=_remote_policy(port, "dies", reuse=False))
    run_stage(stage, spec, PipelineSettings("d", steps=1, batch_size=8, num_microbatches=4, trace=False,
                                            log_every=99), upstream=None, downstream=None)
    stop.set()
    srv.close()
    stage.residency.client.close()       # also drop pooled connections: the worker is gone
    stage.residency.client.closed = False
    t0 = time.time()
    with pytest.raises((RemoteMemoryError, PipelineError, RuntimeError)) as e:
        run_stage(stage, spec, PipelineSettings("d", steps=1, batch_size=8, num_microbatches=4, step_offset=1,
                                                trace=False, log_every=99), upstream=None, downstream=None)
    msg = str(e.value)
    assert "loopback-ram" in msg or "127.0.0.1" in msg
    assert time.time() - t0 < 60
    stage.close()


def test_stress_100_steps_no_leaks_no_stale_versions(ram_server):
    srv, port = ram_server
    pol = _remote_policy(port, "stress", reuse=True, prefetch=2)
    stage, spec = build_single_stage(DEFAULT_MODEL, policy=pol)
    static, _ = build_single_stage(DEFAULT_MODEL)
    used_after = []
    local_peaks = []
    for i in range(100):
        s = PipelineSettings("s", steps=1, batch_size=8, num_microbatches=4, step_offset=i, trace=False,
                             log_every=999)
        r = run_stage(stage, spec, s, upstream=None, downstream=None)
        ref = run_stage(static, spec, s, upstream=None, downstream=None)
        assert r.losses == ref.losses                      # no divergence, every step
        used_after.append(srv.stats()["jobs"]["stress"]["used"])
        local_peaks.append(r.step_metrics[0]["tensor_store"]["local_peak"])
        assert not stage.residency._writebacks             # no queue growth across steps
    assert len(set(used_after)) == 1                       # remote memory constant: nothing leaked
    assert max(local_peaks[5:]) <= max(local_peaks[:5])    # local buffers not growing
    for g in stage.residency.remote_groups:
        for tid, _ in g.params:
            assert g.versions[tid] == 101 and g.versions[f"{tid}.optim.exp_avg"] == 100
    assert stage.residency.audit()["problems"] == []
    stage.close()
    static.close()
    assert "stress" not in srv.stats()["jobs"]


def test_emulated_slow_link_still_trains_and_is_slower(ram_server):
    _, port = ram_server
    fast = train_trace(DEFAULT_MODEL, steps=2, policy=_remote_policy(port, "fast"), capture=False)
    slow = train_trace(DEFAULT_MODEL, steps=2, capture=False, policy=_remote_policy(
        port, "slow", emulate_bandwidth_Bps=5e6, emulate_latency_s=0.002))
    assert slow.losses == fast.losses
    t_fast = fast.stats[-1]["step_s"]
    t_slow = slow.stats[-1]["step_s"]
    assert t_slow > t_fast


@pytest.mark.cuda
def test_cuda_compute_with_remote_cpu_ram(ram_server):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _, port = ram_server
    local = train_trace(DEFAULT_MODEL, device="cuda", steps=4, policy=ResidencyPolicy(
        strategy="manual_offload", optimizer_execution="cpu_offload", keep_resident=("layers.0",)))
    remote = train_trace(DEFAULT_MODEL, device="cuda", steps=4, policy=_remote_policy(port, "cuda-remote"))
    for r in compare_traces(local, remote):   # same CPU AdamW kernels: remote == local offload
        assert r["loss_abs_diff"] == 0 and r["param_max_abs_diff"] == 0 and r["grad_max_abs_diff"] == 0
    st = remote.stats[-1]["residency"]
    assert st["remote_fetch_count"] > 0 and st["H2D_bytes"] > 0
