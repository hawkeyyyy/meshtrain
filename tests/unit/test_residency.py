"""V2 TensorStore + ResidencyManager unit tests (CPU; the CPU working set
stands in for accelerator memory, with real copies)."""

import pytest
import torch

from meshtrain.config import ConfigError, parse_bytes, parse_config
from meshtrain.models import TinyTransformerSpec
from meshtrain.runtime.distributed_autograd import MicrobatchContext
from meshtrain.runtime.offload import GroupState, ResidencyPolicy
from meshtrain.runtime.stage import Stage
from meshtrain.runtime.tensor_store import (
    LocalTensorStore,
    MemoryBudgetError,
    MemoryTier,
    TensorRole,
    format_tensor_table,
)

ACC, RAM = MemoryTier.LOCAL_ACCELERATOR, MemoryTier.LOCAL_RAM
SPEC = TinyTransformerSpec(layers=3, hidden_size=32, heads=4, vocab_size=32, seq_len=8)


def _stage(policy=None, budget=None, optimizer="adamw", offset=0):
    n = SPEC.num_layers
    return Stage(SPEC.build_stage(offset, n), stage_index=0, num_stages=1, optimizer=optimizer, lr=1e-2,
                 loss_fn=SPEC.loss_fn, layer_offset=offset, residency=policy, accelerator_budget=budget)


def _train_step(st, step=0, M=2):
    x, y = SPEC.make_batch(step, 4)
    for mb, (xm, ym) in enumerate(zip(x.chunk(M), y.chunk(M))):
        ctx = MicrobatchContext(step, mb)
        st.forward(xm, ctx)
        st.loss_backward(ym, (step, mb), loss_scale=1.0 / M)
    st.optimizer_step()
    st.zero_grad()


# -- TensorStore metadata (V2.0) ------------------------------------------------
def test_register_metadata_and_tier_totals():
    store = LocalTensorStore("w", "cuda", compute_stage=2)
    m = store.register("model.layers.4.qkv.weight", torch.zeros(8, 4), TensorRole.PARAMETER)
    assert (m.tier, m.size_bytes, m.shape, m.dtype, m.owner_worker, m.compute_stage) == (ACC, 128, (8, 4), "float32",
                                                                                         "w", 2)
    assert not m.dirty and not m.pinned and m.version == 0 and m.current_location == ACC
    store.register("x", torch.zeros(4), TensorRole.OPTIMIZER_STATE, RAM)
    assert store.bytes_by_tier() == {"local_accelerator": 128, "local_ram": 16}
    st = store.stats()
    assert st["accelerator_resident"] == 128 and st["ram_resident"] == 16 and st["optimizer_bytes"] == 16
    again = store.register("model.layers.4.qkv.weight", torch.zeros(8, 4), TensorRole.PARAMETER)
    assert again.version == 1 and store.bytes_by_tier()["local_accelerator"] == 128  # replaced, not double-counted
    assert "model.layers.4.qkv.weight" in format_tensor_table(store.records())


def test_static_stage_tracks_all_parameters_on_accelerator_tier():
    st = _stage(ResidencyPolicy())  # static policy object: V1.5 path, no manager
    assert st.residency is None
    ids = st.tensor_store.ids(TensorRole.PARAMETER)
    assert len(ids) == sum(1 for _ in st.module.parameters()) and "model.layers.1.qkv.weight" in ids
    assert {st.tensor_store.locate(i).tier for i in ids} == {RAM}  # CPU stage: computes from RAM (V1.5)


def test_unimplemented_tiers_raise():
    store = LocalTensorStore("w", "cuda")
    store.register("t", torch.zeros(2), TensorRole.PARAMETER)
    for tier in (MemoryTier.REMOTE_RAM, MemoryTier.REMOTE_ACCELERATOR, MemoryTier.LOCAL_NVME, MemoryTier.RECOMPUTE):
        with pytest.raises(NotImplementedError):
            store.move("t", tier)
        with pytest.raises(NotImplementedError):
            store.set_residency("t", tier)


def test_standalone_move_and_dirty_writeback_rules():
    store = LocalTensorStore("w", "cpu", offload=True, accelerator_budget=64)
    store.register("t", torch.arange(4.0), TensorRole.TEMPORARY, RAM)
    m = store.move("t", ACC)
    assert m.tier == ACC and store.stats()["accelerator_resident"] == 16
    store.set_residency("t", RAM, (ACC,))
    store.mark_dirty("t")
    m = store.locate("t")
    assert m.dirty and m.tier == ACC and m.version == 1  # moves keep the version; updates bump it
    with pytest.raises(RuntimeError, match="dirty"):
        store.set_residency("t", RAM, ())  # never silently drop a dirty copy
    store.set_residency("t", RAM, (ACC,), dirty=False)  # written back
    with pytest.raises(MemoryBudgetError) as e:
        store.reserve_accelerator(1000, "big", RAM)
    assert e.value.requested == 1000 and e.value.available == 64 - 16 and "big" in str(e.value)


def test_config_strategies_and_budget_parsing():
    assert parse_bytes("6GB") == 6 * 1024**3 and parse_bytes("512 MB") == 512 * 1024**2
    cfg = parse_config({"model": {"type": "mlp"}, "memory": {"strategy": "auto_offload", "accelerator_budget": "2GB",
                                                              "optimizer_offload": True, "prefetch_distance": 2}})
    pol = cfg.residency_policy()
    assert pol.strategy == "auto_offload" and pol.optimizer_execution == "cpu_offload" and pol.prefetch_distance == 2
    assert cfg.memory.budget_bytes() == 2 * 1024**3
    assert parse_config({"model": {"type": "mlp"}, "memory": {"accelerator_budget_mb": 2048}}).memory.budget_bytes() \
        == 2048 * 1024**2
    assert not parse_config({"model": {"type": "mlp"}}).residency_policy().active  # default = V1.5
    with pytest.raises(ConfigError, match="remote RAM"):
        parse_config({"model": {"type": "mlp"}, "memory": {"use_remote_ram": True}})


# -- ResidencyManager (V2.1+) ----------------------------------------------------
def test_manual_offload_keeps_parameter_objects_and_ram_authority():
    st = _stage(ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0",), prefetch_distance=0))
    res = st.residency
    params_before = [id(p) for p in st.module.parameters()]
    assert [g.group_id for g in res.hot] == ["model.layers.0"] and len(res.cold) == SPEC.num_layers - 1
    for g in res.cold:
        assert g.state == GroupState.RESIDENT_RAM
        for tid, p in g.params:
            assert p.data_ptr() == g.host[id(p)].data_ptr() and st.tensor_store.locate(tid).tier == RAM
    _train_step(st)
    assert [id(p) for p in st.module.parameters()] == params_before  # identity stable
    assert all(p in st.optimizer.state for p in st.module.parameters())  # optimizer saw the same objects
    for g in res.cold:
        assert g.state == GroupState.RESIDENT_RAM
        for tid, p in g.params:
            meta = st.tensor_store.locate(tid)
            assert meta.tier == RAM and not meta.dirty and meta.version >= 1  # updated + written back
            assert p.data_ptr() == g.host[id(p)].data_ptr()
    assert res.audit()["problems"] == []


def test_lifecycle_transitions_are_validated():
    st = _stage(ResidencyPolicy(strategy="manual_offload", keep_resident=("layers.0",)))
    g = st.residency.cold[0]
    with pytest.raises(RuntimeError, match="invalid residency transition"):
        g.transition(GroupState.WRITEBACK)          # RESIDENT_RAM -> WRITEBACK is not a lifecycle edge
    with pytest.raises(RuntimeError, match="resident"):
        st.residency.unload(st.residency.hot[0])    # HOT layers are never evicted
    st.residency.ensure_loaded(g, "forward")
    with pytest.raises(RuntimeError, match="in_use_forward"):
        st.residency.unload(g)                      # cannot evict a layer that is computing


def test_eviction_counts_prefetch_hits_and_writeback():
    st = _stage(ResidencyPolicy(strategy="manual_offload", prefetch_distance=1))
    _train_step(st)
    stats = st.residency.step_stats()
    n_cold = len(st.residency.cold)
    assert stats["tensor_eviction_count"] >= 2 * n_cold        # after forward and after backward
    assert stats["prefetch_hits"] > 0 and stats["tensor_cache_hit_ratio"] > 0.5
    assert stats["writeback_count"] == n_cold                   # accelerator optimizer dirtied every layer
    assert stats["H2D_bytes"] > 0 and stats["D2H_bytes"] > 0


def test_after_backward_policy_keeps_layer_until_backward():
    st = _stage(ResidencyPolicy(strategy="manual_offload", eviction="after_backward", prefetch_distance=0))
    x, _ = SPEC.make_batch(0, 2)
    st.forward(x, MicrobatchContext(0, 0))
    assert all(g.state == GroupState.SAVED_FOR_BACKWARD for g in st.residency.cold)


def test_budget_enforced_and_pressure_eviction():
    pol = ResidencyPolicy(strategy="manual_offload", prefetch_distance=0, eviction="after_backward")
    probe = _stage(pol)
    reserved = probe.tensor_store.accelerator_resident_bytes()
    biggest = max(g.param_bytes for g in probe.residency.cold)
    # Room for the startup reservations plus exactly one offloaded layer at a time.
    budget = reserved + biggest
    st = _stage(pol, budget=budget)
    _train_step(st)
    stats = st.residency.step_stats()
    assert stats["pressure_evictions"] > 0                      # after_backward had to make room
    assert stats["accelerator_resident_peak"] <= budget
    assert stats["budget_violations"] == 0
    with pytest.raises(MemoryBudgetError) as e:
        _stage(pol, budget=reserved - 1)
    assert e.value.destination == ACC


def test_load_failure_names_tensor_and_tiers():
    pol = ResidencyPolicy(strategy="manual_offload", prefetch_distance=0)
    probe = _stage(pol)
    st = _stage(pol, budget=probe.tensor_store.accelerator_resident_bytes() + 8)  # no room for any layer
    x, _ = SPEC.make_batch(0, 2)
    with pytest.raises(MemoryBudgetError) as e:
        st.forward(x, MicrobatchContext(0, 0))
    msg = str(e.value)
    assert "model.layers." in msg and "local_ram -> local_accelerator" in msg and "requested" in msg


def test_cpu_optimizer_offload_puts_state_in_ram():
    st = _stage(ResidencyPolicy(strategy="manual_offload", optimizer_execution="cpu_offload"))
    _train_step(st)
    store = st.tensor_store
    opt_ids = store.ids(TensorRole.OPTIMIZER_STATE)
    assert opt_ids and all(store.locate(i).tier == RAM for i in opt_ids if ".optim." in i)
    for p in st.module.parameters():
        for v in st.optimizer.state[p].values():
            assert v.device.type == "cpu"
    assert st.residency.step_stats()["writeback_count"] == 0     # CPU step updates the RAM master directly


def test_gradients_are_reported_from_host_accumulators():
    st = _stage(ResidencyPolicy(strategy="manual_offload"))
    x, y = SPEC.make_batch(0, 4)
    for mb, (xm, ym) in enumerate(zip(x.chunk(2), y.chunk(2))):
        st.forward(xm, MicrobatchContext(0, mb))
        st.loss_backward(ym, (0, mb), loss_scale=0.5)
    grads = st.named_gradients()
    assert len(grads) == sum(1 for _ in st.module.parameters())
    assert all(p.grad is None for g in st.residency.cold for _, p in g.params)  # device grads freed
