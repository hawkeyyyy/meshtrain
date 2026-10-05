"""``meshtrain`` command-line interface.

    meshtrain coordinator start [--port 8080]
    meshtrain worker join HOST:PORT [--device cuda|mps|cpu] [--name NAME]
    meshtrain cluster status
    meshtrain cluster benchmark
    meshtrain plan CONFIG
    meshtrain train CONFIG            # on the cluster via the coordinator
    meshtrain train CONFIG --local    # every stage as a local process (no coordinator)
    meshtrain experiment correctness|placement|capacity

Coordinator address and token default to $MESHTRAIN_COORDINATOR and
$MESHTRAIN_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

DEFAULT_TOKEN = "meshtrain-dev-token"


def _token(args) -> str:
    tok = args.token or os.environ.get("MESHTRAIN_TOKEN")
    if not tok:
        print(f"warning: no --token / $MESHTRAIN_TOKEN given, using the insecure default {DEFAULT_TOKEN!r}",
              file=sys.stderr)
        tok = DEFAULT_TOKEN
    return tok


def _client(args):
    from meshtrain.networking.control import ControlClient

    return ControlClient(args.coordinator or os.environ.get("MESHTRAIN_COORDINATOR", "127.0.0.1:8080"), _token(args))


def _gb(n) -> str:
    return f"{n / 1024**3:.1f} GB" if n else "-"


# -- coordinator / worker --------------------------------------------------
def cmd_coordinator_start(args) -> int:
    from meshtrain.coordinator.server import run_coordinator

    run_coordinator(args.host, args.port, token=_token(args), runs_dir=args.runs_dir,
                    heartbeat_timeout_s=args.heartbeat_timeout)
    return 0


def cmd_worker_join(args) -> int:
    from meshtrain.worker.worker import WorkerAgent

    agent = WorkerAgent(args.coordinator_address, _token(args), device=args.device, name=args.name,
                        data_port=args.data_port, advertise_host=args.advertise_host, runs_dir=args.runs_dir,
                        quick_benchmark=args.quick_benchmark)
    print(f"MeshTrain worker {agent.name}: backend={agent.device.backend} device={agent.device.name()} "
          f"data-plane port={agent.dataplane.port}", flush=True)
    try:
        agent.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
    return 0


# -- cluster -----------------------------------------------------------------
def format_status(status: dict) -> str:
    lines = ["MeshTrain Cluster", "",
             f"{'ID':<20}{'Backend':<9}{'Device':<26}{'Memory':<12}{'RAM':<10}{'Score':<7}{'Status':<8}",
             "-" * 92]
    unified = False
    for w in status["workers"]:
        dev = w.get("device") or {}
        mem = _gb(dev.get("memory_total")) if w["backend"] != "cpu" else "-"
        if dev.get("unified_memory"):
            mem += "*"
            unified = True
        score = (w.get("benchmark") or {}).get("compute_score")
        lines.append(f"{w['worker_id'][:19]:<20}{w['backend'].upper():<9}{str(dev.get('name', ''))[:25]:<26}"
                     f"{mem:<12}{_gb(w.get('ram_total')):<10}{(f'{score:.2f}' if score is not None else '-'):<7}"
                     f"{w['status']:<8}" + (f" job={w['current_job']}" if w.get("current_job") else ""))
    if unified:
        lines += ["", "* unified memory"]
    if status.get("jobs"):
        lines += ["", "Jobs:"] + [f"  {j['job_id']}  {j['status']}" for j in status["jobs"]]
    return "\n".join(lines)


def cmd_cluster_status(args) -> int:
    print(format_status(_client(args).status()))
    return 0


def format_benchmark(b: dict) -> str:
    from meshtrain.profiler.network import format_matrix

    lines = [f"Benchmark status: {b.get('status')}", "",
             f"{'worker':<20}{'backend':<8}{'matmul GFLOP/s':>16}{'mlp fwd ms':>12}{'mlp bwd ms':>12}{'score':>8}"]
    for wid, r in sorted(b.get("workers", {}).items(), key=lambda kv: -kv[1]["compute_score"]):
        lines.append(f"{wid[:19]:<20}{r['backend']:<8}{r['matmul_gflops']:>16.1f}{r['mlp_forward_s'] * 1000:>12.2f}"
                     f"{r['mlp_backward_s'] * 1000:>12.2f}{r['compute_score']:>8.2f}")
    names = sorted(b.get("workers", {}))
    links = {tuple(k.split("->")): v for k, v in b.get("links", {}).items()}
    if names and links:
        lines += ["", "Bandwidth (row -> column):",
                  format_matrix(names, {k: v["bandwidth_Mbps"] for k, v in links.items()},
                                lambda v: "?" if v is None else f"{v:.0f}Mbps"),
                  "", "Latency (row -> column):",
                  format_matrix(names, {k: v["latency_s"] for k, v in links.items()},
                                lambda v: "?" if v is None else f"{v * 1000:.2f}ms")]
    for e in b.get("errors", []):
        lines.append(f"error: {e}")
    return "\n".join(lines)


def cmd_cluster_benchmark(args) -> int:
    c = _client(args)
    c.start_benchmark(pings=args.pings, payload_mb=args.payload_mb, network=not args.no_network)
    print("benchmarking cluster (compute + worker-to-worker network)...", flush=True)
    while True:
        b = c.benchmark()
        if b.get("status") != "RUNNING":
            break
        time.sleep(1)
    print(format_benchmark(b))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(b, f, indent=2)
    return 0 if not b.get("errors") else 1


# -- plan / train ----------------------------------------------------------
def _load_raw(path: str) -> dict:
    from meshtrain.config import load_config

    cfg = load_config(path)  # validates
    return cfg.model_dump(mode="json")


def cmd_plan(args) -> int:
    resp = _client(args).plan(_load_raw(args.config))
    print(resp["text"])
    return 0 if resp["plan"]["feasible"] else 1


def cmd_train(args) -> int:
    if args.local:
        return _train_local(args)
    c = _client(args)
    raw = _load_raw(args.config)
    print("planning model...", flush=True)
    job = c.start_job(raw)
    print(job["plan_text"])
    for s in job["plan"]["stages"]:
        print(f"Stage {s['stage']} -> {s['worker']} [{s['backend']}] layers {s['layers'][0]}-{s['layers'][1] - 1}")
    print(f"\ntraining started: job {job['job_id']}  (metrics -> {job['run_dir']}/metrics.jsonl)\n", flush=True)
    seen = 0
    try:
        while True:
            j = c.job(job["job_id"])
            for step, loss in j["losses"][seen:]:
                last = j["last_metrics"]
                comm = sum(m.get("comm_s", 0) for m in last.values())
                sent = sum(m.get("bytes_sent", 0) for m in last.values())
                if step % raw["training"]["log_every"] == 0 or step == raw["training"]["steps"] - 1:
                    print(f"step {step:<6} loss = {loss:.4f}   step {last.get('0', {}).get('step_s', 0) * 1000:.0f} ms"
                          f"   comm {comm * 1000:.0f} ms   sent {sent / 1e6:.2f} MB", flush=True)
            seen = len(j["losses"])
            if j["status"] != "RUNNING":
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping job...")
        c.stop_job(job["job_id"])
        return 130
    print(f"\njob {j['status']}" + (f": {j['error']}" if j.get("error") else ""))
    if j.get("summary"):
        from meshtrain.summary import format_summary

        print(format_summary(j["summary"]))
    return 0 if j["status"] == "COMPLETED" else 1


def _train_local(args) -> int:
    """Run all stages as local processes; placement from config (manual stages) or the planner
    with this machine's devices."""
    from meshtrain.config import load_config
    from meshtrain.runtime.local import LocalStage, run_local_pipeline
    from meshtrain.runtime.pipeline import PipelineSettings
    from meshtrain.summary import format_summary, summarize
    from meshtrain.telemetry import MetricsWriter, new_run_id

    cfg = load_config(args.config)
    n = cfg.model_num_layers()
    if cfg.placement.stages:
        stages = [LocalStage(tuple(s.layers), s.device or "cpu") for s in cfg.placement.stages]
    else:
        k = args.stages or cfg.placement.num_stages or 2
        stages = [LocalStage((n * i // k, n * (i + 1) // k), args.device) for i in range(k)]
    run_id = new_run_id(cfg.job.name)
    run_dir = os.path.join(cfg.job.runs_dir, run_id)
    print(f"local run {run_id}: " + " | ".join(f"stage{i} {s.device} layers {s.layers[0]}-{s.layers[1] - 1}"
                                               for i, s in enumerate(stages)), flush=True)
    settings = PipelineSettings(job_id=run_id, steps=cfg.training.steps, batch_size=cfg.training.batch_size,
                                num_microbatches=cfg.training.num_microbatches, timeout_s=cfg.network.timeout_s,
                                log_every=cfg.training.log_every)
    transport = "tcp" if cfg.network.tensor_transport == "tcp" else "pipe"
    os.environ.pop("MESHTRAIN_QUIET", None)
    results = run_local_pipeline(cfg.model.spec_kwargs(), stages, settings, seed=cfg.job.seed,
                                 optimizer=cfg.training.optimizer, lr=cfg.training.learning_rate,
                                 transport=transport, capture_params=True, torch_threads=args.threads)
    writer = MetricsWriter(os.path.join(run_dir, "metrics.jsonl"))
    for r in results:
        for rec in r.step_metrics:
            writer.write(rec)
    losses = results[0].losses
    for i, l in enumerate(losses):
        if i % cfg.training.log_every == 0 or i == len(losses) - 1:
            print(f"step {i:<6} loss {l:.4f}")
    summary = summarize([rec for r in results for rec in r.step_metrics])
    changed = {r.stage_index: all(not (r.initial_params[k] == r.final_params[k]).all() for k in r.initial_params)
               for r in results}
    summary["parameters_changed"] = changed
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + format_summary(summary))
    print(f"parameters changed per stage: {changed}")
    print(f"metrics: {run_dir}/metrics.jsonl")
    return 0 if summary["loss_decreased"] and all(changed.values()) else 1


def cmd_experiment(args) -> int:
    os.environ.setdefault("MESHTRAIN_QUIET", "1")
    if args.name == "correctness":
        from meshtrain.experiments.correctness import run_experiment1

        print(run_experiment1(args.transport))
    elif args.name == "placement":
        from meshtrain.experiments.placement import run_experiment4

        print(run_experiment4(cluster_file=args.cluster))
    elif args.name == "capacity":
        from meshtrain.experiments.capacity import run_experiment5

        print(run_experiment5(mode=args.mode, cluster_file=args.cluster, steps=args.steps))
    elif args.name == "transformer":
        from meshtrain.experiments.transformer import run_experiment_transformer

        print(run_experiment_transformer())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meshtrain", description="Heterogeneous pipeline-parallel training (V1)")
    p.add_argument("--token", help="cluster token (default $MESHTRAIN_TOKEN)")
    p.add_argument("--coordinator", help="coordinator HOST:PORT (default $MESHTRAIN_COORDINATOR or 127.0.0.1:8080)")
    sub = p.add_subparsers(dest="command", required=True)

    co = sub.add_parser("coordinator").add_subparsers(dest="action", required=True)
    s = co.add_parser("start")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--runs-dir", default="runs")
    s.add_argument("--heartbeat-timeout", type=float, default=15.0)
    s.set_defaults(func=cmd_coordinator_start)

    wo = sub.add_parser("worker").add_subparsers(dest="action", required=True)
    j = wo.add_parser("join")
    j.add_argument("coordinator_address")
    j.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    j.add_argument("--name")
    j.add_argument("--data-port", type=int, default=29500, help="data-plane TCP port (0 = any)")
    j.add_argument("--advertise-host", help="address peers should use to reach this worker")
    j.add_argument("--runs-dir", default="runs")
    j.add_argument("--quick-benchmark", action="store_true")
    j.set_defaults(func=cmd_worker_join)

    cl = sub.add_parser("cluster").add_subparsers(dest="action", required=True)
    cl.add_parser("status").set_defaults(func=cmd_cluster_status)
    b = cl.add_parser("benchmark")
    b.add_argument("--pings", type=int, default=10)
    b.add_argument("--payload-mb", type=float, default=16.0)
    b.add_argument("--no-network", action="store_true")
    b.add_argument("--output", help="write raw results JSON (input for `experiment placement --cluster`)")
    b.set_defaults(func=cmd_cluster_benchmark)

    pl = sub.add_parser("plan")
    pl.add_argument("config")
    pl.set_defaults(func=cmd_plan)

    t = sub.add_parser("train")
    t.add_argument("config")
    t.add_argument("--local", action="store_true", help="run every stage as a local process")
    t.add_argument("--stages", type=int, help="--local: number of equal stages if the config has none")
    t.add_argument("--device", default="cpu", help="--local: device for auto-split stages")
    t.add_argument("--threads", type=int, default=None, help="--local: torch threads per stage process")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("experiment")
    e.add_argument("name", choices=["correctness", "placement", "capacity", "transformer"])
    e.add_argument("--transport", default="tcp", choices=["pipe", "tcp"])
    e.add_argument("--mode", default="emulated", choices=["emulated", "hardware"],
                   help="capacity: emulated device budgets on CPU, or real devices of this cluster")
    e.add_argument("--cluster", help="JSON from `cluster benchmark --output` (else a documented example cluster)")
    e.add_argument("--steps", type=int, default=5)
    e.set_defaults(func=cmd_experiment)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # meaningful one-line errors for the user
        from meshtrain.config import ConfigError
        from meshtrain.networking.control import ControlError

        if isinstance(exc, (ControlError, ConfigError, FileNotFoundError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
