"""``meshtrain`` command-line interface.

Simple form:

    meshtrain start                   # first machine: coordinator + worker, prints a join code
    meshtrain join TOKEN@HOST         # every other machine
    meshtrain status | benchmark | train CONFIG

The last cluster started or joined is remembered in ~/.meshtrain/cluster.json.
Explicit form:

    meshtrain coordinator start [--port 8080]
    meshtrain worker join HOST:PORT [--device cuda|mps|cpu] [--name NAME]
    meshtrain cluster status
    meshtrain cluster benchmark
    meshtrain plan CONFIG
    meshtrain train CONFIG            # on the cluster via the coordinator
    meshtrain train CONFIG --local    # every stage as a local process (no coordinator)
    meshtrain experiment correctness|placement|capacity

Coordinator address and token come from --coordinator/--token, then
$MESHTRAIN_COORDINATOR/$MESHTRAIN_TOKEN, then the remembered cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import yaml

from meshtrain.terminal import banner, enable_windows_color, heading, paint, table, tools_help

DEFAULT_TOKEN = "meshtrain-dev-token"
DEFAULT_PORT = 8080


# -- remembered cluster (~/.meshtrain/cluster.json) ------------------------
def _state_path() -> str:
    home = os.environ.get("MESHTRAIN_HOME") or os.path.join(os.path.expanduser("~"), ".meshtrain")
    return os.path.join(home, "cluster.json")


def load_saved() -> dict:
    try:
        with open(_state_path()) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cluster(coordinator: str, token: str) -> None:
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"coordinator": coordinator, "token": token}, f)
    try:
        os.chmod(path, 0o600)  # holds the cluster token
    except OSError:
        pass


def parse_join_target(target: str) -> tuple[str, str | None]:
    """``TOKEN@HOST[:PORT]`` or ``HOST[:PORT]`` -> ("HOST:PORT", token or None)."""
    token = None
    if "@" in target:
        token, target = target.rsplit("@", 1)
        token = token or None
    target = target.removeprefix("http://")
    if not target:
        raise ValueError("missing coordinator address")
    if ":" not in target:
        target = f"{target}:{DEFAULT_PORT}"
    return target, token


def lan_ip() -> str:
    """Best guess of this machine's LAN address (no packets are sent)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def _token(args=None) -> str:
    tok = getattr(args, "token", None) or os.environ.get("MESHTRAIN_TOKEN") or load_saved().get("token")
    if not tok:
        print(f"warning: no --token / $MESHTRAIN_TOKEN given, using the insecure default {DEFAULT_TOKEN!r}",
              file=sys.stderr)
        tok = DEFAULT_TOKEN
    return tok


def _client(args=None):
    from meshtrain.networking.control import ControlClient

    address = (getattr(args, "coordinator", None) or os.environ.get("MESHTRAIN_COORDINATOR") or load_saved().get("coordinator")
               or f"127.0.0.1:{DEFAULT_PORT}")
    return ControlClient(address, _token(args))


def _gb(n) -> str:
    return f"{n / 1024**3:.1f} GB" if n else "-"


# -- coordinator / worker --------------------------------------------------
def cmd_coordinator_start(args) -> int:
    from meshtrain.coordinator.server import run_coordinator

    run_coordinator(args.host, args.port, token=_token(args), runs_dir=args.runs_dir,
                    heartbeat_timeout_s=args.heartbeat_timeout)
    return 0


def _run_worker(args, address: str, token: str) -> int:
    from meshtrain.worker.worker import WorkerAgent

    agent = WorkerAgent(address, token, device=args.device, name=args.name,
                        data_port=args.data_port, advertise_host=args.advertise_host, runs_dir=args.runs_dir,
                        quick_benchmark=args.quick_benchmark)
    print(f"MeshTrain worker {agent.name}: backend={agent.device.backend} device={agent.device.name()} "
          f"data-plane port={agent.dataplane.port}  (Ctrl-C to leave)", flush=True)
    try:
        agent.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
    return 0


def cmd_worker_join(args) -> int:
    address, code_token = parse_join_target(args.coordinator_address)
    token = args.token or code_token or os.environ.get("MESHTRAIN_TOKEN") or load_saved().get("token")
    if not token:
        print("error: no cluster token. Use the join code printed by `meshtrain start` "
              "(meshtrain join TOKEN@HOST) or pass --token.", file=sys.stderr)
        return 2
    save_cluster(address, token)  # so `meshtrain status` / `train` work here without flags
    return _run_worker(args, address, token)


def cmd_start(args) -> int:
    """Coordinator (subprocess) + a worker for this machine, with a generated token."""
    import secrets
    import subprocess

    from meshtrain.networking.control import ControlClient, ControlError

    print(banner() + "\n", flush=True)
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"  # no look-alikes, never starts with "-"
    token = (args.token or os.environ.get("MESHTRAIN_TOKEN")
             or "".join(secrets.choice(alphabet) for _ in range(10)))
    env = {**os.environ, "MESHTRAIN_TOKEN": token}
    coord = subprocess.Popen([sys.executable, "-m", "meshtrain.cli", "coordinator", "start", "--host", args.host,
                              "--port", str(args.port), "--runs-dir", args.runs_dir], env=env)
    local = f"127.0.0.1:{args.port}"
    client = ControlClient(local, token)
    for _ in range(100):
        if coord.poll() is not None:
            print(f"error: coordinator exited (is port {args.port} already in use?)", file=sys.stderr)
            return 1
        try:
            client.status()
            break
        except ControlError:
            time.sleep(0.1)
    save_cluster(local, token)
    host = args.advertise_host or lan_ip()
    args.advertise_host = host  # remote stages must dial this machine's LAN address, not 127.0.0.1
    port_part = "" if args.port == DEFAULT_PORT else f":{args.port}"
    print(f"\n{heading('MeshTrain cluster started')}\n\n  {paint('On every other machine run:', 'accent')}\n\n"
          f"      meshtrain join {token}@{host}{port_part}\n\n"
          f"  Then, on any machine in the cluster:\n"
          f"      meshtrain console\n      meshtrain dashboard\n      meshtrain train configs/local_cpu.yaml\n", flush=True)
    try:
        if args.no_worker:
            coord.wait()
            return coord.returncode or 0
        return _run_worker(args, local, token)
    except KeyboardInterrupt:
        return 0
    finally:
        coord.terminate()
        try:
            coord.wait(10)
        except subprocess.TimeoutExpired:
            coord.kill()


# -- cluster -----------------------------------------------------------------
def format_status(status: dict) -> str:
    workers = status["workers"]
    online = sum(w["status"] in ("ONLINE", "BUSY") for w in workers)
    lines = [heading("MeshTrain Cluster", f"{online}/{len(workers)} workers online"), ""]
    rows = []
    unified = False
    for w in workers:
        dev = w.get("device") or {}
        mem = _gb(dev.get("memory_total")) if w["backend"] != "cpu" else "-"
        if dev.get("unified_memory"):
            mem += "*"
            unified = True
        score = (w.get("benchmark") or {}).get("compute_score")
        rows.append([w['worker_id'], w['backend'].upper(), str(dev.get('name', '')), mem,
                     _gb(w.get('ram_total')), f'{score:.2f}' if score is not None else '-', w['status']])
    if shutil.get_terminal_size((110, 24)).columns >= 100:
        lines.append(table(["WORKER", "BACKEND", "DEVICE", "MEMORY", "RAM", "SCORE", "STATUS"],
                           rows, [16, 7, 25, 11, 9, 6, 9]))
    else:
        lines.append(table(["WORKER", "BACKEND", "DEVICE", "STATUS"],
                           [[r[0], r[1], r[2], r[6]] for r in rows], [16, 7, 26, 9]))
        lines += [paint(f"  {r[0]}: device memory {r[3]} | RAM {r[4]} | score {r[5]}", "dim") for r in rows]
    if not workers:
        lines.append(paint("No workers yet. Run meshtrain join CODE on another machine.", "dim"))
    for w in workers:
        if w.get("current_job"):
            lines.append(f"  {w['worker_id']} -> {paint(w['current_job'], 'warn')}")
    if unified:
        lines += ["", paint("* unified memory", "dim")]
    if status.get("jobs"):
        lines += ["", heading("Jobs")] + [f"  {j['job_id']}  {paint(j['status'], 'good' if j['status'] == 'COMPLETED' else 'dim')}" for j in status["jobs"]]
    lines += ["", paint("jobs: metrics  |  dashboard: charts  |  console: tools", "dim")]
    return "\n".join(lines)


def cmd_cluster_status(args) -> int:
    client = _client(args)
    with client.http:
        print(format_status(client.status()))
        print(paint(f"Coordinator: {client.base}", "dim"))
    return 0


def format_jobs(jobs: list[dict]) -> str:
    rows = []
    for job in sorted(jobs, key=lambda j: j.get("created_at", 0), reverse=True):
        losses = dict(job.get("losses", []))
        metrics = job.get("last_metrics", {})
        first = metrics[min(metrics, key=int)] if metrics else {}
        step_s = first.get("step_s", (job.get("summary") or {}).get("mean_step_s"))
        rows.append([job["job_id"], job["status"], str(len(losses)),
                     f"{losses[max(losses)]:.5g}" if losses else "-", f"{step_s * 1000:.1f}" if step_s is not None else "-"])
    lines = [heading("Training runs", f"{len(jobs)} recorded"), "",
             table(["RUN", "STATUS", "STEPS", "LOSS", "STEP ms"], rows, [32, 10, 6, 10, 10])]
    if not jobs:
        lines.append(paint("No runs yet. Start one with meshtrain train CONFIG.", "dim"))
    for job in jobs:
        if job.get("error"):
            lines.append(paint(f"  {job['job_id']}: {job['error']}", "bad"))
    lines += ["", paint("Latest first-stage timing. Use dashboard for loss history.", "dim")]
    return "\n".join(lines)


def cmd_jobs(args) -> int:
    client = _client(args)
    with client.http:
        print(format_jobs(client.jobs()["jobs"]))
    return 0


def cmd_console(args) -> int:
    from meshtrain.terminal import run_console

    run_console(args)
    return 0


def cmd_dashboard(args) -> int:
    from meshtrain.dashboard import run_dashboard

    run_dashboard(_client(args), args.port)
    return 0


def cmd_benchmark(args) -> int:
    if args.what == "cluster":
        return cmd_cluster_benchmark(args)
    os.environ.setdefault("MESHTRAIN_QUIET", "1")
    from meshtrain.experiments import pipeline_bench

    if args.cluster:
        if not args.config:
            print("error: --cluster needs --config", file=sys.stderr)
            return 2
        r = pipeline_bench.run_cluster_benchmark(_client(args), _load_raw(args.config))
        print(r["markdown"])
        return 0
    model = None
    batch, mbs = args.batch_size, args.microbatches
    if args.config:
        from meshtrain.config import load_config

        cfg = load_config(args.config)
        model, batch, mbs = cfg.model.spec_kwargs(), cfg.training.batch_size, cfg.training.num_microbatches
    devices = args.devices.split(",") if args.devices else None
    r = pipeline_bench.run_local_benchmark(model, num_stages=len(devices) if devices else args.stages,
                                           batch_size=batch, num_microbatches=mbs, steps=args.steps,
                                           bandwidth_mbps=args.bandwidth_mbps or None, latency_ms=args.latency_ms,
                                           devices=devices)
    print(r["markdown"])
    print(f"\nreport: docs/v1-vs-v1.5.md   timelines: {r['run_dir']}/timeline-*.json (chrome://tracing, Perfetto)")
    return 0


def format_benchmark(b: dict) -> str:
    from meshtrain.profiler.network import format_matrix

    lines = [heading("Cluster benchmark", f"status: {b.get('status')}"), ""]
    rows = []
    for wid, r in sorted(b.get("workers", {}).items(), key=lambda kv: -kv[1]["compute_score"]):
        rows.append([wid, r['backend'].upper(), f"{r['matmul_gflops']:.1f}", f"{r['mlp_forward_s'] * 1000:.2f}",
                     f"{r['mlp_backward_s'] * 1000:.2f}", f"{r['compute_score']:.2f}"])
    lines.append(table(["WORKER", "BACKEND", "MATMUL GFLOP/s", "FWD ms", "BWD ms", "SCORE"],
                       rows, [16, 7, 14, 12, 12, 7]))
    names = sorted(b.get("workers", {}))
    links = {tuple(k.split("->")): v for k, v in b.get("links", {}).items()}
    if names and links:
        lines += ["", heading("Bandwidth", "row -> column"),
                  format_matrix(names, {k: v["bandwidth_Mbps"] for k, v in links.items()},
                                lambda v: "?" if v is None else f"{v:.0f}Mbps"),
                  "", heading("Latency", "row -> column"),
                  format_matrix(names, {k: v["latency_s"] for k, v in links.items()},
                                lambda v: "?" if v is None else f"{v * 1000:.2f}ms")]
    for e in b.get("errors", []):
        lines.append(paint(f"error: {e}", "bad"))
    return "\n".join(lines)


def cmd_cluster_benchmark(args) -> int:
    c = _client(args)
    c.start_benchmark(pings=args.pings, payload_mb=args.payload_mb, network=not args.no_network)
    print(heading("Benchmarking cluster", "compute + worker-to-worker network"), flush=True)
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
    return cfg.model_dump(mode="json", by_alias=True)


def cmd_plan(args) -> int:
    resp = _client(args).plan(_load_raw(args.config))
    print(resp["text"])
    return 0 if resp["plan"]["feasible"] else 1


def cmd_inspect_placement(args) -> int:
    from meshtrain.config import load_config
    from meshtrain.planner.report import format_placement

    cfg = load_config(args.config)
    resp = _client(args).plan(cfg.model_dump(mode="json", by_alias=True))
    spec = cfg.build_model_spec()
    print(format_placement(resp["plan"], [spec.layer_name(i) for i in range(spec.num_layers)]))
    return 0 if resp["plan"]["feasible"] else 1


def cmd_train(args) -> int:
    if args.local:
        return _train_local(args)
    c = _client(args)
    raw = _load_raw(args.config)
    print(heading("Planning training", args.config), flush=True)
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
    print("\n" + heading("Training " + j['status']) + (paint(f": {j['error']}", "bad") if j.get("error") else ""))
    if j.get("summary"):
        from meshtrain.summary import format_summary

        print(format_summary(j["summary"]))
    return 0 if j["status"] == "COMPLETED" else 1


def _train_local(args) -> int:
    """Run all stages as local processes; placement from config (manual stages) or the planner
    with this machine's devices."""
    from meshtrain.config import load_config
    from meshtrain.runtime.local import LocalStage, run_local_pipeline
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
    settings = cfg.pipeline_settings(run_id)
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
    from meshtrain.runtime.trace_report import ascii_timeline, format_breakdown, write_run_timeline

    lanes = [(r.stage_index, r.worker, r.timeline or []) for r in results]
    trace = write_run_timeline(os.path.join(run_dir, "timeline.json"), lanes)
    print("\n" + format_summary(summary))
    print("\n" + format_breakdown(summary))
    print("\n" + ascii_timeline(lanes, step=len(losses) - 1))
    print(f"parameters changed per stage: {changed}")
    print(f"timeline: {trace} (chrome://tracing or ui.perfetto.dev)")
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

        client = _client(args) if args.mode == "hardware" else None  # same cluster/token as `meshtrain status`
        print(run_experiment5(mode=args.mode, cluster_file=args.cluster, steps=args.steps, client=client))
    elif args.name == "device-correctness":
        from meshtrain.experiments.device_correctness import format_device_correctness, run_device_correctness

        r = run_device_correctness(args.devices.split(","), steps=args.steps)
        print(format_device_correctness(r))
        return 0 if r["passed"] else 1
    elif args.name == "transformer":
        from meshtrain.experiments.transformer import run_experiment_transformer

        print(run_experiment_transformer())
    return 0


def cmd_results_record(args) -> int:
    from meshtrain.experiments.record import record_run

    status = None
    try:
        status = _client(args).status()
    except Exception:
        pass  # coordinator not reachable: record without device names
    print(record_run(args.run_dir, args.section, reference_steps=args.reference_steps, cluster_status=status))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # --token / --coordinator are accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--token", default=argparse.SUPPRESS, help="cluster token (default $MESHTRAIN_TOKEN)")
    common.add_argument("--coordinator", default=argparse.SUPPRESS,
                        help="coordinator HOST:PORT (default $MESHTRAIN_COORDINATOR or 127.0.0.1:8080)")
    p = argparse.ArgumentParser(prog="meshtrain", description=heading("MeshTrain", "Heterogeneous pipeline training"),
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=tools_help() + "\n\nRun meshtrain console for an interactive command prompt.")
    # Separate actions (not parents=[common]): set_defaults would mutate the shared actions.
    p.add_argument("--token", default=None, help="cluster token (default $MESHTRAIN_TOKEN)")
    p.add_argument("--coordinator", default=None, help="coordinator HOST:PORT")
    sub = p.add_subparsers(dest="command", required=True, title="tools", metavar="COMMAND")

    co = sub.add_parser("coordinator").add_subparsers(dest="action", required=True)
    s = co.add_parser("start", parents=[common])
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--runs-dir", default="runs")
    s.add_argument("--heartbeat-timeout", type=float, default=15.0)
    s.set_defaults(func=cmd_coordinator_start)

    def worker_args(j):
        j.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
        j.add_argument("--name", help="worker name (default: hostname)")
        j.add_argument("--data-port", type=int, default=29500, help="data-plane TCP port (0 = any)")
        j.add_argument("--advertise-host", help="address peers should use to reach this machine")
        j.add_argument("--runs-dir", default="runs")
        j.add_argument("--quick-benchmark", action="store_true")

    # Short forms: `meshtrain start`, `meshtrain join CODE`, `meshtrain status`, `meshtrain benchmark`.
    st = sub.add_parser("start", parents=[common],
                        help="start a cluster on this machine (coordinator + local worker) and print the join code")
    worker_args(st)
    st.add_argument("--host", default="0.0.0.0", help="coordinator bind address")
    st.add_argument("--port", type=int, default=DEFAULT_PORT)
    st.add_argument("--no-worker", action="store_true", help="run only the coordinator here")
    st.set_defaults(func=cmd_start)

    jn = sub.add_parser("join", parents=[common], help="join a cluster: meshtrain join TOKEN@HOST[:PORT]")
    jn.add_argument("coordinator_address", metavar="TOKEN@HOST[:PORT]")
    worker_args(jn)
    jn.set_defaults(func=cmd_worker_join)

    sub.add_parser("status", parents=[common], help="show cluster workers").set_defaults(func=cmd_cluster_status)
    sub.add_parser("jobs", parents=[common], help="show training status, loss, and step timings").set_defaults(func=cmd_jobs)
    sub.add_parser("console", parents=[common], help="open the interactive tools and command prompt").set_defaults(func=cmd_console)

    dashboard = sub.add_parser("dashboard", parents=[common], help="view live training metrics in a local browser")
    dashboard.add_argument("--port", type=int, default=8081, help="local dashboard port (default: 8081)")
    dashboard.set_defaults(func=cmd_dashboard)

    wo = sub.add_parser("worker").add_subparsers(dest="action", required=True)
    j = wo.add_parser("join", parents=[common])
    j.add_argument("coordinator_address", metavar="[TOKEN@]HOST[:PORT]")
    worker_args(j)
    j.set_defaults(func=cmd_worker_join)

    cl = sub.add_parser("cluster").add_subparsers(dest="action", required=True)
    cl.add_parser("status", parents=[common]).set_defaults(func=cmd_cluster_status)
    for b in (cl.add_parser("benchmark", parents=[common]),
              sub.add_parser("benchmark", parents=[common],
                             help="`benchmark` / `benchmark cluster`: compute + network of the cluster; "
                                  "`benchmark pipeline`: V1 vs V1.5 execution modes")):
        b.add_argument("--pings", type=int, default=10)
        b.add_argument("--payload-mb", type=float, default=16.0)
        b.add_argument("--no-network", action="store_true")
        b.add_argument("--output", help="write raw results JSON (input for `experiment placement --cluster`)")
        b.set_defaults(func=cmd_cluster_benchmark)
    b.add_argument("what", nargs="?", default="cluster", choices=["cluster", "pipeline"])
    b.add_argument("--config", help="pipeline: model/training config (default: built-in tiny Transformer)")
    b.add_argument("--stages", type=int, default=3, help="pipeline (local): number of stage processes")
    b.add_argument("--devices", help="pipeline (local): one device per stage, e.g. cuda,cuda")
    b.add_argument("--microbatches", type=int, default=8)
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--steps", type=int, default=8)
    b.add_argument("--bandwidth-mbps", type=float, default=100.0,
                   help="pipeline (local): emulated link bandwidth; 0 = raw loopback")
    b.add_argument("--latency-ms", type=float, default=1.0)
    b.add_argument("--cluster", action="store_true", help="pipeline: run the modes as jobs on the real cluster")
    b.set_defaults(func=cmd_benchmark)

    pl = sub.add_parser("plan", parents=[common], help="preview model placement")
    pl.add_argument("config")
    pl.set_defaults(func=cmd_plan)

    ins = sub.add_parser("inspect").add_subparsers(dest="action", required=True)
    ip = ins.add_parser("placement", parents=[common], help="detailed placement plan for a config")
    ip.add_argument("config")
    ip.set_defaults(func=cmd_inspect_placement)

    t = sub.add_parser("train", parents=[common], help="train a model from a YAML config")
    t.add_argument("config")
    t.add_argument("--local", action="store_true", help="run every stage as a local process")
    t.add_argument("--stages", type=int, help="--local: number of equal stages if the config has none")
    t.add_argument("--device", default="cpu", help="--local: device for auto-split stages")
    t.add_argument("--threads", type=int, default=None, help="--local: torch threads per stage process")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("results").add_subparsers(dest="action", required=True)
    rr = r.add_parser("record", parents=[common], help="add a finished cluster run to docs/v1-results.md")
    rr.add_argument("run_dir")
    rr.add_argument("--section", required=True, help="e.g. experiment2 or experiment3")
    rr.add_argument("--reference-steps", type=int, default=20)
    rr.set_defaults(func=cmd_results_record)

    e = sub.add_parser("experiment", parents=[common],
                       help="run correctness, placement, capacity, or device experiments")
    e.add_argument("name", choices=["correctness", "placement", "capacity", "transformer", "device-correctness"])
    e.add_argument("--devices", default="cpu,cpu", help="device-correctness: one device per stage, e.g. cuda,cuda,mps")
    e.add_argument("--transport", default="tcp", choices=["pipe", "tcp"])
    e.add_argument("--mode", default="emulated", choices=["emulated", "hardware"],
                   help="capacity: emulated device budgets on CPU, or real devices of this cluster")
    e.add_argument("--cluster", help="JSON from `cluster benchmark --output` (else a documented example cluster)")
    e.add_argument("--steps", type=int, default=5)
    e.set_defaults(func=cmd_experiment)
    return p


def main(argv: list[str] | None = None) -> int:
    enable_windows_color()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # meaningful one-line errors for the user
        from meshtrain.config import ConfigError
        from meshtrain.networking.control import ControlError

        if isinstance(exc, (ControlError, ConfigError, FileNotFoundError)):
            print(paint(f"error: {exc}", "bad"), file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
