"""Start a coordinator plus N CPU workers on this machine (development / demos).

    uv run python scripts/local_cluster.py --workers 3 --port 8090
    MESHTRAIN_TOKEN=dev uv run meshtrain --coordinator 127.0.0.1:8090 cluster status

All processes are stopped with Ctrl-C. Worker names are labels only; every
worker here is a CPU process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default=os.environ.get("MESHTRAIN_TOKEN", "dev"))
    p.add_argument("--names", default="", help="comma-separated worker names")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()
    env = {**os.environ, "MESHTRAIN_TOKEN": args.token, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    cli = [sys.executable, "-m", "meshtrain.cli"]
    procs = [subprocess.Popen(cli + ["coordinator", "start", "--port", str(args.port), "--runs-dir", args.runs_dir],
                              env=env)]
    time.sleep(2)
    names = [n for n in args.names.split(",") if n] or [f"worker{i}" for i in range(args.workers)]
    for name in names:
        procs.append(subprocess.Popen(cli + ["worker", "join", f"127.0.0.1:{args.port}", "--device", "cpu",
                                             "--name", name, "--data-port", "0", "--advertise-host", "127.0.0.1",
                                             "--quick-benchmark", "--runs-dir", args.runs_dir], env=env))
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
