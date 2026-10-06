"""`meshtrain start` on one "machine" + `meshtrain join CODE` on another (separate MESHTRAIN_HOME)."""

import os
import re
import socket
import subprocess
import sys
import time

from meshtrain.networking.control import ControlClient


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_prints_join_code_and_join_works(tmp_path):
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k not in ("MESHTRAIN_TOKEN", "MESHTRAIN_COORDINATOR")}
    env["OMP_NUM_THREADS"] = "1"
    common = ["--data-port", "0", "--advertise-host", "127.0.0.1", "--quick-benchmark", "--runs-dir",
              str(tmp_path / "runs")]
    a = subprocess.Popen([sys.executable, "-m", "meshtrain.cli", "start", "--port", str(port), "--name", "a", *common],
                         env={**env, "MESHTRAIN_HOME": str(tmp_path / "a")}, stdout=subprocess.PIPE, text=True)
    procs = [a]
    try:
        code = None
        t0 = time.time()
        while code is None and time.time() - t0 < 30:
            line = a.stdout.readline()
            m = re.search(r"meshtrain join (\S+)", line)
            if m:
                code = m.group(1)
        assert code and "@" in code
        token = code.split("@")[0]
        assert re.fullmatch(r"[a-z0-9]{10}", token)
        procs.append(subprocess.Popen([sys.executable, "-m", "meshtrain.cli", "join", code, "--name", "b", *common],
                                      env={**env, "MESHTRAIN_HOME": str(tmp_path / "b")},
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        client = ControlClient(f"127.0.0.1:{port}", token)
        t0 = time.time()
        while time.time() - t0 < 30:
            names = {w["worker_id"] for w in client.status()["workers"] if w["status"] == "ONLINE"}
            if names == {"a", "b"}:
                break
            time.sleep(0.3)
        assert names == {"a", "b"}
        # the joining machine remembered the cluster
        assert token in (tmp_path / "b" / "cluster.json").read_text()
    finally:
        import psutil

        # `start` runs the coordinator as a child process; on Windows terminate() does not reach it.
        for p in procs[::-1]:
            try:
                children = psutil.Process(p.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            p.terminate()
            for c in children:
                try:
                    c.kill()
                except psutil.NoSuchProcess:
                    pass
        for p in procs:
            p.wait(15)
