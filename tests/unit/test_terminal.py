import argparse
import io
import sys

from meshtrain import cli
from meshtrain.terminal import LOGO, MeshTrainConsole, banner, paint, table


def test_color_is_terminal_only_and_respects_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert paint("ONLINE", "good") == "ONLINE"
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert paint("ONLINE", "good") == "\033[32mONLINE\033[0m"
    monkeypatch.setenv("NO_COLOR", "")
    assert "\033" not in table(["WORKER", "STATUS"], [["gpu", "ONLINE"]], [10, 10])


def test_console_dispatch_preserves_paths_and_survives_errors(monkeypatch):
    calls = []

    def main(argv):
        calls.append(argv)
        if "bad-command" in argv or "--help" in argv:
            raise SystemExit(2 if "bad-command" in argv else 0)
        return 0

    monkeypatch.setattr(cli, "main", main)
    output = io.StringIO()
    shell = MeshTrainConsole(argparse.Namespace(coordinator="h:8090", token="tok"),
                             stdin=io.StringIO('status\n\ntrain "D:\\My configs\\model.yaml"\nbad-command\nhelp train\nhelp exit\nconsole\nexit\n'),
                             stdout=output)
    shell.use_rawinput = False
    shell.cmdloop()
    prefix = ["--coordinator", "h:8090", "--token", "tok"]
    assert calls == [prefix + ["status"], prefix + ["train", "D:\\My configs\\model.yaml"],
                     prefix + ["bad-command"], prefix + ["train", "--help"]]
    assert "already in" in output.getvalue() and "Console closed" in output.getvalue()
    assert LOGO in output.getvalue() and "running cluster continues" in output.getvalue()


def test_banner_is_ascii_and_fits_a_standard_terminal(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert banner().isascii()
    assert all(len(line) <= 80 for line in banner().splitlines())


def test_jobs_show_unique_steps_and_latest_loss():
    job = {"job_id": "test-run", "status": "RUNNING", "losses": [[0, 2.0], [1, 1.5], [0, 2.0]],
           "last_metrics": {"1": {"step_s": 0.3}, "0": {"step_s": 0.1}}}
    output = cli.format_jobs([job])
    assert "test-run" in output and "RUNNING" in output and "1.5" in output and "100.0" in output
    assert "No runs yet" in cli.format_jobs([])
