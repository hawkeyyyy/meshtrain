"""Small terminal presentation helpers and an interactive shell using the standard library."""

import cmd
import os
import shlex
import sys

TONES = {"accent": "36", "good": "32", "warn": "33", "bad": "31", "dim": "90", "bold": "1"}
STATUS_TONES = {"ONLINE": "good", "COMPLETED": "good", "RUNNING": "warn", "BUSY": "warn",
                "FAILED": "bad", "OFFLINE": "bad", "STOPPED": "dim", "PLANNED": "dim"}
LOGO = r""" __  __ _____ ____  _   _ _____ ____      _    ___ _   _
|  \/  | ____/ ___|| | | |_   _|  _ \    / \  |_ _| \ | |
| |\/| |  _| \___ \| |_| | | | | |_) |  / _ \  | ||  \| |
| |  | | |___ ___) |  _  | | | |  _ <  / ___ \ | || |\  |
|_|  |_|_____|____/|_| |_| |_| |_| \_\/_/   \_\___|_| \_|"""


def paint(text: str, tone: str = "accent") -> str:
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        return f"\033[{TONES[tone]}m{text}\033[0m"
    return text


def enable_windows_color() -> None:
    if os.name == "nt" and sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 4)


def heading(title: str, detail: str = "") -> str:
    return paint(title, "bold") + ("  " + paint(detail, "dim") if detail else "")


def banner() -> str:
    return paint(LOGO, "accent") + "\n" + paint("  Heterogeneous pipeline training", "dim")


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    def row(values, header=False):
        cells = []
        for value, width in zip(values, widths):
            text = "".join(c for c in str(value) if c.isprintable())
            if len(text) > width:
                text = text[:width - 3] + "..."
            padded = f"{text:<{width}}"
            tone = "accent" if header else STATUS_TONES.get(text)
            if text in ("CUDA", "MPS", "CPU"):
                tone = "accent"
            cells.append(paint(padded, tone) if tone else padded)
        return "  ".join(cells).rstrip()

    lines = [row(headers, True), paint("  ".join("-" * width for width in widths), "dim")]
    lines.extend(row(values) for values in rows)
    return "\n".join(lines)


def tools_help() -> str:
    groups = [
        ("CLUSTER", [("start", "Start a coordinator and local worker"),
                     ("join CODE", "Connect another machine"),
                     ("status", "Show workers and available devices"),
                     ("benchmark", "Measure compute and network performance"),
                     ("benchmark pipeline", "Compare V1 and V1.5 pipeline modes")]),
        ("TRAINING", [("plan CONFIG", "Preview model placement"),
                      ("inspect placement CONFIG", "Detailed stages, memory, and communication"),
                      ("train CONFIG", "Run training on the connected cluster"),
                      ("experiment capacity --mode hardware", "Measure real cluster capacity"),
                      ("experiment device-correctness", "Check per-device gradients against CPU")]),
        ("MONITOR", [("jobs", "Inspect run status, loss, and step time"),
                     ("dashboard", "Serve live browser metrics on port 8081")]),
    ]
    lines = []
    for name, tools in groups:
        lines.extend([paint(name, "accent"), *[f"  {paint(f'{command:<38}', 'bold')}{paint(help_text, 'dim')}"
                                             for command, help_text in tools], ""])
    return "\n".join(lines).rstrip()


class MeshTrainConsole(cmd.Cmd):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.connection_flags = []
        for name in ("coordinator", "token"):
            value = getattr(args, name, None)
            if value:
                self.connection_flags.extend([f"--{name}", value])
        self.prompt = paint("meshtrain", "accent") + paint(" > ", "dim")
        self.intro = (banner() + "\n\n" + heading("MeshTrain console", "type a command below") + "\n\n" + tools_help()
                      + "\n\n" + paint("help COMMAND for options · exit to leave · Ctrl-C cancels a command", "dim"))

    def emptyline(self):
        pass  # Blank input must not repeat a training or benchmark command.

    def default(self, line):
        from meshtrain.cli import main

        try:
            # On Windows, keep backslashes in paths; remove surrounding quotes after splitting.
            argv = shlex.split(line, posix=os.name != "nt")
            if os.name == "nt":
                argv = [word[1:-1] if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'" else word
                        for word in argv]
            if argv[:3] == ["uv", "run", "meshtrain"]:
                argv = argv[3:]
            elif argv[:1] == ["meshtrain"]:
                argv = argv[1:]
            if not argv:
                return
            if argv[0] == "console":
                self.stdout.write("You are already in the MeshTrain console.\n")
                return
            main([*self.connection_flags, *argv])
        except ValueError as exc:
            self.stdout.write(paint(f"error: {exc}", "bad") + "\n")
        except SystemExit:
            pass  # argparse help/errors should return to the prompt.
        except KeyboardInterrupt:
            self.stdout.write("\n" + paint("Command interrupted. Ready for the next command.", "warn") + "\n")
        except Exception as exc:
            self.stdout.write(paint(f"error: {type(exc).__name__}: {exc}", "bad") + "\n")

    def do_help(self, topic):
        if topic in ("exit", "quit", "EOF"):
            self.stdout.write(self.do_exit.__doc__ + "\n")
        elif topic:
            self.default(topic + " --help")
        else:
            self.stdout.write(tools_help() + "\n\nhelp COMMAND for options · exit to leave\n")

    def do_exit(self, line):
        """Leave the console. The running cluster continues in its own process."""
        self.stdout.write(paint("Console closed.", "dim") + "\n")
        return True

    do_quit = do_exit
    do_EOF = do_exit


def run_console(args) -> None:
    console = MeshTrainConsole(args)
    while True:
        try:
            console.cmdloop()
            return
        except KeyboardInterrupt:
            console.intro = None
            print("\n" + paint("Type exit to leave the console.", "dim"))
