"""Run-level timeline artifacts: Chrome trace, per-worker breakdown, ASCII Gantt.

* ``write_run_timeline`` merges per-stage spans into ``runs/<id>/timeline.json``
  (Chrome Trace Event format: open in chrome://tracing or ui.perfetto.dev).
* ``worker_breakdown`` / ``format_breakdown``: per worker, the share of step
  time spent computing, communicating (and how much of that was exposed,
  i.e. not hidden behind compute) and idle.
* ``ascii_timeline``: one step drawn as text lanes (compute + transfer per
  stage) for terminals and docs.
"""

from __future__ import annotations

from pathlib import Path

from meshtrain.runtime.timeline import COMPUTE, COMM_WORK, chrome_events_from_export, write_chrome_trace

MAX_SPANS_PER_STAGE = 50_000


def write_run_timeline(path: str | Path, stages: list[tuple[int, str, list[dict]]]) -> Path:
    """``stages``: (stage_index, worker, exported spans)."""
    return write_chrome_trace(path, [chrome_events_from_export(spans[-MAX_SPANS_PER_STAGE:], idx, worker)
                                     for idx, worker, spans in stages])


def worker_breakdown(summary: dict) -> dict:
    out = {}
    step = summary.get("mean_step_s") or 0.0
    for idx, st in summary.get("stages", {}).items():
        if not step:
            continue
        comm = st.get("comm_s", 0.0)
        out[idx] = {"worker": st.get("worker"), "backend": st.get("backend"),
                    "compute": st.get("compute_s", 0.0) / step, "communication": comm / step,
                    "exposed_communication": st.get("exposed_communication_s", comm) / step,
                    "idle": st.get("idle_s", 0.0) / step, "overlap_ratio": st.get("overlap_ratio", 0.0)}
    return out


def format_breakdown(summary: dict) -> str:
    lines = []
    for idx, b in worker_breakdown(summary).items():
        lines += [f"Worker {b['worker']} (stage {idx}, {b['backend']})", "",
                  f"  compute               {b['compute']:6.0%}",
                  f"  communication         {b['communication']:6.0%}   (overlap ratio {b['overlap_ratio']:.0%})",
                  f"  exposed communication {b['exposed_communication']:6.0%}",
                  f"  idle                  {b['idle']:6.0%}", ""]
    return "\n".join(lines).rstrip()


_CHARS = {"FORWARD_COMPUTE": "F", "BACKWARD_COMPUTE": "B", "OPTIMIZER_STEP": "O", "NETWORK_SEND": ">",
          "NETWORK_RECV": "<", "D2H_COPY": "d", "H2D_COPY": "h", "SERIALIZE": "s", "DESERIALIZE": "s"}


def ascii_timeline(stages: list[tuple[int, str, list[dict]]], step: int, width: int = 96) -> str:
    """Two lanes per stage for one step: compute (F/B/O) and transfers (> send, < recv, d/h copies)."""
    spans = [(idx, s) for idx, _, ss in stages for s in ss if s.get("step") == step]
    if not spans:
        return f"(no spans for step {step})"
    t0 = min(s["start"] for _, s in spans)
    t1 = max(s["end"] for _, s in spans)
    scale = (width - 1) / max(t1 - t0, 1e-9)
    lines = [f"step {step}: {(t1 - t0) * 1000:.1f} ms   F forward  B backward  O optimizer  "
             "> send  < recv  d/h device copies  . idle"]
    for idx, worker, ss in stages:
        for lane, cats in (("compute ", COMPUTE), ("transfer", COMM_WORK)):
            row = ["."] * width
            for s in ss:
                if s.get("step") != step or s["category"] not in cats:
                    continue
                a, b = int((s["start"] - t0) * scale), int((s["end"] - t0) * scale)
                for i in range(a, max(b, a + 1)):
                    if 0 <= i < width:
                        row[i] = _CHARS.get(s["category"], "#")
            lines.append(f"stage {idx} {lane} |{''.join(row)}|")
    return "\n".join(lines)
