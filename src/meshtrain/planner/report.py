"""Human-readable placement plans and planner prediction accuracy."""

from __future__ import annotations

GB = 1024**3


def _gb(n) -> str:
    return f"{n / GB:.2f} GB"


def _mb(n) -> str:
    return f"{n / 1e6:.1f} MB"


def _ms(s) -> str:
    return f"{s * 1000:.2f} ms"


def format_placement(plan: dict, layer_names: list[str] | None = None) -> str:
    """``meshtrain inspect placement``: per-stage memory breakdown and per-boundary transfer cost."""
    lines = [f"MeshTrain Placement Plan ({plan['strategy']})", ""]
    for s in plan["stages"]:
        m = s["memory"]
        a, b = s["layers"]
        names = f" ({layer_names[a]} .. {layer_names[b - 1]})" if layer_names else ""
        lines += [
            f"Stage {s['stage']}",
            f"  worker       {s['worker']}",
            f"  backend      {s['backend'].upper()}",
            f"  layers       {a}-{b - 1}{names}",
            f"  parameters   {_gb(m['parameters'])}",
            f"  gradients    {_gb(m['gradients'])}",
            f"  optimizer    {_gb(m['optimizer_state'] + m.get('optimizer_step_temporary', 0))}"
            f"  (state {_gb(m['optimizer_state'])} + step temporaries {_gb(m.get('optimizer_step_temporary', 0))})",
            f"  activations  {_gb(m['saved_activations'])}  ({m.get('in_flight', '?')} microbatches in flight)",
            f"  buffers      {_gb(m.get('input_buffers', 0) + m.get('output_buffers', 0) + m.get('transport_buffers', 0))}",
            f"  workspace    {_gb(m.get('temporary_workspace', 0))}",
            f"  estimated    {_gb(m['required'])}",
            f"  budget       {_gb(m['budget'])}  of {_gb(m['total'])} ({'fits' if m['fits'] else 'OVER BUDGET'})",
            f"  compute      {_ms(s['compute_s_per_mb'])}/microbatch, communication {_ms(s['comm_s_per_mb'])}/microbatch"
            + (" (overlapped)" if s.get("overlap") else " (blocking)"),
            "",
        ]
    for bd in plan.get("boundaries", []):
        f, bwd = bd.get("forward", {}), bd.get("backward", {})
        lines += [
            f"Boundary {bd['from_stage']} -> {bd['to_stage']}",
            f"  activation    {_mb(bd['activation_bytes'])}",
            f"  gradient      {_mb(bd['gradient_bytes'])}",
            f"  forward       D2H {_ms(f.get('d2h_s', 0))} + net {_ms(f.get('network_s', 0))} + "
            f"H2D {_ms(f.get('h2d_s', 0))} = {_ms(f.get('total_s', 0))}",
            f"  backward      D2H {_ms(bwd.get('d2h_s', 0))} + net {_ms(bwd.get('network_s', 0))} + "
            f"H2D {_ms(bwd.get('h2d_s', 0))} = {_ms(bwd.get('total_s', 0))}",
            "",
        ]
    if plan["stages"]:
        b = max(plan["stages"], key=lambda s: s.get("time_per_mb", s["compute_s_per_mb"] + s["comm_s_per_mb"]))
        lines += ["Predicted bottleneck:", f"  stage {b['stage']} ({b['worker']})",
                  f"  step time {_ms(plan['predicted_step_s'])} (M={plan['num_microbatches']}, S={len(plan['stages'])})",
                  f"  network/step {_mb(plan['communication_bytes_per_step'])}"]
    for w, why in plan.get("excluded", {}).items():
        lines.append(f"  excluded: {w} ({why})")
    for v in plan.get("violations", []):
        lines.append(f"  VIOLATION: {v}")
    return "\n".join(lines)


def prediction_accuracy(plan: dict, summary: dict, memory_records: dict | None = None) -> dict:
    """Predicted vs actual per stage (time and memory): error = |pred - actual| / actual."""
    M = plan["num_microbatches"]
    out = {"stages": {}, "step": None}

    def err(p, a):
        return abs(p - a) / a if a else None

    for s in plan["stages"]:
        st = summary.get("stages", {}).get(s["stage"]) or summary.get("stages", {}).get(str(s["stage"])) or {}
        actual = st.get("compute_s") or (st.get("forward_s", 0) + st.get("backward_s", 0))
        pred = s["compute_s_per_mb"] * M
        row = {"predicted_compute_s": pred, "actual_compute_s": actual, "compute_error": err(pred, actual)}
        mv = (memory_records or {}).get(str(s["stage"])) or {}
        peak = mv.get("actual_peak_step0")
        if peak:
            row.update(predicted_memory=s["memory"]["required"], actual_memory=peak,
                       memory_error=err(s["memory"]["required"], peak))
        out["stages"][s["stage"]] = row
    if summary.get("mean_step_s"):
        out["step"] = {"predicted_s": plan["predicted_step_s"], "actual_s": summary["mean_step_s"],
                       "error": err(plan["predicted_step_s"], summary["mean_step_s"])}
    return out


def format_accuracy(acc: dict) -> str:
    lines = ["planner prediction accuracy:"]
    if acc.get("step"):
        st = acc["step"]
        lines.append(f"  step time   predicted {_ms(st['predicted_s'])}   actual {_ms(st['actual_s'])}   "
                     f"error {st['error']:.1%}")
    for idx, r in acc["stages"].items():
        line = f"  stage {idx}     compute predicted {_ms(r['predicted_compute_s'])} actual {_ms(r['actual_compute_s'])}"
        if r.get("compute_error") is not None:
            line += f" (error {r['compute_error']:.1%})"
        if r.get("memory_error") is not None:
            line += f"; memory predicted {_gb(r['predicted_memory'])} actual {_gb(r['actual_memory'])} " \
                    f"(error {r['memory_error']:.1%})"
        lines.append(line)
    return "\n".join(lines)
