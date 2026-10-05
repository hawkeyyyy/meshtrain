import itertools

import pytest

from meshtrain.runtime.scheduler import (
    ActionKind,
    Event,
    MicrobatchState,
    SchedulerError,
    StageStateMachine,
    format_schedule,
    max_inflight_of,
    stage_schedule,
)


def test_gpipe_fill_drain():
    assert format_schedule(stage_schedule("gpipe", 0, 3, 4)) == "F0 F1 F2 F3 B0 B1 B2 B3"
    # the last stage has its inputs immediately: forward/loss/backward per microbatch
    assert format_schedule(stage_schedule("gpipe", 2, 3, 4)) == "F0 B0 F1 B1 F2 B2 F3 B3"


def test_1f1b_matches_canonical_three_stage_example():
    assert format_schedule(stage_schedule("1f1b", 0, 3, 6)) == "F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 B4 B5"
    assert format_schedule(stage_schedule("1f1b", 1, 3, 6)) == "F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 B5"
    assert format_schedule(stage_schedule("1f1b", 2, 3, 6)) == "F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5"


@pytest.mark.parametrize("S,M", [(1, 1), (1, 5), (2, 1), (2, 8), (3, 2), (4, 16), (5, 3), (8, 32)])
@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_every_microbatch_once_and_forward_before_backward(schedule, S, M):
    for s in range(S):
        acts = stage_schedule(schedule, s, S, M)
        assert sorted(a.microbatch for a in acts if a.kind == ActionKind.FORWARD) == list(range(M))
        assert sorted(a.microbatch for a in acts if a.kind == ActionKind.BACKWARD) == list(range(M))
        seen = set()
        for a in acts:
            if a.kind == ActionKind.FORWARD:
                seen.add(a.microbatch)
            else:
                assert a.microbatch in seen


@pytest.mark.parametrize("S,M", [(2, 8), (3, 8), (4, 16)])
def test_1f1b_bounds_inflight_microbatches(S, M):
    for s in range(S):
        assert max_inflight_of(stage_schedule("gpipe", s, S, M)) == (M if s < S - 1 else 1)
        assert max_inflight_of(stage_schedule("1f1b", s, S, M)) == min(S - s, M)


def test_max_inflight_cap():
    for s in range(4):
        assert max_inflight_of(stage_schedule("1f1b", s, 4, 16, max_inflight=2)) <= 2


def _simulate(schedule, S, M, max_inflight=None):
    """Dependency simulation: F(s,m) needs F(s-1,m); B(s,m) needs B(s+1,m)."""
    plans = [stage_schedule(schedule, s, S, M, max_inflight) for s in range(S)]
    cursor = [0] * S
    done = set()
    progressed = True
    while progressed:
        progressed = False
        for s in range(S):
            if cursor[s] >= len(plans[s]):
                continue
            a = plans[s][cursor[s]]
            if a.kind == ActionKind.FORWARD:
                ok = s == 0 or ("F", s - 1, a.microbatch) in done
            else:
                ok = ("F", s, a.microbatch) in done and (s == S - 1 or ("B", s + 1, a.microbatch) in done)
            if ok:
                done.add((a.kind.value, s, a.microbatch))
                cursor[s] += 1
                progressed = True
    return all(cursor[s] == len(plans[s]) for s in range(S))


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_schedules_are_deadlock_free(schedule):
    for S, M in itertools.product(range(1, 7), range(1, 13)):
        for cap in (None, 1, 2, 3):
            assert _simulate(schedule, S, M, cap if schedule == "1f1b" else None), (schedule, S, M, cap)


def test_state_machine_happy_path_middle_stage():
    sm = StageStateMachine(1, 3, 2, step=0, schedule="1f1b")
    a = sm.next_action()
    assert a.kind == ActionKind.FORWARD and not sm.ready(a)
    sm.on(Event.ACTIVATION_RECEIVED, 0)
    assert sm.ready(a)
    sm.on(Event.FORWARD_STARTED, 0)
    sm.on(Event.FORWARD_FINISHED, 0)
    sm.on(Event.SEND_COMPLETED, 0)
    assert sm.mbs[0].state == MicrobatchState.SENT_FORWARD
    sm.on(Event.GRADIENT_RECEIVED, 0)
    assert sm.mbs[0].state == MicrobatchState.BACKWARD_READY
    sm.on(Event.BACKWARD_STARTED, 0)
    sm.on(Event.BACKWARD_FINISHED, 0)
    assert sm.mbs[0].state == MicrobatchState.COMPLETE and sm.inflight == 0


def test_state_machine_rejects_protocol_violations():
    sm = StageStateMachine(1, 3, 2, step=0)
    with pytest.raises(SchedulerError, match="gradient"):
        sm.on(Event.GRADIENT_RECEIVED, 0)  # before forward
    sm.on(Event.ACTIVATION_RECEIVED, 0)
    with pytest.raises(SchedulerError, match="duplicate"):
        sm.on(Event.ACTIVATION_RECEIVED, 0)
    with pytest.raises(SchedulerError, match="unknown"):
        sm.on(Event.ACTIVATION_RECEIVED, 7)
    with pytest.raises(SchedulerError, match="expected"):
        sm.on(Event.BACKWARD_STARTED, 0)


def test_gradient_may_arrive_before_send_completion():
    sm = StageStateMachine(0, 2, 1, step=0)
    sm.on(Event.FORWARD_STARTED, 0)
    sm.on(Event.FORWARD_FINISHED, 0)
    sm.on(Event.GRADIENT_RECEIVED, 0)  # async send notification still in flight
    assert sm.ready(sm.actions[1])
    sm.on(Event.SEND_COMPLETED, 0)
    assert sm.mbs[0].state == MicrobatchState.BACKWARD_READY


def test_unknown_schedule():
    with pytest.raises(ValueError):
        stage_schedule("zero-bubble", 0, 2, 2)
