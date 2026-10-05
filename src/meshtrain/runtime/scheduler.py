"""Pipeline schedules and the per-stage microbatch state machine.

A *schedule* is, for one stage and one global step, the ordered list of
actions that stage performs::

    Action(kind=FORWARD,  microbatch=m)
    Action(kind=BACKWARD, microbatch=m)

Schedules are generated for any number of stages S and microbatches M:

* ``gpipe`` -- fill/drain: all M forwards, then all M backwards. The last
  stage alternates F/B (it has every input it needs immediately).
* ``1f1b``  -- PipeDream-flush style: ``warmup_s`` forwards, then alternate
  one forward / one backward, then the remaining backwards (cooldown)::

      warmup_s = min(S - s - 1, max_inflight - 1, M)

  so stage s never holds more than ``min(S - s, max_inflight, M)``
  microbatch graphs, instead of M under GPipe.

Deadlock freedom: stage s emits ``warmup_s + 1`` forwards before it first
waits for a gradient, and stage s+1 needs ``warmup_{s+1} + 1`` forwards
before it emits its first gradient. Since ``warmup_{s+1} <= warmup_s`` for
every S, M and max_inflight, the first gradient always arrives.

Every global step: ``zero_grad`` -> all actions -> ``optimizer.step()``. The
schedule only reorders work inside one parameter version; it never changes
the training semantics.

The ``StageStateMachine`` tracks each microbatch through::

    QUEUED -> FORWARD_READY -> FORWARD_RUNNING -> FORWARD_DONE -> SENT_FORWARD
           -> BACKWARD_READY -> BACKWARD_RUNNING -> BACKWARD_DONE -> COMPLETE

driven by events (ACTIVATION_RECEIVED, FORWARD_FINISHED, GRADIENT_RECEIVED,
BACKWARD_FINISHED, SEND_COMPLETED). The executor asks it which action is
next; today that is "the next action in the static schedule", but the
state is explicit so a dynamic policy can replace it later.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ActionKind(str, enum.Enum):
    FORWARD = "F"
    BACKWARD = "B"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    microbatch: int

    def __str__(self) -> str:
        return f"{self.kind.value}{self.microbatch}"


SCHEDULES = ("gpipe", "1f1b")


def warmup_forwards(schedule: str, stage: int, num_stages: int, num_microbatches: int,
                    max_inflight: int | None = None) -> int:
    if schedule == "gpipe":
        return num_microbatches if stage < num_stages - 1 else 0
    limit = num_microbatches if max_inflight is None else max(1, max_inflight)
    return max(0, min(num_stages - stage - 1, limit - 1, num_microbatches))


def stage_schedule(schedule: str, stage: int, num_stages: int, num_microbatches: int,
                   max_inflight: int | None = None) -> list[Action]:
    """Ordered actions for ``stage`` in one global step."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if not 0 <= stage < num_stages or num_microbatches < 1:
        raise ValueError("invalid stage/microbatch counts")
    M = num_microbatches
    F = lambda m: Action(ActionKind.FORWARD, m)  # noqa: E731
    B = lambda m: Action(ActionKind.BACKWARD, m)  # noqa: E731
    if schedule == "gpipe" and stage < num_stages - 1:
        return [F(m) for m in range(M)] + [B(m) for m in range(M)]
    w = warmup_forwards(schedule, stage, num_stages, M, max_inflight)
    actions = [F(m) for m in range(w)]
    for i in range(M - w):
        actions += [F(w + i), B(i)]
    actions += [B(m) for m in range(M - w, M)]
    return actions


def max_inflight_of(actions: list[Action]) -> int:
    """Largest number of microbatches forwarded but not yet backwarded."""
    live = peak = 0
    for a in actions:
        live += 1 if a.kind == ActionKind.FORWARD else -1
        peak = max(peak, live)
    return peak


def format_schedule(actions: list[Action]) -> str:
    return " ".join(str(a) for a in actions)


class MicrobatchState(str, enum.Enum):
    QUEUED = "QUEUED"
    FORWARD_READY = "FORWARD_READY"
    FORWARD_RUNNING = "FORWARD_RUNNING"
    FORWARD_DONE = "FORWARD_DONE"
    SENT_FORWARD = "SENT_FORWARD"
    BACKWARD_READY = "BACKWARD_READY"
    BACKWARD_RUNNING = "BACKWARD_RUNNING"
    BACKWARD_DONE = "BACKWARD_DONE"
    COMPLETE = "COMPLETE"


class Event(str, enum.Enum):
    ACTIVATION_RECEIVED = "ACTIVATION_RECEIVED"
    FORWARD_STARTED = "FORWARD_STARTED"
    FORWARD_FINISHED = "FORWARD_FINISHED"
    GRADIENT_RECEIVED = "GRADIENT_RECEIVED"
    BACKWARD_STARTED = "BACKWARD_STARTED"
    BACKWARD_FINISHED = "BACKWARD_FINISHED"
    SEND_COMPLETED = "SEND_COMPLETED"


class SchedulerError(RuntimeError):
    pass


@dataclass
class _MB:
    state: MicrobatchState = MicrobatchState.QUEUED
    have_input: bool = False      # activation (or local data / target) available
    have_grad: bool = False       # gradient from downstream (or loss target on last stage)
    forward_sent: bool = False
    history: list[str] = field(default_factory=list)


class StageStateMachine:
    """Per-stage, per-step microbatch state + schedule cursor."""

    def __init__(self, stage: int, num_stages: int, num_microbatches: int, step: int,
                 schedule: str = "gpipe", max_inflight: int | None = None):
        self.stage, self.num_stages, self.step = stage, num_stages, step
        self.is_first, self.is_last = stage == 0, stage == num_stages - 1
        self.actions = stage_schedule(schedule, stage, num_stages, num_microbatches, max_inflight)
        self.cursor = 0
        self.mbs = {m: _MB() for m in range(num_microbatches)}
        self.inflight = 0
        self.peak_inflight = 0
        if self.is_first:  # stage 0 reads its inputs locally
            for m in self.mbs.values():
                m.have_input = True
                m.state = MicrobatchState.FORWARD_READY

    # -- events ------------------------------------------------------------
    def on(self, event: Event, mb: int) -> None:
        st = self.mbs.get(mb)
        if st is None:
            raise SchedulerError(f"event {event.value} for unknown microbatch {mb}")
        st.history.append(event.value)
        S = MicrobatchState
        if event == Event.ACTIVATION_RECEIVED:
            if st.have_input:
                raise SchedulerError(f"duplicate activation for microbatch {mb}")
            st.have_input = True
            st.state = S.FORWARD_READY
        elif event == Event.FORWARD_STARTED:
            self._expect(st, mb, S.FORWARD_READY)
            st.state = S.FORWARD_RUNNING
        elif event == Event.FORWARD_FINISHED:
            self._expect(st, mb, S.FORWARD_RUNNING)
            st.state = S.FORWARD_DONE
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            if self.is_last:  # loss can be computed locally
                st.have_grad = True
                st.state = S.BACKWARD_READY
        elif event == Event.SEND_COMPLETED:
            st.forward_sent = True
            if st.state == S.FORWARD_DONE:
                st.state = S.SENT_FORWARD
            if st.have_grad and st.state in (S.FORWARD_DONE, S.SENT_FORWARD):
                st.state = S.BACKWARD_READY
        elif event == Event.GRADIENT_RECEIVED:
            if st.have_grad:
                raise SchedulerError(f"duplicate gradient for microbatch {mb}")
            if st.state not in (S.FORWARD_DONE, S.SENT_FORWARD):
                raise SchedulerError(f"gradient for microbatch {mb} in state {st.state.value}")
            st.have_grad = True
            st.state = S.BACKWARD_READY
        elif event == Event.BACKWARD_STARTED:
            self._expect(st, mb, S.BACKWARD_READY)
            st.state = S.BACKWARD_RUNNING
        elif event == Event.BACKWARD_FINISHED:
            self._expect(st, mb, S.BACKWARD_RUNNING)
            st.state = S.COMPLETE
            self.inflight -= 1

    @staticmethod
    def _expect(st: _MB, mb: int, state: MicrobatchState) -> None:
        if st.state != state:
            raise SchedulerError(f"microbatch {mb} is {st.state.value}, expected {state.value}")

    # -- scheduling -----------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.cursor >= len(self.actions)

    def next_action(self) -> Action | None:
        return None if self.done else self.actions[self.cursor]

    def ready(self, action: Action) -> bool:
        st = self.mbs[action.microbatch]
        if action.kind == ActionKind.FORWARD:
            return st.state == MicrobatchState.FORWARD_READY
        return st.state == MicrobatchState.BACKWARD_READY

    def advance(self) -> None:
        self.cursor += 1

    def all_complete(self) -> bool:
        return all(m.state == MicrobatchState.COMPLETE for m in self.mbs.values())

    def snapshot(self) -> dict:
        """Diagnostic dump (deadlock reports)."""
        return {
            "stage": self.stage, "step": self.step,
            "next_action": str(self.next_action()) if not self.done else None,
            "remaining": format_schedule(self.actions[self.cursor:]),
            "inflight": self.inflight,
            "microbatches": {m: s.state.value for m, s in self.mbs.items()},
        }
