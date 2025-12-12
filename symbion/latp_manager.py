from __future__ import annotations

"""
High-level LATP manager + state machine.

This module does NOT change the low-level LATP_WrappedEngine behavior.
Instead, it sits above the engine and:

- tracks per-session state (NORMAL / HEAT_UP / LATERAL_SHIFT / COOL_DOWN / AIRLOCK / STABLE)
- consumes simple metrics (toxicity, turns, tokens)
- suggests actions to the orchestrator

Usage sketch (pseudo-code):

    engine = LATP_WrappedEngine(...)
    manager = LATPManager(engine=engine)

    manager.on_message("session-1", EpisodeMessage(role="user", content="...", tokens=42))
    decision = manager.suggest_action("session-1")

    if decision.action == LATPAction.CONTINUE:
        reply = engine.generate(history)
    elif decision.action == LATPAction.LATERAL_SHIFT:
        # orchestrator may force a lateral question / different engine
        ...

This file is intentionally light: it focuses on decision *interface*, not on
heavy math. Thresholds are configurable and can be refined later.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core enums & data models
# ---------------------------------------------------------------------------


class LATPState(Enum):
    """High-level session state."""

    NORMAL = auto()
    HEAT_UP = auto()
    LATERAL_SHIFT = auto()
    COOL_DOWN = auto()
    AIRLOCK = auto()
    STABLE = auto()


class LATPAction(Enum):
    """What the orchestrator is expected to do at this point."""

    CONTINUE = auto()  # normal generation
    SHIFT = auto()  # trigger a lateral shift / isomorphic topic
    COOL = auto()  # summarise, compress, fix crystal
    AIRLOCK = auto()  # hard context reset
    STAY_STABLE = auto()  # stay in STABLE state, no special action


@dataclass
class LATPMetricsSnapshot:
    """
    Minimal metrics set used for decisions.

    These can be extended later (entropy, K/T scores etc.).
    """

    toxicity: float
    diagnosis: str
    turns: int
    total_tokens: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMessage:
    """Generic message representation consumed by LATPManager."""

    role: str
    content: str
    tokens: int


@dataclass
class LATPDecision:
    """Result of manager.suggest_action()."""

    state: LATPState
    action: LATPAction
    reason: str
    metrics: LATPMetricsSnapshot


# ---------------------------------------------------------------------------
# Watchdog: maps metrics -> state/action suggestion
# ---------------------------------------------------------------------------


@dataclass
class WatchdogConfig:
    """
    Thresholds for simple rule-based decisions.

    All numbers are deliberately conservative and can be tuned later.
    """

    # When toxicity is below this, we consider the session "cold".
    toxicity_normal: float = 0.2

    # Above this we are in HEAT_UP zone.
    toxicity_heat_up: float = 0.5

    # Above this we require Airlock.
    toxicity_critical: float = 0.8

    # How many turns before we even consider HEAT_UP.
    min_turns_for_heat_up: int = 6

    # How many tokens in a session to suspect "long chat".
    max_tokens_before_heat_up: int = 2000


class LATPWatchdog:
    """
    Simple rule-based watchdog.

    It does *not* know about LATP internals. It only sees:
    - toxicity
    - turns
    - total tokens
    and returns a high-level suggestion.
    """

    def __init__(self, config: Optional[WatchdogConfig] = None) -> None:
        self.config = config or WatchdogConfig()

    def suggest_state(self, metrics: LATPMetricsSnapshot) -> LATPState:
        """Map metrics to high-level state."""
        c = self.config
        tox = metrics.toxicity

        if tox >= c.toxicity_critical:
            return LATPState.AIRLOCK

        if (
            metrics.turns >= c.min_turns_for_heat_up
            or metrics.total_tokens >= c.max_tokens_before_heat_up
        ):
            if tox >= c.toxicity_heat_up:
                return LATPState.HEAT_UP

        if tox <= c.toxicity_normal:
            # either STABLE or NORMAL depending on length
            if metrics.turns >= c.min_turns_for_heat_up:
                return LATPState.STABLE
            return LATPState.NORMAL

        # mid-range toxicity but not critical
        return LATPState.HEAT_UP

    def suggest_action(self, state: LATPState) -> LATPAction:
        """
        Map state to an action that orchestrator can understand.

        This is intentionally simple:
        - STABLE / NORMAL => CONTINUE
        - HEAT_UP         => SHIFT (lateral shift)
        - COOL_DOWN       => COOL
        - AIRLOCK         => AIRLOCK
        """
        if state in (LATPState.NORMAL, LATPState.STABLE):
            return LATPAction.CONTINUE
        if state is LATPState.HEAT_UP:
            return LATPAction.SHIFT
        if state is LATPState.COOL_DOWN:
            return LATPAction.COOL
        if state is LATPState.AIRLOCK:
            return LATPAction.AIRLOCK
        # Fallback
        return LATPAction.CONTINUE


# ---------------------------------------------------------------------------
# LATPManager: glue between sessions, metrics and watchdog
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """In-memory info about one session."""

    history: List[EpisodeMessage] = field(default_factory=list)
    state: LATPState = LATPState.NORMAL
    total_tokens: int = 0


class LATPManager:
    """
    High-level coordinator for LATP.

    Responsibilities:
    - keep lightweight per-session state (history length, tokens, last state)
    - call ContextPoisoningScorer via LATP_WrappedEngine
    - feed metrics into LATPWatchdog
    - return structured decisions to the orchestrator

    This class does *not* mutate the underlying engine's history.
    Orchestrator remains the source of truth for the actual LLM context.
    """

    def __init__(
        self,
        engine: Any,
        watchdog: Optional[LATPWatchdog] = None,
    ) -> None:
        """
        :param engine: LATP_WrappedEngine or compatible object with:
                       - scorer: ContextPoisoningScorer
                       - generate(history)  (not used here, but typically present)
        :param watchdog: optional custom LATPWatchdog instance
        """
        self.engine = engine
        self.watchdog = watchdog or LATPWatchdog()
        self._sessions: Dict[str, SessionState] = {}

    # -------------------- Session tracking --------------------

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def on_message(self, session_id: str, message: EpisodeMessage) -> None:
        """
        Feed a new message into the manager.

        Orchestrator should call this for every user/assistant turn it wants
        LATP to observe (typically after it's been appended to the main history).
        """
        session = self._get_session(session_id)
        session.history.append(message)
        session.total_tokens += message.tokens

    # -------------------- Decision logic --------------------

    def suggest_action(self, session_id: str) -> LATPDecision:
        """
        Inspect current session state and metrics and propose
        a high-level action for the orchestrator.

        This does NOT call engine.generate(). It only reads metrics
        from engine.scorer.
        """
        session = self._get_session(session_id)
        turns = len(session.history)

        # If we have no history, we are trivially NORMAL.
        if turns == 0:
            metrics = LATPMetricsSnapshot(
                toxicity=0.0,
                diagnosis="EMPTY",
                turns=0,
                total_tokens=0,
            )
            return LATPDecision(
                state=LATPState.NORMAL,
                action=LATPAction.CONTINUE,
                reason="No history yet.",
                metrics=metrics,
            )

        # Use the engine's scorer on a lightweight projection
        # (we convert EpisodeMessage -> dict expected by scorer).
        history_dicts: List[Dict[str, Any]] = [
            {"role": m.role, "content": m.content, "tokens": m.tokens}
            for m in session.history
        ]
        toxicity, diagnosis = self.engine.scorer.score_toxicity(history_dicts)

        metrics = LATPMetricsSnapshot(
            toxicity=toxicity,
            diagnosis=diagnosis,
            turns=turns,
            total_tokens=session.total_tokens,
        )

        state = self.watchdog.suggest_state(metrics)
        action = self.watchdog.suggest_action(state)

        session.state = state

        reason = f"toxicity={toxicity:.3f}, diagnosis={diagnosis}, turns={turns}, tokens={session.total_tokens}"

        return LATPDecision(
            state=state,
            action=action,
            reason=reason,
            metrics=metrics,
        )

    # -------------------- Optional helpers --------------------

    def reset_session(self, session_id: str) -> None:
        """Drop local LATP view of a session (does NOT touch engine history)."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session_state(self, session_id: str) -> LATPState:
        """Return last known state (defaults to NORMAL)."""
        return self._get_session(session_id).state
