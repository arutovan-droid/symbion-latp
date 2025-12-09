import pytest
from symbion.latp_core import (
    ContextPoisoningScorer,
    LATP_WrappedEngine,
    FakeModel,
    FakeLibrarium,
)


def test_scorer_normal():
    """Basic sanity check: small clean history should be NORMAL."""
    history = [
        {"role": "system", "content": "You are an assistant.", "tokens": 10},
        {"role": "user", "content": "What is 2+2?", "tokens": 8},
        {"role": "assistant", "content": "Four.", "tokens": 3},
    ]
    scorer = ContextPoisoningScorer(model_window=100)
    toxicity, diag = scorer.score_toxicity(history)
    assert diag == "NORMAL"
    assert toxicity <= scorer.CRITICAL_THRESHOLD


def test_latp_engine_basic():
    """LATP_WrappedEngine should at least return some answer without crashing."""
    engine = LATP_WrappedEngine(FakeModel(), FakeLibrarium())
    history = [{"role": "user", "content": "Hello, LATP!", "tokens": 5}]
    answer = engine.generate(history)
    assert isinstance(answer, str)
    assert "Hello, LATP!" in answer


def test_rem_cycle_bypasses_watchdog():
    """REM cycle should eventually bypass the watchdog once REM flag is implemented."""
    # TODO: when REM becomes a runtime flag on LATP_WrappedEngine,
    #       this test should assert that responses are returned even
    #       if the watchdog would normally block them.
    pass
