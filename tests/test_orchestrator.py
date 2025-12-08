import pytest
from symbion.orchestrator import build_fake_orchestrator


def test_orchestrator_basic():
    """Orchestrator should return some answer without crashing."""
    orchestrator = build_fake_orchestrator()
    history = []
    answer = orchestrator.get_answer("Hello, Relay!", history)
    assert isinstance(answer, str)
    assert answer != ""
