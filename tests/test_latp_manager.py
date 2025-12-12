import pytest

from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium
from symbion.latp_manager import (
    LATPManager,
    LATPState,
    LATPAction,
    EpisodeMessage,
)


def build_manager_with_dummy_engine(toxicity_value: float, diagnosis: str = "NORMAL"):
    """Helper: создаёт LATPManager с движком, у которого scorer возвращает фиксированную токсичность."""
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )

    def fake_score_toxicity(history):
        # Игнорируем history, возвращаем заранее заданные значения.
        return toxicity_value, diagnosis

    # Подменяем scorer внутри engine
    engine.scorer.score_toxicity = fake_score_toxicity  # type: ignore[assignment]

    manager = LATPManager(engine=engine)
    return manager


def _feed_n_messages(manager: LATPManager, session_id: str, n: int, tokens: int = 10) -> None:
    """Быстро накидываем N сообщений в историю менеджера."""
    for i in range(n):
        msg = EpisodeMessage(role="user", content=f"msg-{i}", tokens=tokens)
        manager.on_message(session_id, msg)


def test_latp_manager_normal_state():
    """
    При низкой токсичности и короткой истории
    состояние должно быть NORMAL, действие CONTINUE.
    """
    manager = build_manager_with_dummy_engine(toxicity_value=0.1, diagnosis="NORMAL")
    session_id = "s-normal"

    # Мало сообщений, далеко до порогов HEAT_UP
    _feed_n_messages(manager, session_id, n=3, tokens=20)

    decision = manager.suggest_action(session_id)

    assert decision.state == LATPState.NORMAL
    assert decision.action == LATPAction.CONTINUE
    assert "toxicity=0.100" in decision.reason


def test_latp_manager_heat_up_shift():
    """
    При средней токсичности и длинной истории
    менеджер должен увидеть HEAT_UP и предложить SHIFT.
    """
    manager = build_manager_with_dummy_engine(toxicity_value=0.6, diagnosis="WARNING")
    session_id = "s-heat"

    # Длина истории >= min_turns_for_heat_up (по умолчанию 6)
    _feed_n_messages(manager, session_id, n=8, tokens=50)

    decision = manager.suggest_action(session_id)

    assert decision.state == LATPState.HEAT_UP
    assert decision.action == LATPAction.SHIFT
    assert "WARNING" in decision.reason


def test_latp_manager_airlock():
    """
    При высокой токсичности выше критического порога
    менеджер должен перевести сессию в AIRLOCK и предложить AIRLOCK.
    """
    manager = build_manager_with_dummy_engine(toxicity_value=0.9, diagnosis="CRITICAL")
    session_id = "s-airlock"

    _feed_n_messages(manager, session_id, n=2, tokens=5)

    decision = manager.suggest_action(session_id)

    assert decision.state == LATPState.AIRLOCK
    assert decision.action == LATPAction.AIRLOCK
    assert "toxicity=0.900" in decision.reason
