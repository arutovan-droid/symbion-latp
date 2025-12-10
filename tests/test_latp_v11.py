import pytest

from symbion.latp_core import (
    ContextPoisoningScorer,
    LATP_WrappedEngine,
    FakeModel,
    FakeLibrarium,
    AirlockModule,
)


def test_scorer_normal():
    """Short, healthy history should be diagnosed as NORMAL."""
    history = [
        {"role": "system", "content": "You are an assistant.", "tokens": 10},
        {"role": "user", "content": "Сколько будет 2+2?", "tokens": 8},
        {"role": "assistant", "content": "Четыре.", "tokens": 3},
    ]

    scorer = ContextPoisoningScorer(model_window=100)
    toxicity, diagnosis = scorer.score_toxicity(history)

    assert diagnosis == "NORMAL"
    assert toxicity <= scorer.CRITICAL_THRESHOLD


def test_latp_generate_basic():
    """LATP_WrappedEngine should at least delegate to FakeModel and not crash."""
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )

    history = [
        {"role": "system", "content": "You are Symbion.", "tokens": 5},
        {"role": "user", "content": "Hello, Symbion?", "tokens": 5},
    ]

    out = engine.generate(history)

    assert "FAKE: echo ->" in out
    assert "Hello, Symbion?" in out


def test_airlock_creates_crystal_and_clean_context():
    """Airlock should produce a clean context and persist a crystal in Librarium."""
    librarium = FakeLibrarium()
    airlock = AirlockModule(librarium_client=librarium, ral=None)

    history = [
        {"role": "system", "content": "You are Symbion.", "tokens": 5},
        {"role": "user", "content": "Turn 1", "tokens": 5},
        {"role": "assistant", "content": "Answer 1", "tokens": 5},
        {"role": "user", "content": "Final question?", "tokens": 5},
    ]

    clean_context, crystal = airlock.sanitize_session(history)

    # В чистом контексте должен быть системный тег с кристаллом
    assert any(
        "[LATP Crystal]" in msg.get("content", "") for msg in clean_context
    )
    # Кристалл должен ссылаться хотя бы на один id в Librarium
    assert crystal.librarium_refs
    # Минимум: system + crystal + последний вопрос
    assert len(clean_context) >= 2


def test_lateral_shift_injected(capsys):
    """
    LateralShiftEngine должен сработать, когда токсичность в "профилактической" зоне.

    Мы форсим:
    - scorer.score_toxicity -> всегда (0.55, WARNING)
    - sorbet.generate_bridge -> всегда возвращает мост

    Это гарантирует, что ветка латерального сдвига выполнится и в stdout
    появится "[LATP] Lateral shift injected.".
    """
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )

    # 1) Форсим токсичность в нужный диапазон (0.5 < t <= 0.62)
    def fake_score_toxicity(history):
        return 0.55, "WARNING: Context oversaturation"

    engine.scorer.score_toxicity = fake_score_toxicity

    # 2) Гарантируем, что латеральный сдвиг вернёт какой-то мост
    def fake_generate_bridge(current_topic: str) -> str:
        return "FAKE BRIDGE TOPIC"

    engine.sorbet.generate_bridge = fake_generate_bridge

    # len(history) == 3 => 3 % 3 == 0 => Diagnostics-фаза включится
    history = [
        {"role": "system", "content": "You are Symbion.", "tokens": 5},
        {"role": "user", "content": "Turn 1", "tokens": 10},
        {"role": "assistant", "content": "Answer 1", "tokens": 10},
    ]

    engine.generate(history)

    captured = capsys.readouterr().out
    assert "[LATP] Lateral shift injected." in captured

