from __future__ import annotations

from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium
from symbion.metrics import LATPMetricsLogger


def build_engine() -> LATP_WrappedEngine:
    """
    Build a minimal LATP_WrappedEngine for a demo.

    В этом примере мы не трогаем VectorLibrarium напрямую –
    латеральный сдвиг форсим через monkeypatch sorbet.generate_bridge.
    """
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )
    return engine


def force_mid_toxicity(engine: LATP_WrappedEngine) -> None:
    """
    Подменяем:
    - scorer.score_toxicity -> всегда (0.55, WARNING)
    - sorbet.generate_bridge -> всегда возвращает один и тот же мостик.

    Так мы гарантируем, что:
    - сработает ветка "профилактического" токсичности,
    - будет инжектирован латеральный сдвиг.
    """

    def fake_score_toxicity(history):
        print("[DEMO] Forcing toxicity=0.55 (mid-range)")
        return 0.55, "WARNING: Context oversaturation"

    def fake_generate_bridge(current_topic: str) -> str:
        print("[DEMO] Forcing fake lateral bridge")
        return (
            "Мы обсуждали память в камне и хачкары. "
            "Покажи, как это похоже на ДНК как код без автора."
        )

    # Жёсткий monkeypatch поверх реальных методов
    engine.scorer.score_toxicity = fake_score_toxicity  # type: ignore[assignment]
    engine.sorbet.generate_bridge = fake_generate_bridge  # type: ignore[assignment]


def main() -> None:
    """Запускает маленькое демо LATP + когнитивный сорбет + логгер метрик."""

    engine = build_engine()

    # Локальный логгер метрик (пишет в latp_metrics.jsonl)
    metrics = LATPMetricsLogger(
        file_path="latp_metrics.jsonl",
        session_id="demo_vector",
    )

    # Форсим зону средней токсичности, чтобы сработал латеральный сдвиг
    force_mid_toxicity(engine)

    # История длиной 3 => 3 % 3 == 0 => включится диагностика.
    history = [
        {"role": "system", "content": "You are Symbion.", "tokens": 5},
        {
            "role": "assistant",
            "content": "Earlier we talked about different kinds of memory.",
            "tokens": 20,
        },
        {
            "role": "user",
            "content": (
                "Мы обсуждали память в камне и хачкары. "
                "Покажи, как это похоже на ДНК как код без автора."
            ),
            "tokens": 25,
        },
    ]

    print("\n[DEMO] === Running LATP_WrappedEngine.generate(...) ===\n")

    # Логируем диагностическое событие (мы знаем, что подменили на 0.55)
    metrics.log_diagnostic(
        toxicity=0.55,
        diagnosis="WARNING: Context oversaturation",
    )

    # Основной вызов LATP-движка
    reply = engine.generate(history)

    # Логируем факт использования латерального сдвига
    metrics.log_lateral_shift()

    print("\n[DEMO] === Final model reply ===")
    print(reply)
    print(
        "\n[DEMO] Если в логе выше есть строка '[LATP] Lateral shift injected.',"
    )
    print("      значит LateralShiftEngine успешно подкинул когнитивный сорбет.")


if __name__ == "__main__":
    main()
