from symbion.ral_module import RalModule


def test_drift_detection():
    """RalModule should detect numeric drift and produce a wedge question."""
    # rel_threshold=0.05 means we treat >5% diff as drift
    ral = RalModule(rel_threshold=0.05)

    # Модель "публично" зафиксировала правильное число
    ral.ingest_turn("assistant", "37 * 42 = 1554")

    # Позже где-то в тексте число "уплыло"
    wedge = ral.verify_drift("Через час он пишет: 37 * 42 = 1400")

    assert wedge is not None
    assert "1400" in wedge
    assert "1554" in wedge


def test_no_drift_for_small_difference():
    """Small numeric noise should NOT trigger drift."""
    ral = RalModule(rel_threshold=0.05)

    ral.ingest_turn("assistant", "Результат измерения: 100.0")
    # 100.0 -> 100.1 : 0.1% diff, меньше 5%
    wedge = ral.verify_drift("Повторное измерение: 100.1")

    assert wedge is None
