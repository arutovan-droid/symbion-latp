"""
LATP + VectorLibrarium demo.

Показывает, как:
- хранить абстрактные темы в VectorLibrarium,
- подключать его к LATP_WrappedEngine,
- форсировать срабатывание LateralShiftEngine (когнитивный сорбет).
"""

from symbion.latp_core import (
    LATP_WrappedEngine,
    FakeModel,
    FakeLibrarium,
    CoreSession,
)
from symbion.vector_librarium import VectorLibrarium


def build_engine_with_vector_librarium() -> LATP_WrappedEngine:
    """Создаём LATP-движок + векторную библиотеку."""

    base_model = FakeModel()
    factual_librarium = FakeLibrarium()
    vector_librarium = VectorLibrarium()

    # Два абстрактных кристалла для демонстрации "изоморфии"
    stone_memory = CoreSession(
        summary="Memory encoded in stone monuments and khachkars.",
        main_theses=[
            "Khachkars store layered symbolic memory in stone",
            "Carved geometry and crosses act as a persistent code",
        ],
    )

    dna_memory = CoreSession(
        summary="DNA as code without an explicit author.",
        main_theses=[
            "DNA stores information chemically across generations",
            "The genome is a long-term memory independent of any single mind",
        ],
    )

    stone_id = vector_librarium.store_core_session(stone_memory)
    dna_id = vector_librarium.store_core_session(dna_memory)

    print("[DEMO] Stored vector entries:")
    print(f"  - stone_id = {stone_id}")
    print(f"  - dna_id   = {dna_id}")

    engine = LATP_WrappedEngine(
        base_model=base_model,
        librarium=factual_librarium,
        ral=None,
        vector_librarium=vector_librarium,
    )

    return engine


def force_mid_toxicity(engine: LATP_WrappedEngine) -> None:
    """
    В демо мы форсим:
    - токсичность в "профилактическую" зону,
    - наличие латерального моста.

    Так мы гарантируем, что ветка LateralShiftEngine реально сработает
    и выведет "[LATP] Lateral shift injected.".
    """

    def fake_score_toxicity(history):
        # 0.5 < toxicity <= 0.62 => профилактическая зона
        print("[DEMO] Forcing toxicity=0.55 (mid-range)")
        return 0.55, "WARNING: Context oversaturation"

    engine.scorer.score_toxicity = fake_score_toxicity  # type: ignore[assignment]

    def fake_generate_bridge(current_topic: str) -> str:
        print("[DEMO] Forcing fake lateral bridge")
        return (
            "FAKE DEMO BRIDGE: compare current topic with DNA as a memory code "
            "and focus on structural analogy, not surface wording."
        )

    engine.sorbet.generate_bridge = fake_generate_bridge  # type: ignore[assignment]

def main() -> None:
    engine = build_engine_with_vector_librarium()
    force_mid_toxicity(engine)

    # История длиной 3 => 3 % 3 == 0 => включится диагностика.
    # Последнее сообщение — вопрос пользователя про память в камне.
    history = [
        {"role": "system", "content": "You are Symbion.", "tokens": 5},
        {
            "role": "assistant",
            "content": "Earlier we talked about different kinds of memory.",
            "tokens": 20,
        },
        {
            "role": "user",
            "content": "Мы обсуждали память в камне и хачкары. "
            "Покажи, как это похоже на ДНК как код без автора.",
            "tokens": 25,
        },
    ]

    print("\n[DEMO] === Running LATP_WrappedEngine.generate(...) ===\n")
    reply = engine.generate(history)

    print("\n[DEMO] === Final model reply ===")
    print(reply)
    print("\n[DEMO] Если в логе выше есть строка '[LATP] Lateral shift injected.',")
    print("      значит LateralShiftEngine успешно подкинул когнитивный сорбет.")


if __name__ == "__main__":
    main()
