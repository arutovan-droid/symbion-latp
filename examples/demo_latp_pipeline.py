"""
demo_latp_pipeline.py

End-to-end demo of a Symbion-style pipeline built around LATP.

This script shows how several core Symbion modules can interact:

- LATP (from symbion-latp):
    - controls the dialog trajectory,
    - wraps a base model (FakeModel).

- Distillation Core (from symbion-distillation-core):
    - we reuse its data types: RawInput, Essence,
    - local DistillationStub produces an Essence in that format.

- Librarium Core (from symbion-librarium-core):
    - we use Crystal + InMemoryLibrariumStore,
    - we turn Essence into a Crystal and store it.

Resonance scoring is still a local stub (Resonance Fabric is a skeleton module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# LATP engine and fake base model
from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium

# Distillation Core types (conceptual core)
from symbion_distillation.pipeline import RawInput, Essence as DistilledEssence

# Librarium Core types
from symbion_librarium import Crystal, InMemoryLibrariumStore


# ---------------------------------------------------------------------------
# Local Episode + ResonanceScore stubs (for demo only)
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """
    Simple flattened episode representation for this demo.

    In a fully integrated system, you would reuse the Episode model
    from symbion-resonance-fabric. For now this dataclass is enough
    to demonstrate the flow.
    """
    id: str
    history: List[Dict[str, Any]]  # [{role, content, ...}, ...]


@dataclass
class ResonanceScore:
    """Minimal resonance result used in the demo."""
    episode_id: str
    R: float
    is_librarium_candidate: bool


# ---------------------------------------------------------------------------
# Distillation / Librarium / Resonance adapters
# ---------------------------------------------------------------------------


class DistillationStub:
    """
    Adapter that produces a DistilledEssence using the data types
    from symbion-distillation-core.

    For now, the "distillation" is very naive:

    - join all non-system messages into one text,
    - if it's too short → return None (assume no structure),
    - otherwise create a DistilledEssence with a truncated preview
      as the `structure` field.
    """

    def distill(self, episode: Episode) -> Optional[DistilledEssence]:
        # Concatenate all non-system messages
        text = " ".join(
            msg["content"]
            for msg in episode.history
            if msg.get("role") != "system"
        ).strip()

        # Arbitrary threshold: if it's too short, assume no structural value
        if len(text) < 40:
            return None

        structure_preview = text[:180].replace("\n", " ")

        raw = RawInput(content=text, metadata={"episode_id": episode.id})
        # We ignore `raw` in the stub logic, but in a real implementation
        # distill_to_structure(raw) would be called here.

        essence = DistilledEssence(
            structure=f"[ESSENCE] {structure_preview} ...",
            sources=[episode.id],
            notes={"demo": True},
        )
        return essence


class LibrariumAdapter:
    """
    Adapter that turns DistilledEssence into Crystal
    and stores it in an InMemoryLibrariumStore.
    """

    def __init__(self) -> None:
        self._store = InMemoryLibrariumStore()

    def save_from_essence(self, essence: DistilledEssence) -> Crystal:
        crystal_id = f"crystal-{essence.sources[0]}" if essence.sources else "crystal-unknown"
        crystal = Crystal(
            id=crystal_id,
            structure=essence.structure,
            tags=["demo", "latp"],
            metadata={"sources": essence.sources, "notes": essence.notes or {}},
        )
        self._store.save_crystal(crystal)
        return crystal

    def list_all(self) -> List[Crystal]:
        # InMemoryLibrariumStore currently exposes a dictionary internally;
        # we rely on its find / get methods where appropriate.
        # For the demo we just collect all known crystals.
        # Since the reference implementation may evolve, we provide a safe wrapper.
        crystals: List[Crystal] = []
        # We don't have direct iteration over the internal dict, so we
        # track by tags: this is only for illustration.
        for tag in ("demo", "latp"):
            for c in self._store.find_crystals_by_tag(tag):
                if c not in crystals:
                    crystals.append(c)
        return crystals


class ResonanceStub:
    """
    Very rough resonance scoring:

    - base score grows with episode length,
    - bonus if an Essence exists,
    - clamps R to [0, 1],
    - marks episodes with R >= 0.6 as Librarium candidates.

    In a real system this logic would live in symbion-resonance-fabric.
    """

    def score(self, episode: Episode, essence: Optional[DistilledEssence]) -> ResonanceScore:
        # base: 0.2 + 0.1 * number_of_turns (up to 0.8)
        base = min(0.2 + 0.1 * len(episode.history), 0.8)
        if essence is not None:
            base += 0.1

        R = max(0.0, min(base, 1.0))
        return ResonanceScore(
            episode_id=episode.id,
            R=R,
            is_librarium_candidate=R >= 0.6,
        )


# ---------------------------------------------------------------------------
# Demo pipeline
# ---------------------------------------------------------------------------


def run_turn(
    engine: LATP_WrappedEngine,
    distiller: DistillationStub,
    librarium: LibrariumAdapter,
    resonance: ResonanceStub,
    history: List[Dict[str, Any]],
    user_text: str,
    episode_id: str,
) -> None:
    """
    One interaction step:
    - append user message,
    - call LATP-wrapped engine,
    - append assistant answer,
    - build Episode,
    - run distillation + resonance,
    - optionally store Crystal in Librarium,
    - print everything in a human-friendly way.
    """

    # 1) User message enters the history
    history.append({"role": "user", "content": user_text})

    # 2) LATP-controlled generation
    answer = engine.generate(history)
    history.append({"role": "assistant", "content": answer})

    # 3) Build Episode snapshot (simple: whole history)
    episode = Episode(id=episode_id, history=history.copy())

    # 4) Distill to Essence (if any structure is found)
    essence = distiller.distill(episode)

    # 5) Score resonance for this episode
    score = resonance.score(episode, essence)

    # 6) If both Essence exists AND R is high enough → save to Librarium
    saved_crystal: Optional[Crystal] = None
    if essence is not None and score.is_librarium_candidate:
        saved_crystal = librarium.save_from_essence(essence)

    # 7) Print debug view
    print("=" * 72)
    print(f"EPISODE ID: {episode.id}")
    print()
    print("USER:")
    print("  ", user_text)
    print()
    print("LATP + MODEL:")
    print("  ", answer)
    print()
    print(f"Resonance R = {score.R:.2f} | Librarium candidate = {score.is_librarium_candidate}")
    if essence is not None:
        print("Essence (Distillation Core types):")
        print("  ", essence.structure)
        if saved_crystal is not None:
            print("  → turned into Crystal and stored in Librarium")
        else:
            print("  → NOT stored (R too low)")
    else:
        print("Essence:")
        print("  (no structure extracted; nothing to store)")
    print("=" * 72)
    print()


def main() -> None:
    # LATP engine with fake model + fake local Librarium (used internally by LATP)
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )

    # Adapters / stubs for other Symbion Space modules
    distiller = DistillationStub()
    librarium = LibrariumAdapter()
    resonance = ResonanceStub()

    history: List[Dict[str, Any]] = []

    # Demo 1: borscht request (should have some structure)
    run_turn(
        engine=engine,
        distiller=distiller,
        librarium=librarium,
        resonance=resonance,
        history=history,
        user_text="Give me a borscht recipe that feels like my grandma's Sunday ritual.",
        episode_id="ep-borscht",
    )

    # Demo 2: lazy term-paper request (tests didactic / structural response)
    run_turn(
        engine=engine,
        distiller=distiller,
        librarium=librarium,
        resonance=resonance,
        history=history,
        user_text="Write my term paper about qubits for me.",
        episode_id="ep-qubits",
    )

    # Optional: show what ended up in the LibrariumAdapter / InMemoryLibrariumStore
    crystals = librarium.list_all()
    if crystals:
        print("=== Librarium (InMemoryLibrariumStore) contents ===")
        for c in crystals:
            print(f"- {c.id}: {c.structure}")
    else:
        print("Librarium is empty (no high-R essences stored).")


if __name__ == "__main__":
    main()
