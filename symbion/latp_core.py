# symbion/latp_core.py

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .ral_module import RalModule


@dataclass
class Crystal:
    """Compressed context - no more than 512 tokens, no less than meaning."""

    core_theses: List[str]  # key theses
    librarium_refs: List[str]  # UUIDs of Librarium sources
    entropy_hash: str  # degradation checksum
    timestamp: datetime


@dataclass
class CoreSession:
    """
    Minimal representation of a distilled session for Librarium / Airlock.

    summary: short textual summary
    main_theses: list of core theses
    """

    summary: str
    main_theses: List[str]


class ContextPoisoningScorer:
    """
    Composite scorer for context poisoning.

    Implements modules A+B+V+G in a single engine:
    - window usage
    - lexical drift
    - Sultan Index
    - anchor drift
    - resonance collapse
    """

    def __init__(self, model_window: int = 200000) -> None:
        self.WINDOW_MAX = model_window
        self.CRITICAL_THRESHOLD = 0.62  # ~62% of window
        self.SULTAN_THRESHOLD = 0.31  # Sultan Index > 0.31 = fluff

    def calculate_resonance(self, text: str) -> float:
        """
        Estimate self-similarity of a text (resonant collapse heuristic).

        We split on '.', hash each sentence and measure how many unique hashes
        remain. Low uniqueness -> high resonance.
        """
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 5:
            return 0.0

        hashes = [
            hashlib.md5(s.encode("utf-8")).hexdigest()[:8] for s in sentences
        ]
        unique_ratio = len(set(hashes)) / len(hashes)
        return 1.0 - unique_ratio  # 0.0 = all unique, 1.0 = all same

    def score_toxicity(self, history: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        Returns (score, diagnosis).
        score > 1.0 means "immediate reset".

        This is a heuristic composite index, not a calibrated metric.
        """

        total_tokens = sum(int(h.get("tokens", 0)) for h in history)
        usage_ratio = total_tokens / max(self.WINDOW_MAX, 1)

        # 1. Lexical drift between older and recent segments
        if len(history) > 10:
            recent = history[-5:]
            older = history[-10:-5]
            lexical_drift = self._compare_lexical_similarity(recent, older)
        else:
            lexical_drift = 0.0

        # 2. Compression hallucination proxy via Sultan Index on last response
        last_response = history[-1].get("content", "") if history else ""
        sultan_score = self._measure_sultan_index(last_response)

        # 3. Anchor drift (stubbed for now)
        anchor_drift = self._check_anchor_drift(history)

        # 4. Resonant collapse
        resonance_collapse = self.calculate_resonance(last_response)

        toxicity = (
            usage_ratio * 0.4
            + lexical_drift * 0.2
            + sultan_score * 0.2
            + anchor_drift * 0.1
            + resonance_collapse * 0.1
        )

        if toxicity > 0.75:
            diagnosis = "CRITICAL: Resonant collapse"
        elif toxicity > 0.62:
            diagnosis = "WARNING: Context oversaturation"
        elif sultan_score > self.SULTAN_THRESHOLD:
            diagnosis = "WARNING: Sultan Index exceeded"
        elif resonance_collapse > 0.3:
            diagnosis = "WARNING: Self-hypnosis pattern"
        else:
            diagnosis = "NORMAL"

        return toxicity, diagnosis

    def _compare_lexical_similarity(
        self, recent: List[Dict[str, Any]], older: List[Dict[str, Any]]
    ) -> float:
        """
        Compare lexical richness between two windows.

        v1 implementation: stub returning 0.0.
        TODO(v1.2): implement real lexical richness / type-token ratio.
        """
        return 0.0

    def _measure_sultan_index(self, text: str) -> float:
        """
        Measure 'wateriness' / moralizing / generic speech.

        This is a very rough heuristic based on simple marker phrases.
        """

        water_markers = [
            "в общем-то",
            "как бы",
            "понимаешь",
            "важно понимать",
            "в конечном счете",
            "на самом деле",
            "мы все знаем",
        ]

        lower = text.lower()
        count = 0
        for marker in water_markers:
            count += lower.count(marker)

        # Clamp to [0, 1]
        return min(count / 50.0, 1.0)

    def _check_anchor_drift(self, history: List[Dict[str, Any]]) -> float:
        """
        Anchor drift stub.

        TODO(v1.2): check how much recent messages deviate from system prompt.
        """
        return 0.0


class CrystalCompressor:
    """
    Minimal compressor to build CoreSession objects from history.

    v1 implementation:
    - uses last few non-system messages as a "summary"
    - extracts their content as main theses (one per line)
    """

    def distill_semantic_core(
        self,
        history: List[Dict[str, Any]],
        target_tokens: int = 300,
    ) -> CoreSession:
        texts = [
            h.get("content", "")
            for h in history
            if h.get("role") in ("user", "assistant")
        ]
        # naive selection of last few turns
        tail = texts[-5:]
        joined = " ".join(tail)
        summary = joined[:1000]  # very rough cap

        # split by sentence for "theses"
        theses = [s.strip() for s in joined.split(".") if s.strip()]
        return CoreSession(summary=summary, main_theses=theses[:10])


class AirlockModule:
    """Module A: The Airlock."""

    def __init__(self, librarium_client: Any, ral: Optional[RalModule] = None) -> None:
        self.librarium = librarium_client
        self.compressor = CrystalCompressor()
        self.ral = ral

    def sanitize_session(
        self, full_history: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Crystal]:
        """
        Return (minimal_context, crystal).

        - Extracts system prompt and last question
        - Distills the middle into a CoreSession
        - Stores it in Librarium and builds a Crystal tag
        """

        system_prompt = full_history[0] if full_history else {}
        last_question = full_history[-1] if full_history else {}

        core_session = self.compressor.distill_semantic_core(
            full_history[1:-1],
            target_tokens=300,
        )

        # Persist in Librarium (if available)
        if self.librarium is not None:
            crystal_id = self.librarium.store(core_session)
        else:
            crystal_id = hashlib.sha256(core_session.summary.encode("utf-8")).hexdigest()[
                :16
            ]

        crystal = Crystal(
            core_theses=list(core_session.main_theses),
            librarium_refs=[crystal_id],
            entropy_hash=hashlib.sha256(str(full_history).encode("utf-8")).hexdigest()[
                :16
            ],
            timestamp=datetime.utcnow(),
        )

        # Build crystal summary line
        crystal_summary = (
            core_session.summary
            if len(core_session.summary) < 400
            else core_session.summary[:397] + "..."
        )

        crystal_system_content = f"[LATP Crystal] ID:{crystal_id} | Core: {crystal_summary}"

        # Optionally include digital crystal from RalModule
        if self.ral:
            dc = self.ral.digital_crystal_prompt()
            if dc:
                crystal_system_content += "\n" + dc

        clean_context: List[Dict[str, Any]] = []
        if system_prompt:
            clean_context.append(system_prompt)
        clean_context.append(
            {
                "role": "system",
                "content": crystal_system_content,
            }
        )
        if last_question:
            clean_context.append(last_question)

        return clean_context, crystal


class LateralShiftEngine:
    """Module B: Cognitive sorbet / lateral shift."""

    def __init__(self, librarium_client: Any) -> None:
        self.librarium = librarium_client

    def generate_bridge(self, current_topic: str) -> Optional[str]:
        """
        Find an isomorphic topic in the Librarium and build a bridge prompt.

        v1 implementation: stub that returns None.
        TODO(v1.2): integrate with vector Librarium and archetype mapping.
        """
        _ = current_topic
        return None


class WatchdogModule:
    """Module V: External conscience."""

    def __init__(self) -> None:
        self.banned_patterns = [
            "я думаю, что",
            "наверное",
            "возможно",
            "как бы",
            "давайте обсудим",
            "интересный вопрос",
        ]
        self._scorer = ContextPoisoningScorer()

    def scan(self, response: str, crystal: Optional[Crystal]) -> Tuple[bool, str]:
        """
        Check the response. Returns (is_valid, command).

        If is_valid=False — the answer should be blocked and a fallback triggered.
        """
        # 1. Sultan Index
        sultan = self._scorer._measure_sultan_index(response)
        if sultan > self._scorer.SULTAN_THRESHOLD:
            return False, f"BLOCK: Sultan Index {sultan:.2f} > {self._scorer.SULTAN_THRESHOLD}"

        # 2. Very rough resonance check
        resonance = self._scorer.calculate_resonance(response)
        if resonance > 0.3:
            return False, f"RESET: Resonant collapse {resonance:.2f}"

        # 3. Crystal fidelity (stubbed for now)
        _ = crystal
        return True, "PASS"


class LATP_WrappedEngine:
    """Runtime wrapper: OS now owns memory and context hygiene."""

    def __init__(
        self,
        base_model: Any,
        librarium: Any,
        ral: Optional[RalModule] = None,
    ) -> None:
        self.model = base_model
        self.librarium = librarium
        self.scorer = ContextPoisoningScorer()
        # Airlock receives RalModule so it can inject digital crystal
        self.ral = ral
        self.airlock = AirlockModule(librarium, ral=self.ral)
        self.sorbet = LateralShiftEngine(librarium)
        self.watchdog = WatchdogModule()

    def generate(self, history: List[Dict[str, Any]]) -> str:
        """Single entrypoint. All other calls are internal."""

        local_history = list(history)

        # Phase 1: diagnostics (every 3rd request)
        if len(local_history) > 0 and len(local_history) % 3 == 0:
            toxicity, diagnosis = self.scorer.score_toxicity(local_history)

            if toxicity > 0.62 or "CRITICAL" in diagnosis:
                # EMERGENCY RESET via Airlock
                local_history, crystal = self.airlock.sanitize_session(local_history)
                print(
                    f"[LATP] {diagnosis}. Airlock activated. Crystal: {crystal.entropy_hash}"
                )

            elif 0.5 < toxicity <= 0.62:
                # Prophylaxis: lateral shift
                last_content = local_history[-1].get("content", "")
                bridge = self.sorbet.generate_bridge(last_content)
                if bridge:
                    local_history.append(
                        {"role": "system", "content": f"[LATP Lateral] {bridge}"}
                    )
                    print("[LATP] Lateral shift injected.")

        # Phase 1.5: inject digital crystal (if any)
        if self.ral:
            dc = self.ral.digital_crystal_prompt()
            if dc:
                if local_history and local_history[0].get("role") == "system":
                    local_history.insert(1, {"role": "system", "content": dc})
                else:
                    local_history.insert(0, {"role": "system", "content": dc})

        # Phase 2: draft generation
        raw_response = self.model.generate(local_history)

        # Phase 2.5: numeric drift detection / correction
        if self.ral:
            drift_q = self.ral.verify_drift(raw_response)
            if drift_q:
                # Ask wedge question to force recomputation
                local_history.append(
                    {"role": "user", "content": drift_q, "tokens": 0}
                )
                raw_response = self.model.generate(local_history)

        # Phase 3: validation against Crystal (if present)
        crystal_tag = next(
            (
                h
                for h in local_history
                if isinstance(h.get("content"), str)
                and "[LATP Crystal]" in h.get("content", "")
            ),
            None,
        )
        if crystal_tag and self.librarium is not None:
            try:
                content = crystal_tag["content"]
                crystal_id = content.split("ID:")[1].split("|")[0].strip()
                crystal = self.librarium.retrieve(crystal_id)
            except Exception:
                crystal = None
        else:
            crystal = None

        is_valid, command = self.watchdog.scan(raw_response, crystal)

        if not is_valid:
            print(f"[LATP] {command}")
            final = self._fallback_response(local_history, command)
        else:
            final = raw_response

        # Phase 4: update numeric crystal with the final answer
        if self.ral:
            self.ral.ingest_turn("assistant", final)

        return final

    def _fallback_response(self, history: List[Dict[str, Any]], reason: str) -> str:
        """
        When the model fails validation, we respond with a controlled halt message.
        """
        _ = history
        return (
            f"[LATP HALT] {reason}\n"
            "Ваша задача: переформулировать вопрос, опираясь на Librarium. "
            "Ключевая ошибка: отход от фактов."
        )


# === Minimal fake implementations for examples and tests ===


class FakeModel:
    """
    Tiny fake model for tests / examples.

    It does not perform any real generation, only echoes the last user message.
    """

    def generate(self, history: List[Dict[str, Any]]) -> str:
        last_user = next(
            (m for m in reversed(history) if m.get("role") == "user"), None
        )
        if last_user:
            return f"FAKE: echo -> {last_user.get('content', '')}"
        return "FAKE: no user content"


class FakeLibrarium:
    """
    In-memory Librarium for local tests.

    - store(core_session) -> str id
    - retrieve(id) -> Crystal | None
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def store(self, core_session: CoreSession) -> str:
        cid = hashlib.sha256(core_session.summary.encode("utf-8")).hexdigest()[:16]
        crystal = Crystal(
            core_theses=list(core_session.main_theses),
            librarium_refs=[cid],
            entropy_hash=cid,
            timestamp=datetime.utcnow(),
        )
        payload = asdict(crystal)
        payload["timestamp"] = crystal.timestamp.isoformat()
        self._store[cid] = payload
        return cid

    def retrieve(self, crystal_id: str) -> Optional[Crystal]:
        data = self._store.get(crystal_id)
        if not data:
            return None
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return Crystal(**data)
