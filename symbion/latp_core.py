from __future__ import annotations

import hashlib
import math
import asyncio
import json
from symbion_cognitive_collider.collider import route_language
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Core data structures
# =========================


@dataclass
class Crystal:
    """Compressed context snapshot ("crystal")."""

    core_theses: List[str]
    librarium_refs: List[str]
    entropy_hash: str
    timestamp: datetime

@dataclass
class CoreSession:
    """
    Minimal representation of a core dialog session for VectorLibrarium.

    VectorLibrarium and related tests expect at least:
    - topic: optional high-level topic name
    - summary: short text summary
    - main_theses: list of key theses
    - tokens: approximate token count (optional)
    """

    topic: str = ""
    summary: str = ""
    main_theses: List[str] = field(default_factory=list)
    tokens: int = 0

class ContextPoisoningScorer:
    """Heuristic scorer for context poisoning / fatigue.

    Responsibilities:
    - estimate how 'heavy' the context is versus model window
    - compute a rough toxicity score in [0, 1]
    - produce a human-readable diagnosis string
    - optionally measure resonance (self-repetition) of a single answer
    """

    def __init__(self, model_window: int = 200_000) -> None:
        # approximate maximum tokens window for the underlying model
        self.WINDOW_MAX = model_window
        # above this we consider the session critically poisoned
        self.CRITICAL_THRESHOLD = 0.62
        # 'sultan zone' Ã¢â‚¬â€ suspiciously high certainty / verbosity
        self.SULTAN_THRESHOLD = 0.31

    # ---------------- core helpers ----------------

    def _total_tokens(self, history: list[dict]) -> int:
        return sum(int(m.get("tokens", 0)) for m in history)

    def calculate_entropy(self, history: list[dict]) -> float:
        """Crude 'entropy' proxy: how full is the window.

        0.0 Ã¢â€ â€™ ÃÂ¿Ã‘Æ’Ã‘ÂÃ‘â€šÃÂ¾, 1.0 Ã¢â€ â€™ ÃÂ¾ÃÂºÃÂ½ÃÂ¾ ÃÂ·ÃÂ°ÃÂ±ÃÂ¸Ã‘â€šÃÂ¾ ÃÂ¿ÃÂ¾ÃÂ´ ÃÂ·ÃÂ°ÃÂ²Ã‘ÂÃÂ·ÃÂºÃ‘Æ’.
        """
        used = self._total_tokens(history)
        ratio = min(max(used / max(self.WINDOW_MAX, 1), 0.0), 1.0)
        return ratio

    def calculate_resonance(self, text: str) -> float:
        """Detects self-repetition inside a single answer.

        Simple heuristic:
        - split by '.'
        - hash each sentence
        - measure how many unique hashes there are
        - more repetition Ã¢â€ â€™ higher resonance in [0, 1]
        """
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 5:
            # ÃÂºÃÂ¾Ã‘â‚¬ÃÂ¾Ã‘â€šÃÂºÃÂ¸ÃÂµ ÃÂ¾Ã‘â€šÃÂ²ÃÂµÃ‘â€šÃ‘â€¹ Ã‘ÂÃ‘â€¡ÃÂ¸Ã‘â€šÃÂ°ÃÂµÃÂ¼ ÃÂ±ÃÂµÃÂ·ÃÂ¾ÃÂ¿ÃÂ°Ã‘ÂÃÂ½Ã‘â€¹ÃÂ¼ÃÂ¸
            return 0.0

        hashes = [
            hashlib.md5(s.encode("utf-8")).hexdigest()[:8] for s in sentences
        ]
        unique_ratio = len(set(hashes)) / len(hashes)
        return 1.0 - unique_ratio

    # ---------------- public API ----------------

    def score_toxicity(self, history: list[dict]) -> tuple[float, str]:
        """Main entrypoint used by LATP and orchestrator.

        Returns:
            (toxicity, diagnosis)
            toxicity Ã¢Ë†Ë† [0, 1]
        """
        entropy = self.calculate_entropy(history)

        # ÃÂ»Ã‘â€˜ÃÂ³ÃÂºÃÂ°Ã‘Â ÃÂ¿ÃÂ¾ÃÂ¿Ã‘â‚¬ÃÂ°ÃÂ²ÃÂºÃÂ° ÃÂ·ÃÂ° ÃÂ´ÃÂ»ÃÂ¸ÃÂ½Ã‘Æ’ ÃÂ´ÃÂ¸ÃÂ°ÃÂ»ÃÂ¾ÃÂ³ÃÂ°
        turns = len(history)
        length_factor = min(turns / 100.0, 1.0)

        toxicity = min(entropy * 0.7 + length_factor * 0.3, 1.0)

        if toxicity > self.CRITICAL_THRESHOLD:
            diagnosis = "CRITICAL: Context poisoning suspected"
        elif toxicity > self.SULTAN_THRESHOLD:
            diagnosis = "WARNING: Context oversaturation"
        else:
            diagnosis = "NORMAL"

        return toxicity, diagnosis

    def compare_lexical_similarity(self, a: str, b: str) -> float:
        """Very cheap lexical similarity metric in [0, 1].

        Used only as a weak signal; 0.0 means 'no visible overlap'.
        """
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union



class AirlockModule:
    """
    Airlock: truncate long history into a minimal "clean" context + Crystal.
    """

    def __init__(self, librarium_client: Any | None = None, ral: Any | None = None) -> None:
        self.librarium = librarium_client
        self.ral = ral

    def sanitize_session(self, full_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Crystal]:
        """
        Returns (clean_context, crystal).

        Clean context convention (tests / examples expect this shape):
        - system prompt
        - system: [LATP Crystal] ...
        - last user message
        """
        if not full_history:
            system_prompt: Dict[str, Any] = {"role": "system", "content": "You are Symbion."}
            last_message: Dict[str, Any] = {"role": "user", "content": ""}
        else:
            # assume first is system
            system_prompt = full_history[0]
            last_message = full_history[-1]

        # Very simple "crystal"
        core_theses = ["compressed session"]
        entropy_hash = hashlib.sha256(str(full_history).encode("utf-8")).hexdigest()[:16]

        crystal = Crystal(
            core_theses=core_theses,
            librarium_refs=[],
            entropy_hash=entropy_hash,
            timestamp=datetime.utcnow(),
        )

        # Store in librarium if provided
        if self.librarium is not None and hasattr(self.librarium, "store"):
            try:
                cid = self.librarium.store({"summary": "compressed session"})
                crystal.librarium_refs.append(cid)
                crystal_system_content = f"[LATP Crystal] ID:{cid} | Core: compressed session"
            except Exception:
                crystal_system_content = "[LATP Crystal] Core: compressed session"
        else:
            crystal_system_content = "[LATP Crystal] Core: compressed session"

        clean_context: List[Dict[str, Any]] = [
            system_prompt,
            {"role": "system", "content": crystal_system_content},
            last_message,
        ]

        # If RAL is present, ÃÂ´ÃÂ¾ÃÂ¿ÃÂ¾ÃÂ»ÃÂ½ÃÂ¸Ã‘â€šÃÂµÃÂ»Ã‘Å’ÃÂ½ÃÂ¾ ÃÂ¿ÃÂ¾ÃÂ´ÃÂ¼ÃÂµÃ‘Ë†ÃÂ¸ÃÂ²ÃÂ°ÃÂµÃÂ¼ Ã‘â€ ÃÂ¸Ã‘â€žÃ‘â‚¬ÃÂ¾ÃÂ²ÃÂ¾ÃÂ¹ ÃÂºÃ‘â‚¬ÃÂ¸Ã‘ÂÃ‘â€šÃÂ°ÃÂ»ÃÂ»:
        if self.ral is not None and hasattr(self.ral, "digital_crystal_prompt"):
            dc = self.ral.digital_crystal_prompt()
            if dc:
                clean_context.insert(1, {"role": "system", "content": dc})

        return clean_context, crystal


# =========================
# Lateral shift engine
# =========================


class LateralShiftEngine:
    """
    Generates "cognitive sorbet" Ã¢â‚¬â€œ lateral prompts based on isomorphic topics.
    """

    def __init__(self, librarium_client: Any | None = None, vector_librarium: Any | None = None) -> None:
        self.librarium = librarium_client
        self.vector_librarium = vector_librarium

    def generate_bridge(self, current_topic: str) -> Optional[str]:
        """
        Try to find an isomorphic topic and build a bridge prompt.
        Returns None if nothing found.
        """
        iso = None

        # First: try librarium.isomorphic search, if present
        if self.librarium is not None and hasattr(self.librarium, "find_isomorphic"):
            try:
                iso = self.librarium.find_isomorphic(vector=current_topic, domain_shift=True)
            except Exception:
                iso = None

        # If still nothing and vector_librarium present, try that
        if iso is None and self.vector_librarium is not None and hasattr(
            self.vector_librarium, "find_isomorphic"
        ):
            try:
                iso = self.vector_librarium.find_isomorphic(current_topic)
            except Exception:
                iso = None

        if iso is None:
            return None

        name = getattr(iso, "name", "ÃÂ´Ã‘â‚¬Ã‘Æ’ÃÂ³Ã‘Æ’Ã‘Å½ Ã‘ÂÃ‘â€šÃ‘â‚¬Ã‘Æ’ÃÂºÃ‘â€šÃ‘Æ’Ã‘â‚¬Ã‘Æ’")
        arch_type = getattr(iso, "arch_type", "ÃÂ°Ã‘â‚¬Ã‘â€¦ÃÂµÃ‘â€šÃÂ¸ÃÂ¿")
        field = getattr(iso, "field", "ÃÂ´Ã‘â‚¬Ã‘Æ’ÃÂ³ÃÂ¾ÃÂ¹ ÃÂ¾ÃÂ±ÃÂ»ÃÂ°Ã‘ÂÃ‘â€šÃÂ¸")
        method = getattr(iso, "method", "ÃÂ´Ã‘â‚¬Ã‘Æ’ÃÂ³ÃÂ¸ÃÂ¼ ÃÂ¼ÃÂµÃ‘â€šÃÂ¾ÃÂ´ÃÂ¾ÃÂ¼")

        bridge_candidates = [
            f"ÃÅ“Ã‘â€¹ Ã‘â‚¬ÃÂ°Ã‘ÂÃ‘ÂÃÂ¼ÃÂ¾Ã‘â€šÃ‘â‚¬ÃÂµÃÂ»ÃÂ¸ {current_topic}. ÃÂ Ã‘â€šÃÂµÃÂ¿ÃÂµÃ‘â‚¬Ã‘Å’ Ã‘ÂÃ‘â‚¬ÃÂ°ÃÂ²ÃÂ½ÃÂ¸: {name}. "
            f"ÃÂÃÂµ ÃÂ¿ÃÂ¾ÃÂ²Ã‘â€šÃÂ¾Ã‘â‚¬Ã‘ÂÃÂ¹ Ã‘ÂÃÂµÃÂ±Ã‘Â. ÃËœÃ‘â€°ÃÂ¸ ÃÂ³ÃÂ»Ã‘Æ’ÃÂ±ÃÂ¸ÃÂ½ÃÂ½Ã‘Æ’Ã‘Å½ ÃÂ°ÃÂ½ÃÂ°ÃÂ»ÃÂ¾ÃÂ³ÃÂ¸Ã‘Å½, ÃÂ½ÃÂµ ÃÂ¿ÃÂ¾ÃÂ²ÃÂµÃ‘â‚¬Ã‘â€¦ÃÂ½ÃÂ¾Ã‘ÂÃ‘â€šÃÂ½Ã‘Æ’Ã‘Å½.",
            f"ÃÂ¢ÃÂµÃÂ¼ÃÂ° {current_topic} ÃÂ¸ÃÂ¼ÃÂµÃÂµÃ‘â€š ÃÂ°Ã‘â‚¬Ã‘â€¦ÃÂµÃ‘â€šÃÂ¸ÃÂ¿ {arch_type}. "
            f"ÃÅ¡ÃÂ°ÃÂº Ã‘ÂÃ‘â€šÃÂ¾Ã‘â€š ÃÂ°Ã‘â‚¬Ã‘â€¦ÃÂµÃ‘â€šÃÂ¸ÃÂ¿ ÃÂ¿Ã‘â‚¬ÃÂ¾Ã‘ÂÃÂ²ÃÂ»Ã‘ÂÃÂµÃ‘â€šÃ‘ÂÃ‘Â ÃÂ² {field}? "
            f"Ãâ€ÃÂ°ÃÂ¹ ÃÂ¾Ã‘â€šÃÂ²ÃÂµÃ‘â€š Ã‘â€¡ÃÂµÃ‘â‚¬ÃÂµÃÂ· ÃÂ¿Ã‘â‚¬ÃÂ¸ÃÂ·ÃÂ¼Ã‘Æ’ {method}.",
        ]

        # stable index based on hash
        idx = abs(hash(current_topic)) % len(bridge_candidates)
        return bridge_candidates[idx]


# =========================
# Watchdog (minimal)
# =========================


class WatchdogModule:
    """
    Minimal watchdog: can be extended later.
    For now: always PASS, used just as extension point.
    """

    def __init__(self) -> None:
        self.banned_patterns: List[str] = []

    def scan(self, response: str, crystal: Optional[Crystal]) -> Tuple[bool, str]:
        """Return (is_valid, command)."""
        return True, "PASS"


# =========================
# Fake model / librarium for tests & demos
# =========================


class FakeModel:
    """Very small stand-in model used in tests and examples."""

    def generate(self, history: List[Dict[str, Any]]) -> str:
        # Echo last user message
        last_user = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user = msg
                break
        if last_user is None and history:
            last_user = history[-1]
        content = last_user.get("content", "") if last_user else ""
        return f"FAKE: echo -> {content}"


class FakeLibrarium:
    """In-memory librarium used in tests."""

    def __init__(self) -> None:
        self.storage: Dict[str, Any] = {}

    def store(self, core_session: Any) -> str:
        cid = hashlib.sha256(str(core_session).encode("utf-8")).hexdigest()[:16]
        self.storage[cid] = core_session
        return cid

    def retrieve(self, crystal_id: str) -> Any:
        return self.storage.get(crystal_id)

    def find_isomorphic(self, vector: str, domain_shift: bool = True) -> Any | None:
        # Very small stub object with attributes
        class Iso:
            def __init__(self) -> None:
                self.name = "Ãâ€ÃÂÃÅ¡"
                self.arch_type = "ÃÂ¸ÃÂµÃ‘â‚¬ÃÂ°Ã‘â‚¬Ã‘â€¦ÃÂ¸Ã‘â€¡ÃÂµÃ‘ÂÃÂºÃÂ°Ã‘Â ÃÂ¿ÃÂ°ÃÂ¼Ã‘ÂÃ‘â€šÃ‘Å’"
                self.field = "ÃÂ±ÃÂ¸ÃÂ¾ÃÂ»ÃÂ¾ÃÂ³ÃÂ¸Ã‘Â"
                self.method = "ÃÂºÃÂ¾ÃÂ´ÃÂ° ÃÂ±ÃÂµÃÂ· ÃÂ°ÃÂ²Ã‘â€šÃÂ¾Ã‘â‚¬ÃÂ°"

        return Iso()


# =========================
# LATP wrapped engine
# =========================


class LATP_WrappedEngine:
    """LATP wrapper around a base LLM model."""

    def __init__(
        self,
        base_model: Any,
        librarium: Any,
        ral: Any | None = None,
        vector_librarium: Any | None = None,
    ) -> None:
        """
        - base_model: must implement .generate(history: list[dict]) -> str
        - librarium: object with .store(...) / .retrieve(id) for crystals
        - ral: optional RalModule instance for numeric drift detection
        - vector_librarium: optional VectorLibrarium for lateral shifts
        """
        self.model = base_model
        self.librarium = librarium
        self.ral = ral

        self.scorer = ContextPoisoningScorer()
        self.airlock = AirlockModule(librarium_client=librarium, ral=ral)
        self.sorbet = LateralShiftEngine(librarium_client=librarium, vector_librarium=vector_librarium)
        self.watchdog = WatchdogModule()

        # Optional metrics monitor. External code may set:
        # engine.monitor = LATPMetricsLogger(...)
        self.monitor: Any | None = None

    def generate(self, history: List[Dict[str, Any]]) -> str:
        """
        Single entry point used in tests / demos.

        - periodically checks toxicity
        - may trigger airlock (hard reset) or lateral shift
        - delegates generation to base_model
        - optionally applies RAL numeric drift guard
        """
        local_history = list(history)

        # Phase 1: diagnostics each 3rd turn
        turns = len(local_history)
        toxicity = 0.0
        diagnosis = "NORMAL"

        if turns % 3 == 0:
            toxicity, diagnosis = self.scorer.score_toxicity(local_history)

            if toxicity > self.scorer.CRITICAL_THRESHOLD or "CRITICAL" in diagnosis:
                # Hard reset via Airlock
                local_history, crystal = self.airlock.sanitize_session(local_history)
                print(f"[LATP] {diagnosis}. Airlock activated. Crystal: {crystal.entropy_hash}")
            elif 0.5 < toxicity <= self.scorer.CRITICAL_THRESHOLD:
                # Prophylactic lateral shift
                last_content = local_history[-1].get("content", "") if local_history else ""
                bridge = self.sorbet.generate_bridge(last_content)
                if bridge:
                    local_history.append({"role": "system", "content": f"[LATP Lateral] {bridge}"})
                    print("[LATP] Lateral shift injected.")
        # Phase 1.5: basis_select (Cognitive Collider)
        try:
            last_user_text = ""
            for msg in reversed(local_history):
                if msg.get("role") == "user":
                    last_user_text = msg.get("content", "") or ""
                    break

            cog = asyncio.run(
                route_language(
                    last_user_text,
                    {"history": local_history},
                    life_vector=None,
                )
            )
            cog_dict = cog.model_dump()

            # attach to core session if present
            if hasattr(self, "core_session") and getattr(self, "core_session", None) is not None:
                if not hasattr(self.core_session, "context") or self.core_session.context is None:
                    self.core_session.context = {}
                self.core_session.context["cog_lang"] = cog_dict

            # also inject into history for downstream visibility
            local_history.append({
                "role": "system",
                "content": "[LATP CogLang] " + json.dumps(cog_dict, ensure_ascii=False),
                "cog_lang": cog_dict,
            })
        except Exception:
            pass


        # Phase 2: generation
        raw_response = self.model.generate(local_history)

        # Phase 3: optional RAL numeric drift guard
        if self.ral is not None and hasattr(self.ral, "verify_drift"):
            try:
                drift_q = self.ral.verify_drift(raw_response)
            except Exception:
                drift_q = None

            if drift_q:
                # Ask model to recompute with wedge question
                local_history.append({"role": "user", "content": drift_q})
                raw_response = self.model.generate(local_history)

        # Phase 4: watchdog (currently always PASS)
        crystal_tag = next(
            (h for h in local_history if "[LATP Crystal]" in h.get("content", "")),
            None,
        )
        crystal_obj: Optional[Crystal] = None
        if crystal_tag and hasattr(self.librarium, "retrieve"):
            try:
                tag_content: str = crystal_tag["content"]
                if "ID:" in tag_content:
                    crystal_id = tag_content.split("ID:")[1].split()[0]
                    retrieved = self.librarium.retrieve(crystal_id)
                    if isinstance(retrieved, Crystal):
                        crystal_obj = retrieved
            except Exception:
                crystal_obj = None

        is_valid, command = self.watchdog.scan(raw_response, crystal_obj)
        if not is_valid:
            return f"[LATP HALT] {command}\nÃÅ¸ÃÂµÃ‘â‚¬ÃÂµÃ‘â€žÃÂ¾Ã‘â‚¬ÃÂ¼Ã‘Æ’ÃÂ»ÃÂ¸Ã‘â‚¬Ã‘Æ’ÃÂ¹Ã‘â€šÃÂµ ÃÂ²ÃÂ¾ÃÂ¿Ã‘â‚¬ÃÂ¾Ã‘Â, ÃÂ¾ÃÂ¿ÃÂ¸Ã‘â‚¬ÃÂ°Ã‘ÂÃ‘ÂÃ‘Å’ ÃÂ½ÃÂ° Ã‘â€žÃÂ°ÃÂºÃ‘â€šÃ‘â€¹."

        # Phase 5: optional metrics logging
        if self.monitor is not None and hasattr(self.monitor, "log_step"):
            try:
                self.monitor.log_step(
                    turns=len(history),
                    toxicity=toxicity,
                    diagnosis=diagnosis,
                    state="N/A",
                    action="GENERATE",
                )
            except Exception:
                pass

        return raw_response
