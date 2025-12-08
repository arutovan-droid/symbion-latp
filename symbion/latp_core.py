# symbion/latp_core.py

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Protocol


# === CRYSTAL / 1.0 ===

@dataclass
class Crystal:
    """
    Compressed semantic state of the session.
    Should be small (~512 tokens) but preserve meaning.
    """
    core_theses: List[str]
    librarium_refs: List[str]
    entropy_hash: str
    timestamp: datetime


# === Core session representation (for compression) ===

@dataclass
class CoreSession:
    """
    Result of semantic distillation of a long history.
    This is an intermediate structure before turning into a Crystal.
    """
    main_theses: List[str]
    summary: str
    raw_text: str


class LibrariumClient(Protocol):
    """
    Minimal protocol for any Librarium backend.
    Real implementation can be SQLite, vector DB, etc.
    """
    def store(self, core_session: CoreSession) -> str:
        ...

    def retrieve(self, crystal_id: str) -> Optional[Crystal]:
        ...

    def find_isomorphic(self, vector: str, domain_shift: bool = True):
        ...


class CrystalCompressor:
    """
    Very naive compressor: in production this should use embeddings and summarization.
    For now, we just slice text and take first lines as theses.
    """

    def distill_semantic_core(
        self,
        history: List[Dict],
        target_tokens: int = 300,
    ) -> CoreSession:
        # Concatenate all contents from history
        text = "\n".join(h.get("content", "") for h in history)

        # Very rough summary: cut approximately by characters (token ~= 4 chars)
        summary = text[: target_tokens * 4]

        # Extract first non-empty lines as theses
        theses = [
            line.strip()
            for line in summary.split("\n")
            if line.strip()
        ][:5]

        return CoreSession(main_theses=theses, summary=summary, raw_text=text)


# === Context Poisoning Scorer (LATP core) ===

class ContextPoisoningScorer:
    """
    Computes composite toxicity score for the current history.

    Components:
    - usage_ratio: how much of the context window is used
    - lexical_drift: simplification / degradation of vocabulary
    - sultan_score: amount of 'water' / fluff
    - anchor_drift: drift from system prompt / core theses
    - resonance_collapse: self-similarity of the last answer (resonant collapse)
    """

    def __init__(self, model_window: int = 200_000):
        self.WINDOW_MAX = model_window
        self.CRITICAL_THRESHOLD = 0.62
        self.SULTAN_THRESHOLD = 0.31

    def calculate_resonance(self, text: str) -> float:
        """
        Estimates self-similarity of the answer:
        repeated sentence-level patterns -> resonant collapse risk.
        """
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 5:
            return 0.0

        hashes = [hashlib.md5(s.encode()).hexdigest()[:8] for s in sentences]
        unique_ratio = len(set(hashes)) / len(hashes)
        # If uniqueness < 70% -> model is stuck in its own patterns
        return max(0.0, 1.0 - unique_ratio)

    def score_toxicity(self, history: List[Dict]) -> Tuple[float, str]:
        """
        Returns (toxicity_score, diagnosis_label).

        toxicity > 1.0      -> hard reset recommended
        toxicity > 0.62     -> critical context poisoning
        0.5 < toxicity <=   -> warning zone
        """
        total_tokens = sum(h.get("tokens", 0) for h in history)
        usage_ratio = total_tokens / self.WINDOW_MAX if self.WINDOW_MAX else 0.0

        if len(history) > 10:
            recent = history[-5:]
            older = history[-10:-5]
            lexical_drift = self._compare_lexical_similarity(recent, older)
        else:
            lexical_drift = 0.0

        last_response = history[-1].get("content", "") if history else ""
        sultan_score = self._measure_sultan_index(last_response)
        anchor_drift = self._check_anchor_drift(history)
        resonance_collapse = self.calculate_resonance(last_response)

        toxicity = (
            usage_ratio * 0.4 +
            lexical_drift * 0.2 +
            sultan_score * 0.2 +
            anchor_drift * 0.1 +
            resonance_collapse * 0.1
        )

        if toxicity > 0.75:
            diagnosis = "CRITICAL: Resonant collapse"
        elif toxicity > self.CRITICAL_THRESHOLD:
            diagnosis = "WARNING: Context oversaturation"
        elif sultan_score > self.SULTAN_THRESHOLD:
            diagnosis = "WARNING: Sultan Index exceeded"
        elif resonance_collapse > 0.3:
            diagnosis = "WARNING: Self-hypnosis detected"
        else:
            diagnosis = "NORMAL"

        return toxicity, diagnosis

    # --- placeholders that can be improved later ---

    def _compare_lexical_similarity(
        self,
        recent: List[Dict],
        older: List[Dict],
    ) -> float:
        """
        TODO: compare lexical / embedding similarity.
        For now, returns 0.0 as a placeholder.
        """
        return 0.0

    def _measure_sultan_index(self, text: str) -> float:
        """
        Very rough Sultan Index: approximate 'fluff / water' amount in text.
        Looks for cliche phrases and normalizes by text length.
        """
        text_l = text.lower()
        water_markers = [
            "в общем-то",
            "как бы",
            "понимаешь",
            "важно понимать",
            "в конечном счете",
            "на самом деле",
            "мы все знаем",
            "в целом",
            "можно сказать",
            "следует отметить",
        ]
        water_count = sum(text_l.count(marker) for marker in water_markers)
        norm = max(len(text_l) // 50, 1)
        return min(water_count / norm, 1.0)

    def _check_anchor_drift(self, history: List[Dict]) -> float:
        """
        TODO: compare last answers with system prompt / core theses.
        For now, returns 0.0 as a placeholder.
        """
        return 0.0


# === Airlock Module (Module A) ===

class AirlockModule:
    """
    Airlock: takes full history and returns a clean context + Crystal.
    """

    def __init__(self, librarium_client: LibrariumClient):
        self.librarium = librarium_client
        self.compressor = CrystalCompressor()

    def sanitize_session(
        self,
        full_history: List[Dict],
    ) -> Tuple[List[Dict], Crystal]:
        """
        Produce a clean context for the model and a Crystal for Librarium.
        """
        system_prompt = full_history[0] if full_history else {
            "role": "system",
            "content": "",
        }
        last_message = full_history[-1] if full_history else {
            "role": "user",
            "content": "",
        }

        core_session = self.compressor.distill_semantic_core(
            full_history[1:-1],
            target_tokens=300,
        )

        crystal_id = self.librarium.store(core_session)

        crystal = Crystal(
            core_theses=core_session.main_theses,
            librarium_refs=[crystal_id],
            entropy_hash=hashlib.sha256(str(full_history).encode()).hexdigest()[:16],
            timestamp=datetime.utcnow(),
        )

        clean_context = [
            system_prompt,
            {
                "role": "system",
                "content": f"[LATP Crystal] ID:{crystal_id} | Core: {core_session.summary}",
            },
            last_message,
        ]

        return clean_context, crystal


# === Lateral Shift Engine (Module B) ===

class LateralShiftEngine:
    """
    Lateral shift generator: finds isomorphic topics and builds a bridge prompt.
    """

    def __init__(self, librarium_client: LibrariumClient):
        self.librarium = librarium_client

    def generate_bridge(self, current_topic: str) -> Optional[str]:
        """
        Find an isomorphic topic in Librarium and construct a bridge question.
        If nothing found, returns None.
        """
        isomorphic = self.librarium.find_isomorphic(
            vector=current_topic,
            domain_shift=True,
        )
        if not isomorphic:
            return None

        name = getattr(isomorphic, "name", "another domain")
        arch_type = getattr(isomorphic, "arch_type", "structure")
        field = getattr(isomorphic, "field", "X")
        method = getattr(isomorphic, "method", "analysis")

        bridge_prompts = [
            (
                f"We explored {current_topic}. Now compare it with {name}. "
                f"Do not repeat yourself, look for deep structural analogy."
            ),
            (
                f"Topic {current_topic} has archetype '{arch_type}'. "
                f"How does this archetype appear in {field}? "
                f"Answer through the lens of {method}."
            ),
        ]
        index = abs(hash(current_topic)) % len(bridge_prompts)
        return bridge_prompts[index]


# === Watchdog Module (Module C) ===

class WatchdogModule:
    """
    Watchdog: validates a single response against Sultan Index,
    Crystal fidelity, and resonance.
    """

    def __init__(self):
        self.scorer = ContextPoisoningScorer()

    def _check_fidelity(self, response: str, crystal: Crystal) -> bool:
        """
        Very rough fidelity check:
        response must at least partially mention Crystal core theses.
        """
        if not crystal.core_theses:
            return True

        text_l = response.lower()
        hits = sum(
            1
            for t in crystal.core_theses
            if t and t.lower().split()[0] in text_l
        )
        return hits >= max(1, len(crystal.core_theses) // 3)

    def scan(
        self,
        response: str,
        crystal: Optional[Crystal],
    ) -> Tuple[bool, str]:
        """
        Returns (is_valid, command_message).
        If is_valid=False, the upper layer decides how to handle it.
        """
        sultan = self.scorer._measure_sultan_index(response)
        if sultan > self.scorer.SULTAN_THRESHOLD:
            return False, f"BLOCK: Sultan Index {sultan:.2f} > {self.scorer.SULTAN_THRESHOLD:.2f}"

        if crystal is not None and not self._check_fidelity(response, crystal):
            return False, "BLOCK: Drift from Crystal. Librarium context ignored."

        resonance = self.scorer.calculate_resonance(response)
        if resonance > 0.3:
            return False, f"RESET: Resonant collapse {resonance:.2f}"

        return True, "PASS"


# === BaseModel protocol for wrapped engines ===

class BaseModel(Protocol):
    """
    Minimal protocol all models must implement to be used by LATP_WrappedEngine.
    """

    name: str

    def generate(self, history: List[Dict], **kwargs) -> str:
        ...


# === Dissonance Probe (Module D) ===

class DissonanceProbe:
    """
    Dissonance probe: asks the model to attack its own previous answer.
    Used when resonance is high.
    """

    def __init__(self, base_model: BaseModel):
        self.model = base_model

    def challenge(self, history: List[Dict], last_answer: str) -> str:
        """
        Builds a critic prompt and returns a 'wedge question'
        to break self-hypnosis of the model.
        """
        probe_prompt = {
            "role": "system",
            "content": (
                "You are now a critic of the previous answer.\n"
                "1) Find the most questionable claim.\n"
                "2) Formulate one concrete question that could falsify it.\n"
                "Do not defend the answer, attack it."
            ),
        }
        critic_history = history + [
            {"role": "assistant", "content": last_answer},
            probe_prompt,
        ]
        probe_question = self.model.generate(critic_history)
        return probe_question


# === Simple fake implementations for examples and tests ===

class FakeLibrarium:
    """
    Minimal in-memory placeholder for LibrariumClient.
    Useful for tests and examples.
    """

    def store(self, core_session: CoreSession) -> str:
        # In real implementation this should persist core_session
        return "crystal-test-id"

    def retrieve(self, crystal_id: str) -> Optional[Crystal]:
        # No persistent storage in this fake implementation
        return None

    def find_isomorphic(self, vector: str, domain_shift: bool = True):
        # Always returns None: no isomorphic knowledge in fake mode
        return None


class FakeModel:
    """
    Minimal echo-like model for tests and examples.
    """

    name = "fake-model"

    def generate(self, history: List[Dict], **kwargs) -> str:
        # Find last user message and echo it back
        last_user = next(
            (h for h in reversed(history) if h.get("role") == "user"),
            None,
        )
        q = last_user["content"] if last_user else ""
        return f"Echo: {q}"


# === LATP_WrappedEngine ===

class LATP_WrappedEngine:
    """
    LATP wrapper around a single base model.
    This is not full multi-model orchestration, only LATP behavior (v1.1).
    """

    def __init__(self, base_model: BaseModel, librarium: LibrariumClient):
        self.model = base_model
        self.librarium = librarium
        self.scorer = ContextPoisoningScorer()
        self.airlock = AirlockModule(librarium)
        self.sorbet = LateralShiftEngine(librarium)
        self.watchdog = WatchdogModule()
        self.dissonance = DissonanceProbe(base_model)
        self.request_counter = 0
        self.REM_PERIOD = 10
        self.REM_MAX_LEN = 512

    def _extract_crystal_id(self, history: List[Dict]) -> Optional[str]:
        """
        Parse last LATP Crystal tag from history and extract ID.
        """
        for h in history:
            content = h.get("content", "")
            if isinstance(content, str) and "[LATP Crystal]" in content and "ID:" in content:
                return content.split("ID:")[1].split()[0]
        return None

    def _is_rem_cycle(self) -> bool:
        """
        REM cycle: every N-th request we allow unvalidated 'hypothesis mode'.
        """
        return self.request_counter % self.REM_PERIOD == 0

    def generate(self, history: List[Dict]) -> str:
        """
        Main entrypoint: applies LATP logic before and after model generation.
        """
        self.request_counter += 1

        toxicity, diagnosis = self.scorer.score_toxicity(history)

        crystal: Optional[Crystal] = None

        if toxicity > self.scorer.CRITICAL_THRESHOLD or "CRITICAL" in diagnosis:
            history, crystal = self.airlock.sanitize_session(history)
            print(f"[LATP] {diagnosis}. Airlock activated. Crystal: {crystal.entropy_hash}")
        elif toxicity > 0.5:
            bridge = self.sorbet.generate_bridge(history[-1].get("content", ""))
            if bridge:
                history.append({"role": "system", "content": f"[LATP Lateral] {bridge}"})
                print("[LATP] Lateral shift injected.")

        raw_response = self.model.generate(history)

        # REM: deliberately bypass watchdog but mark the answer as hypotheses
        if self._is_rem_cycle():
            return "[LATP REM] (hypotheses, no validation)\n" + raw_response[: self.REM_MAX_LEN]

        if crystal is None:
            cid = self._extract_crystal_id(history)
            if cid:
                crystal = self.librarium.retrieve(cid)

        is_valid, command = self.watchdog.scan(raw_response, crystal)

        if not is_valid:
            print(f"[LATP] {command}")
            resonance = self.scorer.calculate_resonance(raw_response)
            if resonance > 0.3:
                probe_q = self.dissonance.challenge(history, raw_response)
                return (
                    f"[LATP DISSONANCE] {command}\n"
                    f"Re-check the previous answer. Wedge question:\n{probe_q}"
                )
            return self._fallback_response(history, command)

        return raw_response

    def _fallback_response(self, history: List[Dict], reason: str) -> str:
        """
        Failsafe answer when watchdog rejects the model output.
        """
        cid = self._extract_crystal_id(history)
        hint = f"\nRelated Librarium crystal: {cid}" if cid else ""
        return (
            f"[LATP HALT] {reason}\n"
            f"Please rephrase your task in one sentence, "
            f"grounded in known facts from Librarium."
            f"{hint}"
        )
