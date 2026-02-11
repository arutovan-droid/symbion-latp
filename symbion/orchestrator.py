# symbion/orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Protocol, Literal

from .latp_core import (
    LATP_WrappedEngine,
    FakeModel,
    FakeLibrarium,
    ContextPoisoningScorer,
)


EngineRole = Literal["primary", "backup", "validator", "archivist"]


@dataclass
def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class ValidationResult:
    """
    Normalized validation result used by the routing policy.

    score: 0.0â€“1.0 (higher is better)
    reason: human-readable explanation
    is_hallucination: validator suspects hallucination
    is_resonant_collapse: strong self-similarity / self-hypnosis
    requires_swap: hard signal that we should try a different engine
    """

    score: float
    reason: str
    is_hallucination: bool = False
    is_resonant_collapse: bool = False
    requires_swap: bool = False


@dataclass
def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class EngineSpec:
    """
    Static specification of an engine inside the orchestrator.

    name: logical name (e.g. "gpt4-primary", "claude-backup")
    role: "primary" | "backup" | "validator" | "archivist"
    priority: lower number means higher preference
    latp_engine: LATP-wrapped engine instance
    tags: optional labels, e.g. ["code", "cheap"]
    """

    name: str
    role: EngineRole
    priority: int
    latp_engine: LATP_WrappedEngine
    tags: List[str] | None = None


@dataclass
def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class OrchestratorContext:
    """
    High-level hints for routing.
    This is intentionally minimal for v1.0.
    """

    domain: Optional[str] = None  # e.g. "code", "math", "law"
    max_latency_ms: Optional[int] = None
    max_cost_level: Optional[str] = None  # e.g. "low", "medium", "high"
    user_id: Optional[str] = None


def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class RoutingPolicy(Protocol):
    """
    Routing policy decides when to swap engines and which backup to choose.
    """

    def should_swap(
        self,
        engine: EngineSpec,
        validation: ValidationResult,
        ctx: OrchestratorContext | None,
    ) -> bool: ...

    def choose_backup(
        self,
        engines: List[EngineSpec],
        reason: str,
        ctx: OrchestratorContext | None,
    ) -> Optional[EngineSpec]: ...


def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class SimpleRoutingPolicy:
    """
    Minimal routing policy implementation.

    - swaps when validation.score < quality_threshold
      or when validation.requires_swap is True
    - chooses the backup engine with the lowest priority value
    """

    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold

    def should_swap(
        self,
        engine: EngineSpec,
        validation: ValidationResult,
        ctx: OrchestratorContext | None,
    ) -> bool:
        if validation.requires_swap:
            return True
        if validation.score < self.quality_threshold:
            return True
        return False

    def choose_backup(
        self,
        engines: List[EngineSpec],
        reason: str,
        ctx: OrchestratorContext | None,
    ) -> Optional[EngineSpec]:
        # Filter only backup engines
        backups = [e for e in engines if e.role == "backup"]
        if not backups:
            return None

        # TODO: use ctx.domain and tags to prefer domain-specific engines
        # For v1.0 we simply choose the lowest-priority backup.
        backups_sorted = sorted(backups, key=lambda e: e.priority)
        return backups_sorted[0]


def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class AnswerValidator(Protocol):
    """
    Validator interface: turns a draft answer into a ValidationResult.
    """

    def validate(
        self,
        engine: EngineSpec,
        history: List[Dict],
        draft_answer: str,
    ) -> ValidationResult: ...


def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class LatpScorerValidator:
    """
    Simple validator based on LATP's ContextPoisoningScorer.

    This does not perform true hallucination detection, but:
    - uses toxicity as a proxy for answer quality,
    - uses resonance to flag potential resonant collapse.
    """

    def __init__(self):
        self.scorer = ContextPoisoningScorer()

    def validate(
        self,
        engine: EngineSpec,
        history: List[Dict],
        draft_answer: str,
    ) -> ValidationResult:
        # Append the draft answer as if it were the last assistant message
        extended_history = history + [
            {"role": "assistant", "content": draft_answer, "tokens": 0}
        ]
        toxicity, diagnosis = self.scorer.score_toxicity(extended_history)
        resonance = self.scorer.calculate_resonance(draft_answer)

        # Convert toxicity into a crude quality score
        # 1.0 = no toxicity, 0.0 = maximal toxicity (clamped)
        score = max(0.0, 1.0 - min(toxicity, 1.0))

        is_resonant = resonance > 0.3
        reason = f"toxicity={toxicity:.2f}, resonance={resonance:.2f}, diag={diagnosis}"

        return ValidationResult(
            score=score,
            reason=reason,
            is_hallucination=False,  # TODO: enrich with real checks
            is_resonant_collapse=is_resonant,
            requires_swap=is_resonant,  # heuristic: swap on strong resonance
        )


def _run_collider_sync(user_text: str, history: list[dict], life_vector: dict | None):
    # Safe sync wrapper for async route_language(). Runs in a new thread if loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import threading
        box: dict = {}

        def _worker():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                box["cog"] = new_loop.run_until_complete(
                    route_language(user_text, {"history": history}, life_vector=life_vector)
                )
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=5)
        return box.get("cog")

    return asyncio.run(route_language(user_text, {"history": history}, life_vector=life_vector))


class ModelOrchestrator:
    """
    SYMBION-RELAY v1.0 orchestration skeleton.

    - keeps a pool of EngineSpec instances
    - uses a RoutingPolicy to decide when and how to swap
    - uses a validator to score draft answers
    """

    def __init__(
        self,
        engines: List[EngineSpec],
        routing_policy: Optional[RoutingPolicy] = None,
        validator: Optional[AnswerValidator] = None,
    ):
        if not engines:
            raise ValueError("At least one engine must be provided")

        self.engines = engines
        self.routing_policy = routing_policy or SimpleRoutingPolicy()
        self.validator = validator or LatpScorerValidator()

        # The active engine is the primary with the lowest priority.
        primary_engines = [e for e in self.engines if e.role == "primary"]
        if not primary_engines:
            raise ValueError("At least one primary engine is required")

        self.active_engine: EngineSpec = sorted(
            primary_engines,
            key=lambda e: e.priority,
        )[0]

    def _select_primary_engine(self) -> EngineSpec:
        """
        Select default primary engine (lowest priority primary).
        """
        primary_engines = [e for e in self.engines if e.role == "primary"]
        if not primary_engines:
            raise RuntimeError("No primary engines available")
        return sorted(primary_engines, key=lambda e: e.priority)[0]

    def get_answer(
        self,
        user_query: str,
        history: List[Dict],
        *,
        ctx: Optional[OrchestratorContext] = None,
    ) -> str:
        """
        Main entrypoint for SYMBION-RELAY.

        - Appends the user query to history
        - Delegates to the active engine
        - Validates the draft answer
        - Optionally performs a one-time swap and retries with a backup
        """
        ctx = ctx or OrchestratorContext()
        full_history = history + [{"role": "user", "content": user_query, "tokens": 0}]

        # Ensure we have a primary engine selected
        if self.active_engine.role != "primary":
            self.active_engine = self._select_primary_engine()
        # Phase 0.5: Orchestrator basis_select (Cognitive Collider)
        try:
            cog = _run_collider_sync(user_query, full_history, life_vector=None)
            if cog is not None:
                cog_dict = cog.model_dump()
                # attach to ctx dynamically (no hard dependency on OrchestratorContext schema)
                try:
                    ctx.cog_lang = cog_dict
                except Exception:
                    pass

                full_history.append({
                    "role": "system",
                    "content": "[Orchestrator CogLang] " + json.dumps(cog_dict, ensure_ascii=False),
                    "cog_lang": cog_dict,
                })
        except Exception:
            pass


        # Phase 1: draft answer from active engine
        draft_answer = self.active_engine.latp_engine.generate(full_history)

        # Phase 2: validation
        validation = self.validator.validate(
            engine=self.active_engine,
            history=full_history,
            draft_answer=draft_answer,
        )

        # Phase 3: routing / swapping decision
        if not self.routing_policy.should_swap(self.active_engine, validation, ctx):
            return draft_answer

        # Try a backup engine
        backup = self.routing_policy.choose_backup(self.engines, validation.reason, ctx)
        if backup is None:
            # No backup available: return draft with a warning prefix
            return (
                "[RELAY WARNING] No backup engine available, returning draft answer.\n"
                + draft_answer
            )

        # Use the backup engine once for this turn
        backup_answer = backup.latp_engine.generate(full_history)
        self.active_engine = backup  # Future turns may continue with backup

        return backup_answer


# === Minimal example wiring for local testing ===


def build_fake_orchestrator() -> ModelOrchestrator:
    """
    Convenience function to build a tiny orchestrator
    using FakeModel + FakeLibrarium from latp_core.

    This is only for examples and tests.
    """
    librarium = FakeLibrarium()

    # Primary engine
    primary_latp = LATP_WrappedEngine(FakeModel(), librarium)
    primary_spec = EngineSpec(
        name="fake-primary",
        role="primary",
        priority=10,
        latp_engine=primary_latp,
        tags=["generic"],
    )

    # Backup engine (same fake model for now)
    backup_latp = LATP_WrappedEngine(FakeModel(), librarium)
    backup_spec = EngineSpec(
        name="fake-backup",
        role="backup",
        priority=20,
        latp_engine=backup_latp,
        tags=["generic"],
    )

    orchestrator = ModelOrchestrator(
        engines=[primary_spec, backup_spec],
        routing_policy=SimpleRoutingPolicy(quality_threshold=0.7),
        validator=LatpScorerValidator(),
    )
    return orchestrator
