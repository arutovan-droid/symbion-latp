from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from .metrics import LATPMetricsLogger

from .ral_module import RalModule


@dataclass
class Crystal:
    """Compressed context - no more than 512 tokens, no less than meaning."""
    core_theses: List[str]
    librarium_refs: List[str]
    entropy_hash: str
    timestamp: datetime


@dataclass
class CoreSession:
    """Minimal representation of a distilled session for Librarium / Airlock."""
    summary: str
    main_theses: List[str]


class ContextPoisoningScorer:
    """
    Composite scorer for context poisoning.
    """

    def __init__(self, model_window: int = 200000) -> None:
        self.WINDOW_MAX = model_window
        self.CRITICAL_THRESHOLD = 0.62
        self.SULTAN_THRESHOLD = 0.31

    def calculate_resonance(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 5:
            return 0.0
        hashes = [hashlib.md5(s.encode("utf-8")).hexdigest()[:8] for s in sentences]
        unique_ratio = len(set(hashes)) / len(hashes)
        return 1.0 - unique_ratio

    def _measure_sultan_index(self, text: str) -> float:
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
        return min(count / 50.0, 1.0)

    def _compare_lexical_similarity(
        self, recent: List[Dict[str, Any]], older: List[Dict[str, Any]]
    ) -> float:
        return 0.0

    def _check_anchor_drift(self, history: List[Dict[str, Any]]) -> float:
        return 0.0

    def score_toxicity(self, history: List[Dict[str, Any]]) -> Tuple[float, str]:
        total_tokens = sum(int(h.get("tokens", 0)) for h in history)
        usage_ratio = total_tokens / max(self.WINDOW_MAX, 1)

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


class CrystalCompressor:
    """Minimal compressor to build CoreSession objects from history."""

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
        tail = texts[-5:]
        joined = " ".join(tail)
        summary = joined[:1000]
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
        system_prompt = full_history[0] if full_history else {}
        last_question = full_history[-1] if full_history else {}

        core_session = self.compressor.distill_semantic_core(
            full_history[1:-1],
            target_tokens=300,
        )

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

        crystal_summary = (
            core_session.summary
            if len(core_session.summary) < 400
            else core_session.summary[:397] + "..."
        )

        crystal_system_content = f"[LATP Crystal] ID:{crystal_id} | Core: {crystal_summary}"

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

    def __init__(self, librarium_client: Any, vector_client: Any | None = None) -> None:
        self.librarium = librarium_client
        self.vector = vector_client

    def generate_bridge(self, current_topic: str) -> Optional[str]:
        if not self.vector:
            return None

        results = self.vector.search_similar(
            query=current_topic,
            top_k=1,
            min_score=0.1,
        )
        if not results:
            return None

        entry_id, score = results[0]
        if score <= 0.0:
            return None

        target_text = self.vector.get_text(entry_id).strip()
        if not target_text:
            return None

        return (
            "Мы уже зафиксировали родственный контекст в Librarium. "
            "Сравни текущую тему с этим фрагментом и найди структурную аналогию, "
            "а не поверхностное совпадение:\n\n"
            f"---\n{target_text}\n---"
        )


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
        sultan = self._scorer._measure_sultan_index(response)
        if sultan > self._scorer.SULTAN_THRESHOLD:
            return (
                False,
                f"BLOCK: Sultan Index {sultan:.2f} > {self._scorer.SULTAN_THRESHOLD}",
            )

        resonance = self._scorer.calculate_resonance(response)
        if resonance > 0.3:
            return False, f"RESET: Resonant collapse {resonance:.2f}"

        _ = crystal
        return True, "PASS"
class LATP_WrappedEngine:
    """Операционная система теперь управляет памятью."""

    def __init__(
        self,
        base_model: Any,
        librarium: Any,
        ral: Any | None = None,
        vector_librarium: Any | None = None,
    ) -> None:
        """
        LATP wrapper around a base LLM model.

        - base_model: must implement .generate(history: list[dict]) -> str
        - librarium: object with .store(...) / .retrieve(id) for crystals
        - ral: optional RalModule instance for numeric drift detection
        - vector_librarium: optional VectorLibrarium for lateral shifts
        """
        self.model = base_model
        self.librarium = librarium
        self.scorer = ContextPoisoningScorer()
        self.airlock = AirlockModule(librarium_client=librarium, ral=ral)
        self.sorbet = LateralShiftEngine(librarium_client=librarium)
        self.watchdog = WatchdogModule()
        self.ral = ral  # RalModule or None

    def generate(self, history: List[Dict[str, Any]]) -> str:
        """
        Единственная точка входа. Все остальные вызовы — приватны.

        history — список сообщений вида {"role": "...", "content": "...", "tokens": int?}
        """

        # Работаем с копией, чтобы не ломать исходный список вызывающего кода
        local_history: List[Dict[str, Any]] = list(history)

        # ---------------- Фаза 1: Диагностика контекста ----------------
        if len(local_history) % 3 == 0 and local_history:
            toxicity, diagnosis = self.scorer.score_toxicity(local_history)

            # CRITICAL: срабатывает Airlock (шлюзовая камера)
            if toxicity > self.scorer.CRITICAL_THRESHOLD or "CRITICAL" in diagnosis:
                local_history, crystal = self.airlock.sanitize_session(local_history)
                print(
                    f"[LATP] {diagnosis}. Airlock activated. Crystal: {crystal.entropy_hash}"
                )

            # Профилактическая зона: 0.5 < toxicity <= CRITICAL_THRESHOLD
            elif 0.5 < toxicity <= self.scorer.CRITICAL_THRESHOLD:
                last_content = local_history[-1].get("content", "")
                bridge = self.sorbet.generate_bridge(str(last_content))
                if bridge:
                    local_history.append(
                        {
                            "role": "system",
                            "content": f"[LATP Lateral] {bridge}",
                        }
                    )
                    print("[LATP] Lateral shift injected.")

        # ---------------- Фаза 2: Генерация базовой моделью ----------------
        raw_response = self.model.generate(local_history)

        # RalModule: фиксация чисел и проверка на numeric drift (если подключён)
        if self.ral is not None:
            # запоминаем, какие числа модель публично зафиксировала
            self.ral.ingest_turn("assistant", raw_response)
            drift_q = self.ral.verify_drift(raw_response)
            if drift_q:
                # если есть расхождение, задаём уточняющий вопрос и пересчитываем
                local_history.append({"role": "user", "content": drift_q})
                raw_response = self.model.generate(local_history)

        # ---------------- Фаза 3: Валидация Watchdog’ом ----------------
        # Ищем последний тег кристалла в истории
        crystal_tag = next(
            (
                m
                for m in local_history
                if "[LATP Crystal]" in str(m.get("content", ""))
            ),
            None,
        )

        if crystal_tag:
            try:
                # Формат тега: "[LATP Crystal] ID:<id> | Core: ..."
                tag_content = str(crystal_tag["content"])
                crystal_id = tag_content.split("ID:")[1].split("|")[0].strip()
                crystal = self.librarium.retrieve(crystal_id)
            except Exception:
                crystal = None
        else:
            crystal = None

        is_valid, command = self.watchdog.scan(raw_response, crystal)

        if not is_valid:
            print(f"[LATP] {command}")
            return self._fallback_response(local_history, command)

        return raw_response

    def _fallback_response(
        self,
        history: List[Dict[str, Any]],
        reason: str,
    ) -> str:
        """
        Когда модель провалила валидацию, мы отвечаем сами
        и просим пользователя переформулировать запрос.
        """
        return (
            f"[LATP HALT] {reason}\n"
            "Ваша задача: переформулировать вопрос, опираясь на Librarium. "
            "Ключевая ошибка: отход от фактов."
        )



class FakeModel:
    """Tiny fake model for tests / examples."""

    def generate(self, history: List[Dict[str, Any]]) -> str:
        last_user = next(
            (m for m in reversed(history) if m.get("role") == "user"), None
        )
        if last_user:
            return f"FAKE: echo -> {last_user.get('content', '')}"
        return "FAKE: no user content"


class FakeLibrarium:
    """In-memory Librarium for local tests."""

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
