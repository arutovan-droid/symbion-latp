"""
RalModule – "real-number" cooler.

It watches for numeric drift (e.g. 37 * 42 suddenly becomes 1400
after 50 turns) and forces the LLM to recompute explicit calculations
instead of repeating a cached, wrong value.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Simplified numeric pattern: integers and decimals with optional comma
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Reserved for future use (units, %, etc.)
_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*%|°[CF]?|руб\.|usd|€|кг|г|м\b|км\b|см\b",
    re.IGNORECASE,
)


class RalModule:
    """
    Numeric watchdog with minimal coupling to LATP.

    v1 design (простая и честная):

    - Мы храним просто список чисел, которые модель уже "заявила" как ответ.
    - Для каждого нового числа ищем самое близкое из кристалла.
    - Если относительное отличие > rel_threshold (по умолчанию 5%) —
      считаем это "цифровым дрейфом" и просим модель пересчитать.
    """

    def __init__(self, rel_threshold: float = 0.05) -> None:
        # relative difference threshold: 0.05 = 5% drift
        self.rel_threshold = rel_threshold
        # list of canonical numeric values seen in assistant's outputs
        self._values: List[float] = []

    # ---------- public API, to be called from LATP_WrappedEngine ----------

    def ingest_turn(self, role: str, text: str) -> None:
        """
        Scan a new message and update the numeric "digital crystal".

        For v1 we only track assistant outputs: what the model "publicly commits" to.
        """
        if role != "assistant":
            return

        nums = self._extract_numbers(text)
        # Append all numbers; duplicates не страшны для простого поиска ближайшего.
        self._values.extend(nums)

    def verify_drift(self, candidate_text: str) -> Optional[str]:
        """
        Check if candidate_text contains numbers that contradict the stored crystal.

        Strategy:
        - если нет сохранённых чисел — просто выходим;
        - для каждого числа в candidate_text:
            * ищем ближайшее число из кристалла;
            * считаем относительное отличие |n - v| / max(|v|, 1.0);
            * если > rel_threshold — считаем дрейфом и возвращаем wedge-вопрос.
        """
        if not self._values:
            return None

        nums = self._extract_numbers(candidate_text)
        if not nums:
            return None

        for n in nums:
            closest = min(self._values, key=lambda v: abs(v - n))
            denom = max(abs(closest), 1.0)
            rel_diff = abs(n - closest) / denom

            # Небольшие флуктуации игнорируем, ловим только заметный дрейф.
            if rel_diff >= self.rel_threshold:
                return self._wedge_question(n, closest)

        return None

    def digital_crystal_prompt(self) -> str:
        """
        Small fragment that can be injected into the system prompt / context.

        For now we simply expose the stored values as a list. This is a heuristic
        hint for the model, not a strict contract.
        """
        if not self._values:
            return ""
        unique_vals = sorted(set(self._values))
        lines = [f"- {v}" for v in unique_vals]
        return "[Ral-Crystal]: known numeric anchors:\n" + "\n".join(lines) + "\n"

    # ---------- internals ----------

    def _extract_numbers(self, text: str) -> List[float]:
        return [float(x.replace(",", ".")) for x in _NUMBER_RE.findall(text)]

    def _wedge_question(self, wrong: float, correct: float) -> str:
        """
        Build a "wedge question" that forces the model to recompute the number.
        """
        return (
            f"Ты только что написал {wrong}, но ранее мы установили {correct}. "
            f"Сделай шаг назад и пересчитай это число по шагам, без догадок."
        )
