
"""
RalModule – "real-number" cooler.

It watches for numeric drift (e.g. 37 * 42 suddenly becomes 1400
after 50 turns) and forces the LLM to recompute explicit calculations
instead of repeating a cached, wrong value.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

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

    It maintains a "digital crystal": a mapping key -> canonical numeric value.
    Keys are coarse-grained (normalized) representations to allow some tolerance.
    """

    def __init__(self, tolerance: float = 1e-3) -> None:
        # tolerance for float comparisons
        self.tolerance = tolerance
        # digital crystal: key -> canonical value
        self.crystal: Dict[str, float] = {}

    # ---------- public API, to be called from LATP_WrappedEngine ----------

    def ingest_turn(self, role: str, text: str) -> None:
        """
        Scan a new message and update the numeric crystal.

        For v1 we only track assistant outputs: what the model "publicly commits" to.
        """
        if role != "assistant":
            return

        for n in self._extract_numbers(text):
            key = self._keyify(n)
            # Last value wins: we assume the most recent, explicitly computed
            # number is the canonical one.
            self.crystal[key] = n

    def verify_drift(self, candidate_text: str) -> Optional[str]:
        """
        Check if candidate_text contains numbers that contradict the crystal.

        If drift is detected, return a "wedge question" that forces the model
        to recompute the value step by step. If no drift is found, return None.
        """
        for raw in _NUMBER_RE.findall(candidate_text):
            n = float(raw.replace(",", "."))
            key = self._keyify(n)
            if key in self.crystal:
                canonical = self.crystal[key]
                if abs(n - canonical) > self.tolerance:
                    return self._wedge_question(n, canonical)
        return None

    def digital_crystal_prompt(self) -> str:
        """
        Small fragment that can be injected into the system prompt / context.

        It encodes the digital crystal as plain text, so that after an Airlock
        reset the model still sees the "sacred numbers" we rely on.
        """
        if not self.crystal:
            return ""
        lines = [f"{k} = {v}" for k, v in self.crystal.items()]
        return "[Ral-Crystal]:\n" + "\n".join(lines) + "\n"

    # ---------- internals ----------

    def _extract_numbers(self, text: str) -> List[float]:
        return [float(x.replace(",", ".")) for x in _NUMBER_RE.findall(text)]

    def _keyify(self, n: float) -> str:
        """
        Produce a coarse key for a numeric value.

        For v1 we normalize magnitude and round to ~3 significant digits.
        This is intentionally simple and may cause collisions – acceptable
        as a heuristic, TODO for future versions.
        """
        if n == 0:
            return "0"
        order = int(math.log10(abs(n)))
        # Normalize to mantissa in [1, 10) and keep 3 significant digits
        mantissa = n / 10**order
        return f"{mantissa:.3g}e{order}"

    def _wedge_question(self, wrong: float, correct: float) -> str:
        """
        Build a "wedge question" that forces the model to recompute the number.
        """
        return (
            f"Ты только что написал {wrong}, но ранее мы установили {correct}. "
            f"Сделай шаг назад и пересчитай это число по шагам, без догадок."
        )
