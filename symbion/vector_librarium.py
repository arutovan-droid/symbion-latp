# symbion/vector_librarium.py

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .latp_core import CoreSession

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Very simple tokenizer: split on word characters and lowercase."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def _vectorize(text: str) -> Dict[str, float]:
    """
    Build a very simple bag-of-words vector with L2 normalization.

    This is not a real embedding model, but it's enough for:
    - similarity search in tests,
    - basic isomorphy / topic detection in v1.2.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}

    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    # TF-like weighting
    vec: Dict[str, float] = {t: float(c) for t, c in counts.items()}

    # L2-normalize
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    for t in vec:
        vec[t] /= norm

    return vec


def _cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse word vectors."""
    if not v1 or not v2:
        return 0.0
    # Iterate over the smaller dict for speed
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    dot = 0.0
    for t, w in v1.items():
        if t in v2:
            dot += w * v2[t]
    return dot


@dataclass
class VectorEntry:
    """Single entry stored in the VectorLibrarium."""

    id: str
    text: str
    vector: Dict[str, float]


class VectorLibrarium:
    """
    Minimal in-memory vector Librarium.

    v1 design goals:
    - no external dependencies (no faiss, no chroma)
    - simple cosine similarity on bag-of-words vectors
    - CoreSession-friendly interface
    """

    def __init__(self) -> None:
        self._entries: Dict[str, VectorEntry] = {}

    def store_core_session(self, core_session: CoreSession) -> str:
        """
        Store a CoreSession and return its id.

        We use the summary + main_theses as the text payload.
        """
        text_parts: List[str] = []
        if core_session.summary:
            text_parts.append(core_session.summary)
        text_parts.extend(core_session.main_theses)
        text = " ".join(text_parts).strip() or "empty"

        cid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        vec = _vectorize(text)

        self._entries[cid] = VectorEntry(id=cid, text=text, vector=vec)
        return cid

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Return a list of (id, score) for entries similar to the query.

        Results are sorted by descending score.
        """
        qvec = _vectorize(query)
        if not qvec or not self._entries:
            return []

        scores: List[Tuple[str, float]] = []
        for entry_id, entry in self._entries.items():
            score = _cosine_sim(qvec, entry.vector)
            if score >= min_score:
                scores.append((entry_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_text(self, entry_id: str) -> str:
        """Return stored text for a given id, or empty string if missing."""
        entry = self._entries.get(entry_id)
        return entry.text if entry else ""
