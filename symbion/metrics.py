"""
Simple metrics logger for LATP.

Writes one JSON line per event into latp_metrics.jsonl.
This is intentionally minimal and file-based, so you can
later feed it into Jupyter, Prometheus, etc.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LATPEvent:
    """Single LATP event snapshot."""

    kind: str                # e.g. "diagnostic", "lateral_shift", "watchdog_block"
    session_id: str          # any id, for now we can use "default"
    timestamp: float         # unix time
    toxicity: Optional[float] = None
    diagnosis: Optional[str] = None
    lateral_used: bool = False
    watchdog_blocked: bool = False
    extra: Dict[str, Any] = None


class LATPMetricsLogger:
    """
    Thread-safe append-only JSONL logger.

    Each call to `log_event` appends one line with a JSON dict.
    """

    def __init__(self, file_path: str = "latp_metrics.jsonl", session_id: str = "default") -> None:
        self.path = Path(file_path)
        self.session_id = session_id
        self._lock = threading.Lock()

    def log_event(self, event: LATPEvent) -> None:
        """Append one event as JSON line to the metrics file."""
        payload = asdict(event)
        # Ensure session_id and timestamp are always present
        if not payload.get("session_id"):
            payload["session_id"] = self.session_id
        if not payload.get("timestamp"):
            payload["timestamp"] = time.time()
        if payload.get("extra") is None:
            payload["extra"] = {}

        line = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            # parent directory may not exist (e.g., "logs/latp_metrics.jsonl")
            if self.path.parent and not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # Convenience helpers

    def log_diagnostic(self, toxicity: float, diagnosis: str, session_id: Optional[str] = None) -> None:
        self.log_event(
            LATPEvent(
                kind="diagnostic",
                session_id=session_id or self.session_id,
                timestamp=time.time(),
                toxicity=toxicity,
                diagnosis=diagnosis,
            )
        )

    def log_lateral_shift(self, session_id: Optional[str] = None) -> None:
        self.log_event(
            LATPEvent(
                kind="lateral_shift",
                session_id=session_id or self.session_id,
                timestamp=time.time(),
                lateral_used=True,
            )
        )

    def log_watchdog_block(self, reason: str, session_id: Optional[str] = None) -> None:
        self.log_event(
            LATPEvent(
                kind="watchdog_block",
                session_id=session_id or self.session_id,
                timestamp=time.time(),
                watchdog_blocked=True,
                extra={"reason": reason},
            )
        )
