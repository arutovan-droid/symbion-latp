# symbion/librarium.py

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from .latp_core import Crystal


class SQLiteLibrarium:
    """
    Minimal persistent Librarium implementation backed by SQLite.

    It stores Crystals in a single table:
    - id: short hash of the core session summary
    - data: JSON-encoded Crystal payload
    """

    def __init__(self, db_path: str = "librarium.db") -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crystals (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def store(self, core_session) -> str:
        """
        Store a distilled core_session and return a crystal id.

        core_session is expected to expose:
        - .summary: str
        - .main_theses: list[str]
        """
        summary = getattr(core_session, "summary", "")
        main_theses = getattr(core_session, "main_theses", [])

        cid = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]

        crystal = Crystal(
            core_theses=list(main_theses),
            librarium_refs=[cid],
            entropy_hash=cid,
            timestamp=datetime.utcnow(),
        )

        # Convert Crystal to a JSON-serializable dict
        payload = asdict(crystal)
        payload["timestamp"] = crystal.timestamp.isoformat()

        self.conn.execute(
            "INSERT OR REPLACE INTO crystals (id, data) VALUES (?, ?)",
            (cid, json.dumps(payload)),
        )
        self.conn.commit()
        return cid

    def retrieve(self, crystal_id: str) -> Optional[Crystal]:
        """
        Retrieve a Crystal by id. Returns None if not found.
        """
        row = self.conn.execute(
            "SELECT data FROM crystals WHERE id = ?", (crystal_id,)
        ).fetchone()
        if not row:
            return None

        data = json.loads(row[0])
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return Crystal(**data)

    def close(self) -> None:
        """
        Close the underlying SQLite connection.
        """
        self.conn.close()
