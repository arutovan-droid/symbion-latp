from __future__ import annotations

"""
LATP CLI demo.

Minimal interactive chat loop using:
- LATP_WrappedEngine (Airlock + LateralShift + Watchdog + RalModule)
- FakeModel / FakeLibrarium (no external APIs)
- optional metrics logging to JSONL file.

Run:
    python examples/latp_cli_demo.py
Type:
    exit   -> to quit
    /reset -> to start a fresh session (keeps only the system prompt)
"""

from typing import List, Dict

from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium
from symbion.ral_module import RalModule
from symbion.metrics import LATPMetricsLogger


def build_engine() -> LATP_WrappedEngine:
    """Build a LATP engine with FakeModel and in-memory Librarium."""
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
        ral=RalModule(),  # numeric drift guard
    )
    return engine


def main() -> None:
    """Run an interactive terminal chat powered by LATP."""
    print("=== LATP CLI demo ===")
    print("Type your messages and press Enter.")
    print("Commands: 'exit' to quit, '/reset' to clear the session.\n")

    engine = build_engine()

    # Simple metrics logger writing to JSONL
    metrics = LATPMetricsLogger(
        file_path="latp_cli_metrics.jsonl",
        session_id="cli_demo",
    )

    # Initial system prompt – Symbion personality + hint about LATP
    history: List[Dict] = [
        {
            "role": "system",
            "content": "You are Symbion, protected by LATP (Airlock + LateralShift + Watchdog).",
            "tokens": 14,
        }
    ]

    try:
        while True:
            user_text = input("you> ").strip()

            if not user_text:
                continue

            # Exit command
            if user_text.lower() in {"exit", "quit"}:
                print("symbion> Bye. LATP session closed.")
                break

            # Reset command: drop all history except system prompt
            if user_text.startswith("/reset"):
                history = [history[0]]
                print("[LATP CLI] Session reset. Airlock will create a fresh Crystal on long chats.")
                continue

            # Append user turn
            history.append(
                {
                    "role": "user",
                    "content": user_text,
                    "tokens": len(user_text.split()),
                }
            )

            # Optional: run toxicity diagnostics BEFORE generation
            toxicity, diagnosis = engine.scorer.score_toxicity(history)
            metrics.log_diagnostic(
                toxicity=toxicity,
                diagnosis=diagnosis,
            )

            # Main LATP pipeline (Airlock + Lateral shift + Watchdog + Ral)
            reply = engine.generate(history)

            # Append assistant turn back into history
            history.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "tokens": len(reply.split()),
                }
            )

            print(f"symbion> {reply}\n")

    except KeyboardInterrupt:
        print("\n[CTRL-C] Interrupted. LATP session closed.")


if __name__ == "__main__":
    main()
