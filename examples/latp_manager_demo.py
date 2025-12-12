"""
LATP Manager demo: show state & action after each user message.

Run:

    (.venv) python examples/latp_manager_demo.py

Commands:
    /reset  – reset session (history + LATPManager state)
    exit    – quit
"""

from __future__ import annotations

from typing import List, Dict

from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium
from symbion.latp_manager import (
    LATPManager,
    LATPState,
    LATPAction,
    EpisodeMessage,
)


def approx_tokens(text: str) -> int:
    """Very rough token approximation: whitespace-based."""
    return max(1, len(text.split()))


def build_engine_and_manager() -> tuple[LATP_WrappedEngine, LATPManager]:
    """Construct a LATP engine + manager for the demo."""
    engine = LATP_WrappedEngine(
        base_model=FakeModel(),
        librarium=FakeLibrarium(),
    )
    manager = LATPManager(engine=engine)
    return engine, manager


def print_banner() -> None:
    print("=== LATP Manager demo ===")
    print("Type your messages and press Enter.")
    print("Commands: 'exit' to quit, '/reset' to clear the session.")
    print()


def main() -> None:
    engine, manager = build_engine_and_manager()
    session_id = "latp-manager-demo"

    # Main LLM history (what we pass into LATP_WrappedEngine)
    history: List[Dict[str, object]] = [
        {
            "role": "system",
            "content": "You are Symbion with LATP manager enabled.",
            "tokens": 8,
        }
    ]

    print_banner()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[DEMO] Exiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("symbion> Bye. LATP manager demo closed.")
            break

        if user_input.strip() == "/reset":
            # Reset local history + LATP manager view of the session.
            history = history[:1]  # keep only system prompt
            manager.reset_session(session_id)
            print("[LATP] Session reset. State -> NORMAL.")
            continue

        # 1) Append user message to history and manager
        user_tokens = approx_tokens(user_input)
        history.append(
            {
                "role": "user",
                "content": user_input,
                "tokens": user_tokens,
            }
        )
        manager.on_message(
            session_id,
            EpisodeMessage(role="user", content=user_input, tokens=user_tokens),
        )

        # 2) Ask LATPManager for a decision
        decision = manager.suggest_action(session_id)

        print(
            f"[LATP] state={decision.state.name} "
            f"action={decision.action.name} "
            f"| {decision.reason}"
        )

        # 3) Optionally orchestrator could react here:
        #    - if decision.action == LATPAction.SHIFT: inject lateral prompt
        #    - if AIRLOCK: reset context, etc.
        #
        # For this demo we just always generate a reply to show that
        # LATPManager does not break normal flow.
        reply = engine.generate(history)
        reply_tokens = approx_tokens(str(reply))

        print(f"symbion> {reply}")

        # 4) Feed assistant reply back into manager
        history.append(
            {
                "role": "assistant",
                "content": reply,
                "tokens": reply_tokens,
            }
        )
        manager.on_message(
            session_id,
            EpisodeMessage(role="assistant", content=str(reply), tokens=reply_tokens),
        )


if __name__ == "__main__":
    main()
