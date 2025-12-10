"""
OpenAI adapter for LATP_WrappedEngine.

This example shows how to:
- wrap an OpenAI chat model into the FakeModel-like interface,
- plug it into LATP_WrappedEngine,
- keep using Symbion-style `history` (list[dict] with role/content).

Requirements:
- `pip install openai`
- environment variable OPENAI_API_KEY must be set.

WARNING:
This is a minimal example, not production code.
"""

import os
from typing import Any, Dict, List

from openai import OpenAI

from symbion.latp_core import LATP_WrappedEngine, FakeLibrarium
from symbion.ral_module import RalModule
from symbion.vector_librarium import VectorLibrarium, CoreSession


class OpenAIChatModel:
    """
    Thin wrapper around OpenAI Chat Completion API
    to match the interface expected by LATP_WrappedEngine.

    It implements:
        generate(history: List[Dict[str, Any]]) -> str
    where `history` is a sequence of dicts with keys:
        - role: "system" | "user" | "assistant"
        - content: str
    """

    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, history: List[Dict[str, Any]]) -> str:
        """
        Convert Symbion-style history into OpenAI Chat API format,
        then return assistant's reply as plain text.
        """
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            messages.append({"role": role, "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
        )

        # OpenAI v1 returns response choices like this:
        # response.choices[0].message.content
        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            return ""
        return choice.message.content


def build_openai_latp_engine() -> LATP_WrappedEngine:
    """
    Build a LATP_WrappedEngine that uses:
    - OpenAIChatModel as the base model,
    - FakeLibrarium for factual Librarium (for demo),
    - RalModule for numeric drift guard,
    - VectorLibrarium for lateral isomorphy search.
    """
    base_model = OpenAIChatModel()
    factual_librarium = FakeLibrarium()
    ral = RalModule()
    vector_librarium = VectorLibrarium()

    # Seed the VectorLibrarium with a couple of abstract "memory" concepts
    core_stone = CoreSession(
        summary="Khachkar as stone memory and encoded narrative.",
        main_theses=[
            "Khachkars store layered symbolic memory in stone carvings",
            "Stone crosses function as long-term cultural code",
        ],
    )
    core_dna = CoreSession(
        summary="DNA as a blind yet precise memory carrier.",
        main_theses=[
            "DNA stores biological information across generations",
            "Genome is a persistent memory independent of any ego",
        ],
    )

    vector_librarium.store_core_session(core_stone)
    vector_librarium.store_core_session(core_dna)

    engine = LATP_WrappedEngine(
        base_model=base_model,
        librarium=factual_librarium,
        ral=ral,
        vector_librarium=vector_librarium,
    )
    return engine


def main() -> None:
    engine = build_openai_latp_engine()

    history: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are Symbion, an OS-level orchestrator over LLMs. "
                "You always try to keep context clean and focused on the task."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain in simple terms how context poisoning can make a long chat "
                "with an AI model dumber over time. Then give 3 concrete symptoms."
            ),
            "tokens": 32,
        },
    ]

    print("[DEMO: OpenAI adapter] Calling LATP_WrappedEngine.generate(...)")
    reply = engine.generate(history)
    print("\n[DEMO: OpenAI adapter] Final reply:\n")
    print(reply)


if __name__ == "__main__":
    main()
