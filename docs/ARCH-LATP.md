\# LATP Architecture (LUYS Anti-Toxin Protocol)



LATP (LUYS Anti-Toxin Protocol) is a \*\*stateful proxy\*\* between the user and a base LLM.



It does not replace the model; it controls:



\- context length and saturation,

\- lateral shifts to avoid echo loops,

\- numeric reality (digits, dates, simple arithmetic),

\- session-level state (NORMAL / HEAT\_UP / LATERAL\_SHIFT / COOL\_DOWN / AIRLOCK / STABLE).



---



\## 1. State machine



```mermaid

stateDiagram-v2

&nbsp;   \[\*] --> NORMAL



&nbsp;   NORMAL --> HEAT\_UP: entropy ↑ / length ↑

&nbsp;   NORMAL --> DIGITAL\_DRIFT: RalModule mismatch

&nbsp;   NORMAL --> STABLE: metrics OK



&nbsp;   HEAT\_UP --> LATERAL\_SHIFT: prophylactic shift

&nbsp;   HEAT\_UP --> COOL\_DOWN: summarise \& compress

&nbsp;   HEAT\_UP --> AIRLOCK: critical overload



&nbsp;   LATERAL\_SHIFT --> COOL\_DOWN: shift completed

&nbsp;   DIGITAL\_DRIFT --> REALITY\_CHECK: wedge question

&nbsp;   REALITY\_CHECK --> NORMAL: numbers corrected



&nbsp;   COOL\_DOWN --> STABLE

&nbsp;   STABLE --> NORMAL



&nbsp;   AIRLOCK --> NORMAL: start from Crystal

3\. TL;DR



Airlock keeps the model in a "fresh head" regime.



LateralShift prevents echo loops and mode collapse.



RalModule stabilises digits, dates and simple math.



LATPManager glues everything into a small, testable state machine.



You plug LATP in front of any LLM (OpenAI, Claude, local) and get

a session that:



degrades slower,



loops less,



lies about numbers much rarer.

# LATP Architecture — LUYS Anti-Toxin Protocol

This document describes the internal architecture of the LATP core inside `symbion-latp`.

LATP is a **cooling and hygiene layer** around a base LLM.  
It does **not** change model weights — it controls **context, memory and intervention logic**.

---

## 1. High-level picture

From the outside, LATP looks like this:

```text
User  →  Orchestrator  →  LATP_WrappedEngine  →  Base LLM
                          ↑        ↑      ↑
                          │        │      │
                     Airlock   Sorbet   RalModule
                       │        │
                   Librarium  VectorLibrarium
The main entry point is:

python
Копировать код
from symbion.latp_core import LATP_WrappedEngine
LATP_WrappedEngine wraps a base model (e.g. OpenAI, Claude, local LLM) and coordinates 4 modules:

AirlockModule – context compression / crystal creation (Module A)

LateralShiftEngine – cognitive sorbet, lateral shifts (Module B)

Watchdog / Scorer – context health & toxicity scoring (Module C)

RalModule – numeric drift guard (Module D)

On top of that, LATPManager implements a small state machine for longer sessions.

2. Core data structures
2.1. Message format
Internally, LATP works with a simple message format:

python
Копировать код
Message = dict[str, Any]
# {
#   "role": "system" | "user" | "assistant",
#   "content": str,
#   "tokens": int (optional)
# }
The wrapped model must support:

python
Копировать код
class BaseModelProtocol:
    def generate(self, history: list[Message]) -> str:
        ...
2.2. Crystal
A Crystal is a compressed representation of a session:

python
Копировать код
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Crystal:
    core_theses: List[str]
    librarium_refs: List[str]
    entropy_hash: str
    timestamp: datetime
Crystals are stored either in:

SQLiteLibrarium (simple SQLite-based store), or

VectorLibrarium (in-memory vector store for isomorph search).

3. LATP_WrappedEngine
LATP_WrappedEngine is the main public class:

python
Копировать код
class LATP_WrappedEngine:
    def __init__(
        self,
        base_model: Any,
        librarium: Any,
        ral: Any | None = None,
        vector_librarium: Any | None = None,
    ) -> None:
        self.model = base_model
        self.librarium = librarium
        self.scorer = ContextPoisoningScorer()
        self.airlock = AirlockModule(librarium_client=librarium, ral=ral)
        self.sorbet = LateralShiftEngine(
            librarium_client=librarium,
            vector_librarium=vector_librarium,
        )
        self.ral = ral
High-level generation flow:

Airlock prepares a sanitized context:

trims long history,

optionally compresses it into a Crystal,

injects Librarium references and RalModule digital crystal.

Scorer computes context health:

toxicity, diagnosis, basic loop / saturation heuristics.

LateralShiftEngine may inject a lateral bridge if:

toxicity is in a mid range (not critical, but worth refreshing),

a structurally similar topic is found in VectorLibrarium.

Base model is called with the prepared history.

RalModule (if present) inspects the raw answer for numeric drift:

if numbers contradict previously fixed values, it asks a wedge-question,

engine re-calls the model with this wedge to force recalculation.

4. Modules
4.1. AirlockModule (Module A — context hygiene)
Responsibilities:

Limit context window usage.

Detect when history becomes too long or noisy.

Compress old segments into a Crystal and store it in Librarium.

Rebuild a minimal working context:

System prompt

Last user question

Relevant crystal summary

Optional RalModule digital crystal snippet

Result: the model always sees a fresh, short context + distilled memory instead of a giant chat log.

4.2. LateralShiftEngine (Module B — cognitive sorbet)
Responsibilities:

Detect prolonged, monotonous sessions (high repetition, low diversity).

Query VectorLibrarium for isomorphic topics:

structurally similar, semantically different.

Generate a bridge prompt, e.g.:

"We just explored memory in stone and khachkars.
Now compare it with how DNA stores code without an author."

Inject this bridge into the conversation when toxicity / fatigue is in the mid range.

Result: the model is nudged into a new angle, which resets attention patterns and breaks loops.

4.3. ContextPoisoningScorer (Module C — watchdog)
Responsibilities:

Compute a lightweight context health score:

repetition, length, rough sentiment / toxicity heuristics,

diagnosis labels like "NORMAL", "WARNING", "CRITICAL".

Used both by:

LATP_WrappedEngine (local decisions),

LatpScorerValidator inside orchestrator.py.

It is intentionally simple and deterministic: no extra LLM calls.

4.4. RalModule (Module D — numeric drift guard)
Responsibilities:

Track numeric facts emitted by the assistant:

prices, years, percentages, simple calculations.

Store them in a digital crystal (internal numeric map).

On every new answer draft:

parse numbers,

compare with existing crystal values,

if drift exceeds tolerance → return a wedge-question:

"You wrote 1400, but previously we fixed 1554.
Step back and recompute carefully."

Airlock injects ral.digital_crystal_prompt() into system prompt
after context resets.

Result: numeric consistency survives long dialogues and context resets.

5. LATPManager and state machine
For longer-lived sessions, symbion.latp_manager.LATPManager wraps LATP into a small finite-state machine.

5.1. States
python
Копировать код
class LATPState(str, Enum):
    NORMAL = "NORMAL"
    HEAT_UP = "HEAT_UP"
    LATERAL_SHIFT = "LATERAL_SHIFT"
    COOL_DOWN = "COOL_DOWN"
    AIRLOCK = "AIRLOCK"
    STABLE = "STABLE"
The manager keeps per-session:

current state,

basic metrics (turns, tokens, last toxicity).

5.2. Decisions
python
Копировать код
class LATPAction(str, Enum):
    CONTINUE = "CONTINUE"
    SHIFT = "SHIFT"
    COOL = "COOL"
    AIRLOCK = "AIRLOCK"
Transition examples:

NORMAL → HEAT_UP when toxicity and length grow.

HEAT_UP → LATERAL_SHIFT when mid-range toxicity suggests sorbet.

HEAT_UP → AIRLOCK on critical overload.

LATERAL_SHIFT → COOL_DOWN → STABLE when shift succeeded.

Orchestrator can query:

python
Копировать код
decision = latp_manager.suggest_action(session_id, metrics)
# metrics can include toxicity, diagnosis, tokens, etc.
and then decide how aggressively to intervene.

6. Orchestrator integration
symbion.orchestrator demonstrates how LATP can be used in a multi-engine setup:

Each engine (primary, fallback, REM-mode) is a LATP_WrappedEngine.

LatpScorerValidator uses ContextPoisoningScorer to:

validate draft answers,

decide whether to fall back to another engine.

For full system integration (Symbion.OS / Symbion Relay):

LATP_WrappedEngine is attached to each LLM engine.

LATPManager is attached to the session layer.

Librarium / VectorLibrarium provide persistent memory for crystals.

7. Future work
Planned / suggested extensions:

Persistent LATPManager state

store SessionState in SQLite / Redis.

Async interfaces

AsyncLATPManager for high-throughput WebSocket chats.

Prometheus metrics

latp_state_changes_total

latp_action_suggestions_total

Better isomorph search

plug real vector DB (FAISS, Chroma, Qdrant).

Language-aware wedges

automatically generate RalModule / Dissonance probes
in the user’s language.

For more narrative explanation, see:

README.md – quick start and examples

RFCs in symbion/rfc/ – protocol-level documents (LATP & SYMBION RELAY)

go
Копировать код
