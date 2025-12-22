
\# LATP v1.1 — LUYS-ANTI-TOXIN PROTOCOL



LATP (Luys Anti-Toxin Protocol) is a \*\*cooling and context-rotation layer\*\* for LLMs.  

It sits between the user and the model and prevents \*\*context poisoning\*\* in long-running sessions.



This repo contains the \*\*core LATP v1.1 implementation\*\* plus the RFC spec.



---



\## What is “Context Poisoning”?



In long chats, especially when the context window is heavily used (>60%), LLMs tend to degrade:



1\. \*\*Semantic Saturation\*\*  

&nbsp;  Repetitive phrasing, reduced lexical diversity, answers become smooth but shallow.



2\. \*\*Compression Hallucination\*\*  

&nbsp;  Facts and structure get “flattened”, chapters are mixed, complex ideas are oversimplified.



3\. \*\*Anchor Drift\*\*  

&nbsp;  Recent messages override the System Prompt and base instructions. External knowledge is ignored.



4\. \*\*Resonant Collapse\*\* \*(new in v1.1)\*  

&nbsp;  The model generates answers that sound great to itself — symmetrical, elegant, but false.  

&nbsp;  This is self-hypnosis of patterns, not just random hallucination.



LATP is designed to detect and mitigate all four.


## Architecture overview

LATP (LUYS Anti-Toxin Protocol) is a **stateful proxy** between the user and a base LLM model.

It does not replace the model; it **controls context, cooling and numeric reality** around it.

Core components:

1. **ContextPoisoningScorer** – estimates toxicity / drift based on history.
2. **AirlockModule** – compresses long sessions into Crystals and resets the context.
3. **LateralShiftEngine** – injects lateral prompts from Librarium / VectorLibrarium.
4. **RalModule** – guards numeric reality (digits, dates, simple arithmetic).
5. **LATPManager** – state machine: decides when to continue, shift, cool down or airlock.

---

### LATP state machine (high-level)

```mermaid
stateDiagram-v2
    [*] --> NORMAL

    NORMAL --> HEAT_UP: entropy ↑ / length ↑
    NORMAL --> DIGITAL_DRIFT: RalModule mismatch
    NORMAL --> STABLE: metrics OK

    HEAT_UP --> LATERAL_SHIFT: prophylactic shift
    HEAT_UP --> COOL_DOWN: summarise & compress
    HEAT_UP --> AIRLOCK: critical overload

    LATERAL_SHIFT --> COOL_DOWN: shift completed
    DIGITAL_DRIFT --> REALITY_CHECK: wedge question
    REALITY_CHECK --> NORMAL: numbers corrected

    COOL_DOWN --> STABLE
    STABLE --> NORMAL

    AIRLOCK --> NORMAL: start from Crystal

---



\## Core Modules (A–D)



LATP v1.1 consists of four modules:



\### A. Airlock Module



> “One chapter — one chat.”



\- Distills a long, messy history into a \*\*Crystal\*\* (compressed semantic state).

\- Produces a \*\*clean context\*\* for the model:

&nbsp; - System Prompt

&nbsp; - `\[LATP Crystal]` system tag

&nbsp; - Last user question



\### B. Lateral Shift Engine



> Cognitive sorbet.



\- Detects looping / stagnation.

\- Uses Librarium to find \*\*isomorphic topics\*\* (same structure, different domain).

\- Injects a \*\*bridge question\*\* to wake the model up and shift perspective.



\### C. Watchdog Module



> External conscience.



\- Checks every answer for:

&nbsp; - \*\*Sultan Index\*\* (fluff / clichés / moralizing),

&nbsp; - \*\*Fidelity to Crystal\*\* (is the answer grounded in core theses?),

&nbsp; - \*\*Resonance\*\* (self-similarity → self-hypnosis risk).

\- Can block, warn, or allow answers.



\### D. Dissonance Probe \*(v1.1)\*



> Breaks self-hypnosis.



\- Forces the model into a \*\*critic role\*\* against its own answer.

\- Asks it to:

&nbsp; 1. Find the most questionable claim.

&nbsp; 2. Produce a \*\*wedge question\*\* that could falsify it.

\- Used when resonant collapse is detected.



---



\## Metrics



LATP exposes a monitoring layer (conceptually `LATP\_Monitor`) to track system health:



\- `session\_duration` — how many turns before Airlock has to reset.

\- `toxicity\_peaks` — maximum toxicity levels across sessions.

\- `crystal\_hits` — how often Crystal + Airlock saved a dialogue.

\- `lateral\_success` — how often lateral shifts actually helped.

\- `watchdog\_blocks` — how many answers were blocked.



From these, a \*\*health score\*\* in `\[0.0, 1.0]` can be computed.  

Target for production scenarios: `health\_score > 0.85` for long sessions (1000+ turns).



---



\## Installation (dev mode)



```bash

pip install -e .

Requires Python 3.10+.



Quick Start

Minimal example using the built-in fake model and fake Librarium:



python

Копировать код

from symbion.latp\_core import LATP\_WrappedEngine, FakeModel, FakeLibrarium



engine = LATP\_WrappedEngine(FakeModel(), FakeLibrarium())



history = \[

&nbsp;   {"role": "user", "content": "2+2?", "tokens": 4},

]



answer = engine.generate(history)

print(answer)

In a real system you would:



replace FakeModel with an adapter for your LLM (OpenAI, Anthropic, local, etc.),



replace FakeLibrarium with a real knowledge store (DB / vector DB).



Code Structure

symbion/latp\_core.py

Core LATP v1.1 implementation:



Crystal



ContextPoisoningScorer



AirlockModule



LateralShiftEngine



WatchdogModule



DissonanceProbe



LATP\_WrappedEngine



FakeModel / FakeLibrarium (for tests and examples)



symbion/rfc/RFC-0001-LATP-v1.1.md

Full RFC spec: diagnosis, modules A–D, metrics, and philosophy.



tests/test\_latp\_v11.py

Basic pytest smoke tests.



Philosophy

LATP is built on a few simple but strict beliefs:



The model is a reactor, not a friend.

It must be cooled, monitored, and occasionally rotated — not trusted blindly.



Context is not memory.

Raw chat history is expendable.

Memory lives in Crystal and Librarium under OS control.



Rotation and dissonance are normal.



Context resets (Airlock),



lateral shifts,



watchdog blocks,



REM-like “hypothesis mode”,



dissonance probes

— all are first-class citizens, not rare exceptions.



Endless dialogue is only possible with strict hygiene.

The model may forget you as a “chat partner”,

but it must not forget the truth injected via Librarium and Crystal before every answer.



## RFCs

- [`RFC-0001-LATP-v1.1.md`](symbion/rfc/RFC-0001-LATP-v1.1.md) — LUYS-ANTI-TOXIN PROTOCOL (context hygiene).
- [`RFC-0002-SYMBION-RELAY-v1.0.md`](symbion/rfc/RFC-0002-SYMBION-RELAY-v1.0.md) — multi-engine relay / guard rotation.

## LATP Locks

For presentation / analysis-only sessions you can import:

```python
from symbion.locks import LATP_PRESENTATION_LOCK

print(LATP_PRESENTATION_LOCK)
# "[LATP-LOCK: PRESENTATION MODE] Пользователь — архитектор. ..."
This lock is a protocol hint for higher-level runtimes:
do not edit or “improve” the artefact, only analyse and resonate with it.

python
Копировать код

### 1.2. Блок про SQLiteLibrarium

```markdown
## Librarium (SQLite, minimal)

A minimal persistent Librarium is provided:

```python
from symbion.librarium import SQLiteLibrarium
from symbion.latp_core import LATP_WrappedEngine, FakeModel

librarium = SQLiteLibrarium("librarium.db")
engine = LATP_WrappedEngine(FakeModel(), librarium=librarium)

history = [{"role": "user", "content": "2+2?"}]
answer = engine.generate(history)
print(answer)
This makes LATP executable with a real storage backend, not just in-memory mocks.

## Numeric Drift Guard (RalModule)

LATP can optionally track numeric drift in long-running sessions via `RalModule`.

```python
from symbion.latp_core import LATP_WrappedEngine, FakeModel, FakeLibrarium
from symbion.ral_module import RalModule

engine = LATP_WrappedEngine(
    base_model=FakeModel(),
    librarium=FakeLibrarium(),
    ral=RalModule(),  # numeric drift watchdog
)

history = [
    {"role": "user", "content": "Сколько будет 37 * 42?"},
]

answer = engine.generate(history)
print(answer)
## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
## Relation to other Symbion modules

LATP is not “just a dialog manager”.  
It is the **trajectory engine** that decides *how* an interaction evolves over time and *when* to hand things off to other Symbion subsystems.

In the Symbion Space:

- **Distillation Core (`symbion-distillation-core`)**  
  The structure still.  
  Takes any raw text (even trash), distills structural Essence, burns rhetoric and noise.

- **Librarium**  
  The structural memory plane (the “digital khachkar” fabric).  
  Stores only distilled structures (Essence), not raw texts.

- **Resonance Fabric (`symbion-resonance-fabric`)**  
  The scoring layer.  
  Given an Episode, computes Resonance R and decides:
  - is this Librarium-worthy?
  - is this a TVP candidate?

- **LATP (`symbion-latp`)**  
  Sits **between the user and the base model**, and controls:
  - when to **heat up** (explore, generate hypotheses),
  - when to perform **lateral shifts** (frame changes, Librarium-based),
  - when to **cool down** (summarize, consolidate),
  - when to **airlock** an episode:
    - send it to Distillation Core → Essence → Librarium,
    - send it to Resonance Fabric → R, Librarium / TVP gating.

In short:

- Distillation Core: *what structural content survives the fire?*
- Resonance Fabric: *how strongly does this episode align with the canon?*
- Librarium: *the fabric of all surviving structures.*
- LATP: *the protocol that decides when and how an episode enters this pipeline.*


See: docs/ARCH-LATP.md
