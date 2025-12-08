
# RFC-0002: SYMBION-RELAY Protocol v1.0



\*\*Status:\*\* Core  

\*\*Version:\*\* SYMBION-RELAY/1.0  

\*\*Author:\*\* Symbion OS  

\*\*Depends on:\*\*  

\- RFC-0001: LATP v1.1 (LUYS-ANTI-TOXIN PROTOCOL)  

\- Symbion Librarium (concept)



---



\## 1. Purpose



LATP (RFC-0001) solves \*\*context poisoning\*\* inside a single model.



SYMBION-RELAY addresses a different but related problem:



> No single model should be trusted to handle all tasks, all the time, forever.



The \*\*Relay Protocol\*\* defines how Symbion OS:



\- orchestrates \*\*multiple LLM engines\*\* (cloud + local),

\- performs \*\*guard rotation\*\* (“смена караула”),

\- decides \*\*which model speaks now\*\*,

\- preserves a \*\*single consistent voice\*\* for the user.



This RFC describes the protocol, roles, and routing rules. Concrete implementations (Python classes) are out of scope here but can follow this spec.



---



\## 2. Design Goals and Non-Goals



\### 2.1. Goals



1\. \*\*Model-Agnostic Orchestration\*\*  

&nbsp;  Work with any LLM backend (OpenAI, Anthropic, local models, etc.) via a common interface.



2\. \*\*Guard Rotation\*\*  

&nbsp;  Allow seamless switching between models when:

&nbsp;  - quality drops,

&nbsp;  - hallucinations are detected,

&nbsp;  - cost/latency constraints change,

&nbsp;  - specialization is needed.



3\. \*\*Single Voice, Many Brains\*\*  

&nbsp;  The user should experience \*\*one assistant\*\*, even if multiple engines participate behind the scenes.



4\. \*\*Separation of Concerns\*\*  

&nbsp;  - LATP controls \*\*context hygiene\*\*.

&nbsp;  - SYMBION-RELAY controls \*\*model selection / rotation\*\*.

&nbsp;  - Librarium controls \*\*knowledge\*\*.



5\. \*\*Observability\*\*  

&nbsp;  Capture metrics about model performance and routing decisions.



\### 2.2. Non-Goals



\- Training, fine-tuning, or modifying model weights.

\- Implementing a multi-agent “chat” UI (this protocol is backend-only).

\- Defining the concrete network/API transport (HTTP, gRPC, local calls are allowed).



---



\## 3. Terminology



\- \*\*Engine / Model Engine\*\*  

&nbsp; A wrapper around a concrete LLM (e.g. GPT-4.1, Claude 3.5, local Llama).



\- \*\*Primary Engine\*\*  

&nbsp; The \*\*default\*\* model used for generation at the start of a conversation or segment.



\- \*\*Backup Engine\*\*  

&nbsp; A model that can take over when primary fails or underperforms.



\- \*\*Validator Engine\*\*  

&nbsp; A model used \*\*only\*\* to evaluate or critique answers, not to speak directly to the user.



\- \*\*Archivist Engine\*\*  

&nbsp; A small model used to build/update Crystals and other long-term summaries.



\- \*\*Turn\*\*  

&nbsp; A single user → system → assistant interaction step.



\- \*\*LATP\*\*  

&nbsp; Luys Anti-Toxin Protocol: context cleaning, toxicity scanning, REM mode, etc.



\- \*\*Relay\*\*  

&nbsp; The process of \*\*passing control\*\* from one engine to another.



---



\## 4. Protocol Identifier and Versioning



This spec defines:



> \*\*SYMBION-RELAY/1.0\*\*



Key stability rules:



\- `SYMBION-RELAY/1.x` must:

&nbsp; - keep the core interfaces stable,

&nbsp; - allow internal policy changes and tuning,

&nbsp; - remain compatible with LATP/1.1.



Breaking interface changes (e.g., new required fields) MUST bump the major version (`2.0`, etc).



---



\## 5. Engine Interface and Roles



\### 5.1. Engine Interface



Every model engine implementing SYMBION-RELAY MUST satisfy this abstract interface:



```python

class EngineCapabilities(TypedDict, total=False):

&nbsp;   domains: list\[str]          # e.g. \["code", "math", "law"]

&nbsp;   max\_context\_tokens: int

&nbsp;   cost\_level: str             # "low" | "medium" | "high"

&nbsp;   latency\_level: str          # "fast" | "normal" | "slow"

&nbsp;   supports\_tools: bool

&nbsp;   supports\_vision: bool

&nbsp;   supports\_long\_context: bool





class BaseEngine(Protocol):

&nbsp;   name: str



&nbsp;   def capabilities(self) -> EngineCapabilities:

&nbsp;       ...



&nbsp;   def generate(

&nbsp;       self,

&nbsp;       history: list\[dict],

&nbsp;       \*,

&nbsp;       librarium: dict | None = None,

&nbsp;       system\_overrides: dict | None = None,

&nbsp;   ) -> str:

&nbsp;       ...

5.2. Engine Roles

Each engine in the pool has a role and optional priority:



python

Копировать код

@dataclass

class EngineSpec:

&nbsp;   name: str

&nbsp;   engine: BaseEngine

&nbsp;   role: str          # "primary" | "backup" | "validator" | "archivist"

&nbsp;   priority: int = 100

&nbsp;   tags: list\[str] = None   # e.g. \["code", "reasoning", "cheap"]

Roles:



primary — default engine for generation.



backup — eligible to take over when routing policy decides.



validator — used to score answers, never speaks directly.



archivist — used for crystallization / summarization (complementary to LATP).



6\. Routing Policy

The Routing Policy decides:



whether the current primary engine should generate an answer;



whether the answer is acceptable;



whether a swap is required;



which backup engine to pick.



6.1. Validation Result

All validation (by validators and/or local scorers) MUST be normalized into:



python

Копировать код

@dataclass

class ValidationResult:

&nbsp;   score: float                 # 0.0–1.0, higher is better

&nbsp;   reason: str                  # human-readable diagnostic

&nbsp;   is\_hallucination: bool

&nbsp;   is\_resonant\_collapse: bool

&nbsp;   requires\_swap: bool          # direct signal from validator (optional)

Validators may use:



Sultan Index (fluff),



LATP’s ContextPoisoningScorer,



Librarium-grounded checks,



peer-model comparison (A vs B).



6.2. RoutingPolicy Interface

python

Копировать код

class RoutingPolicy(Protocol):

&nbsp;   def should\_swap(

&nbsp;       self,

&nbsp;       engine: EngineSpec,

&nbsp;       validation: ValidationResult,

&nbsp;       ctx: dict,

&nbsp;   ) -> bool:

&nbsp;       ...



&nbsp;   def choose\_backup(

&nbsp;       self,

&nbsp;       reason: str,

&nbsp;       ctx: dict,

&nbsp;   ) -> EngineSpec | None:

&nbsp;       ...

Minimal recommended logic:



Swap if:



validation.score < quality\_threshold, OR



validation.is\_hallucination, OR



validation.is\_resonant\_collapse, OR



validation.requires\_swap is True.



Choose the first backup engine that:



matches the domain (if ctx\["domain"] is set),



matches any required capabilities (e.g. supports\_tools),



has the lowest priority value.



7\. Turn Lifecycle (Single Request)

SYMBION-RELAY v1.0 defines the following turn-level state machine:



Input Assembly



User message is appended to history.



LATP (RFC-0001) may:



clean or compress history,



inject Crystal tags,



apply Lateral Shifts.



Draft Generation (Primary Engine)



active\_engine.generate(history, librarium=...) is called.



This produces draft\_answer.



Validation



One or more validators produce a ValidationResult for draft\_answer.



LATP’s Watchdog and/or Dissonance Probe may also contribute signals.



Routing Decision



RoutingPolicy.should\_swap(active\_engine, validation, ctx) is evaluated.



If no swap:



answer is returned to the user.



If swap needed:



backup\_engine = RoutingPolicy.choose\_backup(...) is called.



Relay / Swap



If backup\_engine is available:



LATP may re-clean context (Airlock) for the new engine.



backup\_engine.generate(clean\_history, librarium=...) is called.



The second answer becomes the final answer.



If no backup is available:



A failsafe answer is generated (e.g. “I cannot answer reliably”).



Post-Turn Metrics



All decisions (validation score, chosen engine, swap reasons) are logged.



Per-engine health metrics are updated.



8\. Interactions with LATP (RFC-0001)

SYMBION-RELAY is designed to compose with LATP, not replace it.



Recommended layering:



LATP first



history\_clean = LATP.sanitize\_context(history)



Toxicity, resonance, and REM mode are handled here.



Relay second



Use history\_clean as input to ModelOrchestrator.



Unified Validation



LATP’s Watchdog + external validators both contribute to a unified ValidationResult.



Shared Context

The routing policy may use LATP signals:



toxicity level,



presence of REM mode,



number of consecutive blocked answers.



Example: if a model repeatedly triggers LATP’s CRITICAL state → lower its routing priority or temporarily disable it.



9\. ModelOrchestrator Interface

A Model Orchestrator is the main entrypoint for Relay logic.



python

Копировать код

@dataclass

class OrchestratorContext:

&nbsp;   domain: str | None = None       # e.g. "code", "law", "math"

&nbsp;   max\_latency\_ms: int | None = None

&nbsp;   max\_cost\_level: str | None = None   # e.g. "medium"

&nbsp;   user\_id: str | None = None





class ModelOrchestrator(Protocol):

&nbsp;   def get\_answer(

&nbsp;       self,

&nbsp;       user\_query: str,

&nbsp;       history: list\[dict],

&nbsp;       \*,

&nbsp;       ctx: OrchestratorContext | None = None,

&nbsp;   ) -> str:

&nbsp;       ...

Typical internal flow:



Build full\_history = history + \[user\_query].



Call LATP: latp\_history = LATP.sanitize\_context(full\_history).



Call primary\_engine.generate(latp\_history, librarium=...).



Validate → possibly swap → return final answer.



10\. Two-Speed Principle

SYMBION-RELAY inherits the Two-Speed Principle from LATP:



Truth Speed — strict, validated mode (Watchdog + validators active).



Resonance Speed — periodic “dream mode” where the model can explore hypotheses.



Relay-aware behavior:



The orchestrator MAY designate certain turns as REM turns:



less validation,



more experimentation,



but responses MUST be clearly marked (e.g. \[REM Hypothesis]).



Relay MUST ensure that REM mode is not used:



for decisive actions,



for irreversible suggestions,



for security-sensitive outputs.



Example policy:



Every 10th turn, allow REM mode on a specialized “imagination engine” (e.g. a creative model), then feed its output back through a more strict engine in the next turn.



11\. Failure Modes and Fallbacks

SYMBION-RELAY MUST handle at least these failure modes:



Engine Timeout / Transport Error



Mark the engine as temporarily unhealthy.



Log the error.



Try a backup engine if available.



Validation Catastrophic Failure



If validators disagree or fail:



fallback to conservative messaging:

“I am not confident enough to answer this reliably.”



Infinite Swap Loop



Prevent repeated swapping back and forth.



Each turn MUST perform at most one swap attempt.



Cost/Quota Limits



If an engine exceeds its cost or quota:



lower its priority,



or temporarily disable it for the routing policy.



12\. Metrics and Monitoring (Relay Monitor)

A Relay Monitor SHOULD track at least:



turns\_total



turns\_with\_swap



swap\_reasons (histogram: hallucination, low\_score, timeout, etc.)



engine\_usage\[engine\_name] — number of turns answered by each engine.



engine\_win\_rate\[engine\_name] — how often the engine’s draft answer was accepted vs rejected.



validation\_scores — distribution of ValidationResult.score.



latp\_toxicity\_correlation — correlation between LATP toxicity and swaps.



These metrics allow:



automatic tuning of RoutingPolicy,



identifying underperforming models,



business-level cost/performance analysis.



13\. Security and Safety Considerations

Validator engines MUST NOT be given more user data than necessary.



Sensitive content (PII, secrets) SHOULD be minimized or masked when sent to external engines.



REM / Resonance modes MUST always be tagged and never be used for:



instructions with side effects,



legal / medical / financial decisions,



system-level operations.



14\. Summary

SYMBION-RELAY/1.0 defines how Symbion OS:



treats model engines as replaceable guards, not monolithic gods;



routes each user turn to the most appropriate engine;



flips to a backup when the current guard starts hallucinating or collapsing;



remains compatible with LATP v1.1, which protects the context;



exposes clear interfaces and metrics for extension and tuning.



One voice, many brains.

Guard rotation is not a hack — it is the core runtime architecture.


