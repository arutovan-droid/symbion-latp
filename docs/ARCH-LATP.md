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

