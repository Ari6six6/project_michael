# The Protocol: Stateful Kantian Machine

This document describes the architecture and flow of Michael's stateful machine — the three-turn Kantian loop that governs every `michael run` invocation.

## The Three-Turn Flow

### Turn 1: Scripture Interpretation (Read-Only)

```
User invokes: michael run
    ↓
Load Scripture (genesis.md + protocol.md + other files)
    ↓
Restrict toolset to: read_file, list_dir, search_memory
    ↓
Send to LLM:
"You are about to receive a task. First, read and interpret this scripture. 
What is your understanding of the philosophy and constraints?"
    ↓
LLM examines scripture, calls read_file/list_dir as needed
    ↓
LLM responds with interpretation
    ↓
Log event: scripture.interpreted
    ↓
[In normal mode: show to user]
[In god mode: suppress output]
```

**Purpose**: Anchor the LLM in the foundational principles before engaging the specific task.

---

### Turn 2: Task Reception & Target Formulation

```
Read user's actual prompt from stdin
    ↓
Send to LLM:
"Given your understanding of the scripture, here is the task: [user_prompt]

Formulate and state clearly:
1. TARGET: What does success look like for this task?
2. GOAL: What is the immediate objective?
3. CONSTRAINTS: What you know (filesystem state, tools), what you should do (user intent), and what you can hope for (achievable outcomes)."
    ↓
LLM formulates target/goal/constraints
    ↓
Log event: target.formulated
    ↓
[In normal mode: show to user]
[In god mode: suppress output]
```

**Purpose**: Create explicit, shared understanding of the problem scope before tool invocation.

---

### Turn 3+: Kantian Iteration Loop (Full Toolset)

```
iteration_count = 0

LOOP:
  iteration_count += 1
  
  [If iteration_count == max_kantian_iterations (default 5)]:
    Inject nudge: "You have reached the iteration limit (5). 
    Make your final assessment: Are you ready to proceed (Ja), 
    or do blockers remain? Signal your decision."
  
  Send to LLM:
  "Given the target and goal above, iterate through the three questions:
  
  1. WHAT CAN I KNOW?
     - What does the current filesystem reveal about the problem?
     - What tools do you have available?
     - What are the constraints (sandbox limits, timeouts, network)?
     - What is the current state?
  
  2. WHAT SHOULD I DO?
     - What is the user's intent (explicit and implicit)?
     - What follows from the inherent logic of the problem?
     - What is the smallest, most correct action forward?
     - Does this align with the protocol?
  
  3. WHAT CAN I HOPE FOR?
     - Is the target achievable with available tools and time?
     - What is the success criterion?
     - What might go wrong? Can you verify correctness?
     - Is this change reversible?
  
  You may call any tool at any point. Iterate until you are confident, 
  then decide: Ja (proceed) or continue iterating."
  
  ↓
  LLM answers the three questions, calls tools as needed
  ↓
  Log event: kantian.iteration (with iteration_num, question, answer)
  ↓
  [If LLM signals "Ja"]: break LOOP
  [Else]: continue LOOP
```

**Purpose**: Enforce explicit reasoning through epistemic, ethical, and teleological dimensions. The LLM gains autonomy to iterate until it achieves internal certainty.

---

### Turn 4: Execution Phase

```
[Loop ended on "Ja"]
    ↓
Present staged changes to user:
"The agent proposes the following changes:
[show file diffs, tool calls executed]

Accept? (Y/n)"
    ↓
[In normal mode]: wait for user input
[In god mode]: auto-approve (Ja means auto-commit)
    ↓
[If approved]: 
  - Commit staged changes to real filesystem
  - Save trash snapshot for undo
  - Log event: tool.executed (for each change)
  ↓
[If rejected]:
  - Discard staged changes
  - Log event: tool.rejected
  - Return to user
```

**Purpose**: User retains the final gate (except in god mode, where the system has full authority once LLM signals Ja).

---

## Event Log Architecture

Every turn is logged with full context:

```
{
  "seq": 1,
  "ts": "2026-05-08T14:45:00.123456+00:00",
  "type": "michael.run.started",
  "payload": {"mode": "kantian", "user_prompt": "create a hello.py"}
}

{
  "seq": 2,
  "type": "scripture.loaded",
  "payload": {"scripture_length_chars": 5000, "file_count": 2}
}

{
  "seq": 3,
  "type": "scripture.interpreted",
  "payload": {"interpretation": "...full LLM response..."}
}

{
  "seq": 4,
  "type": "target.formulated",
  "payload": {"target": "...", "goal": "...", "constraints": "..."}
}

{
  "seq": 5,
  "type": "kantian.iteration",
  "payload": {"iteration_num": 1, "question": "know", "answer": "...", "tools_called": ["read_file"]}
}

{
  "seq": 6,
  "type": "kantian.iteration",
  "payload": {"iteration_num": 2, "question": "should", "answer": "...", "tools_called": ["write_file"]}
}

{
  "seq": 7,
  "type": "kantian.iteration",
  "payload": {"iteration_num": 3, "question": "hope", "answer": "...", "tools_called": []}
}

{
  "seq": 8,
  "type": "assistant.ja",
  "payload": {"iteration_count": 3, "final_message": "Ja"}
}

{
  "seq": 9,
  "type": "tool.executed",
  "payload": {"path": "hello.py", "action": "write_file", ...}
}
```

---

## Backward Compatibility

The flag `--legacy` disables the Kantian machine and restores the original stateless loop:

```bash
michael run                    # Uses stateful Kantian machine (default)
michael run --legacy           # Uses old stateless loop
michael ask "prompt" --legacy  # One-shot without scripture/three-turn
```

---

## God Mode Behavior

`michael nitro --god` engages the heavy model with full authority:

1. **Turn 1 & 2 are silent** — no output shown to user
2. **Turn 3+ iterations are silent** — no output shown to user
3. **On Ja**: changes are **auto-committed** without user confirmation
4. **Result**: User sees only the final state and the changelog

This is appropriate for large-scale refactoring where you trust the model's read of the full codebase.

---

## Configuration

```json
{
  "use_stateful_kantian": true,
  "max_kantian_iterations": 5,
  "kantian_visible": false,
  "scripture_dir": "scripture"
}
```

- `use_stateful_kantian` — enable/disable the three-turn loop globally
- `max_kantian_iterations` — maximum iterations before nudge-to-Ja
- `kantian_visible` — show Turn 1/2 output even in god mode
- `scripture_dir` — path to scripture files (relative to repo root)

---

## Invariants

1. **Scripture is read-only** — Turn 1 cannot mutate the codebase
2. **Iteration counter increases monotonically** — no infinite loops
3. **Ja is the only exit** — the LLM must explicitly signal completion
4. **Full context on every turn** — the LLM sees full history from the event log
5. **Staging before execution** — all mutations are staged and verified before real commit

---

[Add implementation notes, gotchas, or design rationale here as needed.]

---

**Last updated**: [user to fill]
