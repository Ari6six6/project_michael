# The Protocol: Stateful Kantian Machine

The machine implements a **closed question-answer loop** grounded in Kant's critical philosophy.

## Fundamental Truth

**Question → Iterate → Ja (Answer) → Execute**

User asks question. LLM answers through iteration. "Ja" is the final answer value. System executes. Done.

---

## The Three-Turn Flow

### Turn 1: Scripture & Tools (Full Toolset)

```
User invokes: michael run [question]
    ↓
Load Scripture (genesis.md + protocol.md + project history)
    ↓
Full toolset available: read_file, list_dir, write_file, apply_patch, 
                       run_in_sandbox, run_shell, search_memory
    ↓
Send to LLM:
"Read and interpret this scripture. What is your understanding of the 
philosophy, constraints, and prior work? You may call any tool to 
examine the codebase."
    ↓
LLM reads scripture + explores filesystem as needed
    ↓
LLM responds with interpretation
```

**Purpose**: Ground the LLM in foundation + current state, with full ability to inspect.

---

### Turn 2: Question Clarification

```
LLM reads user's actual question from stdin
    ↓
Send to LLM:
"Given your understanding, here is the question:

[user_question]

Clarify:
1. What is being asked?
2. What is success?
3. What constraints apply?"
    ↓
LLM clarifies the question
```

**Purpose**: Ensure precise understanding before iteration.

---

### Turn 3+: Kantian Iteration Loop (Includes VPS Sandbox Testing)

```
iteration_num = 0

LOOP:
  iteration_num += 1
  
  Send to LLM (with literal questions spelled out):
  
  ===== THE THREE KANTIAN QUESTIONS =====
  
  1. WHAT CAN I KNOW?
     - What does the filesystem reveal?
     - What tools are available?
     - What constraints exist (time, resources, environment)?
     - What can I test?
  
  2. WHAT SHOULD I DO?
     - What is the user intent?
     - What follows from the logic of the problem?
     - What is the smallest correct step?
     - Is this reversible?
  
  3. WHAT CAN I HOPE FOR?
     - Is the goal achievable?
     - Can I verify it works?
     - What is the success criterion?
     - What might go wrong?
  
  =======================================
  
  Answer all three. Then if uncertain, call tools:
  - read_file / list_dir to inspect code
  - write_file / apply_patch to propose changes
  - run_in_sandbox to test on VPS before committing
  - run_shell for local inspection
  
  After testing, loop back or signal: Ja
```

**VPS Sandbox Loop:**
- When proposing changes, run_in_sandbox tests them on VPS
- Verify before committing
- Feed results back into next iteration

**Purpose**: Iterate through three orthogonal dimensions of reasoning. Test. Refine. Until certain.

---

## Exit Condition: Ja

**"Ja" is not a signal. It is the final answer.**

When the LLM has:
1. Answered the original question
2. Verified the answer works (via sandbox testing)
3. Refined through iteration
4. Achieved certainty

It ends the message with **Ja** (case-sensitive, on its own line or trailing).

This contains the final answer. The system:
- Captures it
- Auto-executes all staged changes
- Returns the answer + artifacts to the user
- Loop closes

---

## No Gate, Always Auto-Execute

- The system is **always in god mode**
- User asks question
- LLM iterates through Kantian loop
- On "Ja": changes auto-commit, no approval needed
- User gets answer + filled project directory

---

## Constraints

- **Max iterations:** 5 (configurable). If reached, nudge LLM: "You have reached the limit. Final answer: Ja or blockers?"
- **Iteration counter:** visible to LLM each turn
- **Sandbox testing:** encouraged on every risky change
- **Scripture:** always loaded, always available for reference

---

## Events

- `michael.run.started` — session begins
- `scripture.loaded` — scripture files read
- `scripture.interpreted` — Turn 1 complete
- `question.clarified` — Turn 2 complete
- `kantian.iteration` — Turn 3.N complete (with question, answer, tools called)
- `sandbox.test.run` — test executed on VPS
- `sandbox.test.passed` / `sandbox.test.failed` — result
- `assistant.ja` — final answer received
- `tool.executed` — changes committed
- `michael.run.ended` — session closes

---

## Example Flow

```
$ michael run

>>> What's a good pattern for async error handling in this codebase?

[Turn 1: Scripture interpretation]
LLM reads genesis, protocol, project history, explores filesystem

[Turn 2: Question clarification]
LLM: "You're asking about async error patterns. I'll examine the codebase,
propose examples, test them, and suggest the best pattern."

[Turn 3.1: Kantian iteration]
1. WHAT CAN I KNOW? [reads error handling code, examines tests]
2. WHAT SHOULD I DO? [proposes three patterns]
3. WHAT CAN I HOPE FOR? [can verify with tests]
→ run_in_sandbox to test pattern A
→ test fails, refine

[Turn 3.2: Kantian iteration]
1. WHAT CAN I KNOW? [test results show issue]
2. WHAT SHOULD I DO? [fix the pattern]
3. WHAT CAN I HOPE FOR? [retest]
→ run_in_sandbox again
→ test passes

Ja

Here is the recommended async error handling pattern for your codebase:
[answer with code examples, artifacts]

[System auto-commits, loop closes]
```

---

**Last updated**: [user to fill]
