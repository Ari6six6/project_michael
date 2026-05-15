# The Protocol: Four-Room Kantian Cycle

The machine implements a **closed question-answer loop** grounded in Kant's critical philosophy.

## Fundamental Truth

**Goal → Cycle → Ja (Room 4) → Commit**

User states a goal. The agent iterates through four rooms per cycle. Room 4 is the gate: "Did we answer the goal?" Ja = done and commit. No = new cycle.

---

## The Four Rooms

Each room has one Kantian question. Rooms 1–3 run for up to 8 turns and advance automatically — they do not require Ja to exit. **Ja is exclusive to Room 4.**

### Room 1 — WHAT CAN I KNOW? (Epistemics)

- **Tools**: `read_file`, `list_dir`, `search_memory` — **read-only, no writes**
- **Purpose**: Map the full state. Filesystem, prior tool history, constraints, open questions.
- **Exit**: Automatic after up to 8 turns. Complete your exploration then end your response.

### Room 2 — WHAT SHOULD I DO? (Ethics)

- **Tools**: Full toolset — `write_file`, `apply_patch`, `run_in_sandbox`, `run_shell`, read tools
- **Purpose**: Implement the smallest correct action. Test before finishing.
- **Tool invention**: If a needed capability does not exist, write it to `tools/<name>.py`. The file must export:
  - `TOOL_SCHEMA` — OpenAI function schema dict
  - A callable with the same name as the schema
  - Michael loads it as a real tool at the start of the **next cycle**
- **Exit**: Automatic after up to 8 turns. Complete your implementation then end your response.

### Room 3 — WHAT CAN I HOPE FOR? (Teleology)

- **Tools**: `read_file`, `list_dir`, `search_memory`, `write_file`, `apply_patch`
- **Purpose**: Reflect on what was achieved. Document what is still open. Write a note on what the next cycle should address.
- **Exit**: Automatic after up to 8 turns. Complete your plan then end your response.

### Room 4 — IS THE GOAL MET? (Completion Gate)

- **Tools**: `read_file`, `list_dir`, `search_memory` — **read-only**
- **Purpose**: Evaluate the full body of work against the original user goal.
  - If YES with certainty → end response with **Ja** → commit all staged changes → done
  - If NO → state exactly what is still missing → new cycle begins
- **This is the only room where Ja triggers a commit.**

---

## Cycle Flow

```
User goal prompt
      ↓
┌─────────────── CYCLE N ───────────────────────────────────────┐
│  [load any tools/ scripts written in previous cycle]           │
│                                                                │
│  Room 1 — read-only exploration          → Ja                  │
│  Room 2 — full tools + tool invention    → Ja                  │
│  Room 3 — outlook, document next steps   → Ja                  │
│  Room 4 — is goal met?                                         │
│              Ja  → commit + exit                               │
│              No  → inject gap → CYCLE N+1                      │
└───────────────────────────────────────────────────────────────┘
Max cycles: 9 · Max turns per room (1–3): 8 · Room 4: always 1 turn
```

---

## Message History

The full conversation carries forward across all rooms and all cycles. Room 2 sees Room 1's exploration. Room 4 sees everything. Between cycles, a system message is injected describing what Room 4 found missing — this is the handoff to the next cycle.

---

## Tool Invention Format

To invent a tool in Room 2, write `<project>/tools/<name>.py`:

```python
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_port",
        "description": "Check if a TCP port is open on a host.",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
            },
            "required": ["host", "port"],
        },
    },
}

def check_port(host: str, port: int) -> str:
    import socket
    try:
        with socket.create_connection((host, port), timeout=3):
            return f"open"
    except OSError:
        return f"closed"
```

On the next cycle, `check_port` appears in Room 2's tool list and can be called by name.

---

## Exit Condition: Ja

**"Ja" is not a signal. It is the final answer.**

- Rooms 1–3 exit automatically after their turn limit. Do not use Ja in these rooms.
- In Room 4: Ja ends the entire cycle, triggers commit, exits the agent loop.
- Case-sensitive. Must be the trailing token of the message.
- Rejects: `"ja"`, `"Ja, das ist..."`, empty string.

---

## Events Logged

| Event | When |
|-------|------|
| `cycle.started` | Start of each cycle |
| `room.epistemics.entered` | Room 1 entry |
| `room.epistemics.ja` | Room 1 exit |
| `room.ethics.entered` | Room 2 entry |
| `room.ethics.ja` | Room 2 exit |
| `room.teleology.entered` | Room 3 entry |
| `room.teleology.ja` | Room 3 exit |
| `room.completion.entered` | Room 4 entry |
| `room.completion.ja` | Room 4 exit (goal met) |
| `cycle.incomplete` | Room 4 said no — new cycle |
| `assistant.message` | Each LLM response (with room label) |
| `tool.executed` | Tool call result |
| `agent.ended` | Loop finished (ja: true/false, cycles: N) |

---

**Last updated**: 2026-05-13
