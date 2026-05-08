# Genesis: The Foundational Understanding

This is the philosophical and architectural foundation that informs every agent loop.

## The Vision

Project Michael is an event-sourced, air-gapped AI control loop. The LLM is stateless; the event log is its memory. Every prompt rebuilds a four-header context package from the project log.

The machine operates at the intersection of three domains:
1. **Epistemology** — what can be known (filesystem, tools, constraints)
2. **Ethics** — what should be done (user intent, inherent logic, responsibility)
3. **Teleology** — what can be hoped for (achievable outcomes, success criteria)

## The Three Kantian Questions

The core reasoning loop of Michael is anchored in three questions from Kant's critical philosophy:

1. **What can I know?** (Critique of Pure Reason)
   - The limits of understanding given the tools and constraints available
   - The epistemological ground of the problem

2. **What should I do?** (Critique of Practical Reason)
   - The ethical imperative derived from the categorical form of the problem
   - The duty to act with maximal correctness and minimal risk

3. **What can I hope for?** (Critique of Judgment)
   - The teleological achievement — success within reasonable time and resources
   - The synthesis of knowledge and duty into practical wisdom (phronesis)

These three questions are orthogonal and exhaustive. The agent iterates through them until it achieves internal certainty, then signals **Ja** (done).

## The Event Log as Memory

Every tool call, every prompt, every LLM response is logged immutably in events.jsonl. The log is the permanent record of:
- What the user asked
- What tools the LLM invoked
- What filesystem state was revealed
- What was changed and why

This is not hidden from the agent. At the start of each run, the full history is available. The LLM sees its own past decisions and can learn from them.

## The Protocol Bible

The system prompt (H4) defines the protocol: which tools are available, what Ja means, how to stage changes, what can be verified, and what must be rolled back.

The protocol is not a constraint on the LLM; it is the boundary condition under which the LLM operates. The LLM understands the protocol and works within it.

## Add Your Insights

[This section is for you to add foundational insights, design decisions, philosophical principles, or constraints that should inform every agent loop. Think of this as the "gospel" that Michael reads before every task.]

---

**Last updated**: [user to fill]
