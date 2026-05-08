"""Agent loop: _run_agent_loop and its helpers."""
from __future__ import annotations

import json
from typing import Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.panel import Panel
from rich.syntax import Syntax

import michael.globals as G
from michael.backends import (
    LocalPodmanBackend,
    SandboxBackend,
    _require_endpoint,
    _ssh_preflight,
    llm_client,
    make_backend,
)
from michael.config import Config, ModelProfile
from michael.project import Project, append_event
from michael.tools import (
    PendingChanges,
    TOOLS,
    _resolve_in_project,
    _summary_for,
    commit_pending,
    dispatch_tool_call,
    _format_delta,
)
from michael.utils import (
    build_header,
    load_scripture,
    kantian_turn1_prompt,
    kantian_turn2_prompt,
    kantian_iteration_prompt,
)

_NUDGE_NO_JA = (
    "system reminder: you ended your turn without tool calls and without "
    f"the {G.JA_PASSPHRASE!r} passcode. Either use tools to keep iterating, "
    f"or end your message with `{G.JA_PASSPHRASE}` on its own line to surface "
    "your work to the user. Until then you are talking to Michael, not "
    "the user."
)


def _tools_for_mode(mode: str) -> list[dict[str, Any]]:
    """code/nitro = full toolset; discussion = read-only tools only."""
    if mode == "discussion":
        return [t for t in TOOLS if t["function"]["name"] in G.AUTO_EXEC_TOOLS]
    return TOOLS


def _resolve_nitro_model(cfg: Config, model: Optional[str]) -> tuple[str, ModelProfile]:
    """Pick the heavy model for nitro: explicit --model wins, then 'hacker'/'nitro'/'big'."""
    if model:
        return cfg.get_model(model)
    for candidate in ("hacker", "nitro", "big"):
        if candidate in cfg.models:
            return candidate, cfg.models[candidate]
    raise G.MichaelError(
        "nitro requires a 'hacker', 'nitro', or 'big' model profile in config "
        "(or pass --model NAME explicitly)"
    )


def _resolve_tier(cfg: Config, tier: str) -> tuple[str, ModelProfile, str, bool]:
    """Map a tier flag (coder/instruct/hacker) to (profile_name, profile, mode, god_mode).

    Checks cfg.tier_map for user overrides first, then falls back to TIER_DEFAULTS.
    Hacker has an extended fallback chain for backward compatibility.
    """
    default_profile, mode, god_mode = G.TIER_DEFAULTS[tier]
    profile_name = cfg.tier_map.get(tier, default_profile)
    if tier == "hacker" and profile_name not in cfg.models:
        for candidate in ("hacker", "nitro", "big"):
            if candidate in cfg.models:
                profile_name = candidate
                break
    name, profile = cfg.get_model(profile_name)
    return name, profile, mode, god_mode


def _present_pending_to_user(
    project: Project,
    pending: PendingChanges,
    final_text: str,
) -> bool:
    """Render accumulated pending changes for the user; ask one yes/no. Returns True on apply."""
    if final_text:
        G.console.print(
            Panel(final_text, title="assistant — Ja", border_style="green")
        )
    if not pending.change_log:
        return True

    for i, entry in enumerate(pending.change_log, 1):
        import difflib
        delta = entry["delta"]
        title = (
            f"[{i}/{len(pending.change_log)}] "
            f"{_summary_for(entry['tool'], entry['args'])}  "
            f"+{len(delta['added'])} ~{len(delta['modified'])} "
            f"-{len(delta['removed'])}"
        )
        sections: list[str] = []
        if entry["tool"] == "write_file":
            rel = str(entry["args"].get("path", "?"))
            try:
                real_target = _resolve_in_project(project, rel)
                old = real_target.read_text(errors="replace") if real_target.is_file() else ""
            except G.MichaelError:
                old = ""
            diff = "".join(difflib.unified_diff(
                old.splitlines(keepends=True),
                str(entry["args"].get("content", "")).splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )) or "(no changes)"
            sections.append(diff)
        elif entry["tool"] == "apply_patch":
            sections.append(str(entry["args"].get("unified_diff", "")))
        sections.append(_format_delta(delta))
        if entry.get("verify_rc") is not None:
            sections.append(f"verify rc={entry['verify_rc']}")
        if entry.get("mismatch"):
            sections.append(f"prediction mismatch: {entry['mismatch']}")
        G.console.print(
            Panel(
                Syntax("\n\n".join(sections), "diff", theme="ansi_dark", word_wrap=True),
                title=title, border_style="cyan",
            )
        )

    try:
        choice = (typer.prompt(
            f"Apply all {len(pending.change_log)} pending change(s)? [Y]es / [n]o",
            default="y",
        ) or "").strip().lower()
    except (KeyboardInterrupt, typer.Abort):
        choice = "n"
    return choice in ("", "y", "yes")


def _run_kantian_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    mode: str,
    *,
    verb_label: str,
    god_mode: bool = False,
) -> None:
	"""Kantian machine: closed question-answer loop grounded in critical philosophy.

	Question → Scripture + Tools → Clarify → Iterate (Know/Should/Hope) → Sandbox Test → Ja (Answer) → Auto-Execute

	Always auto-executes on Ja. No user gate. System is always in god mode.
	"""
	endpoint = _require_endpoint(profile, name)
	_ssh_preflight(cfg)

	client = llm_client(endpoint, profile.vllm_api_key)
	backend = make_backend(cfg)
	tools_full = _tools_for_mode(mode)  # Full toolset always available
	base_prompt = cfg.resolved_system_prompt()

	backend_label = (
		"remote-podman (vps)" if cfg.vps_active()
		else ("local-podman" if isinstance(backend, LocalPodmanBackend)
		      else "no-sandbox")
	)
	G.console.print(
		f"[bold cyan]michael {verb_label}[/] [dim]kantian=true  sandbox={backend_label}[/]"
	)
	G.console.print(
		f"[dim]Question → Scripture → Iterate (KNOW / SHOULD / HOPE) → Sandbox Test → Ja[/]"
	)

	session = PromptSession(
		history=FileHistory(str(G.REPL_HISTORY_PATH)),
		auto_suggest=AutoSuggestFromHistory(),
	)

	append_event(
		"agent.started",
		{
			"model": name,
			"served": profile.served_model_name,
			"mode": mode,
			"kantian": True,
			"sandbox": backend_label,
		},
		project=project,
	)

	# Outer loop: handle multiple questions (or exit)
	while True:
		# Read question from user
		try:
			question = session.prompt(">>> ").strip()
		except (EOFError, KeyboardInterrupt):
			break
		if not question or question.lower() in ("quit", "exit"):
			break

		append_event(
			"michael.run.started",
			{"question": question},
			project=project,
		)

		# Load scripture
		scripture = load_scripture()
		if scripture:
			append_event(
				"scripture.loaded",
				{"length": len(scripture), "lines": len(scripture.split("\n"))},
				project=project,
			)

		# TURN 1: Scripture Interpretation (Full Toolset)
		turn1_prompt = kantian_turn1_prompt()
		turn1_msg = f"{scripture}\n\n---\n\n{turn1_prompt}" if scripture else turn1_prompt

		header = build_header(project, base_prompt, mode=mode)
		messages: list[dict[str, Any]] = [
			{"role": "system", "content": header},
			{"role": "user", "content": turn1_msg},
		]

		G.console.print(f"[dim]· Turn 1: interpreting scripture…[/]")
		try:
			resp = client.chat.completions.create(
				model=profile.served_model_name,
				messages=messages,
				tools=tools_full,
				tool_choice="auto",
				stream=False,
				timeout=float(profile.request_timeout_s),
			)
		except Exception as e:
			G.err.print(f"LLM error (Turn 1): {e}")
			append_event("error", {"where": "kantian_turn1", "msg": str(e)}, project=project)
			break

		msg = resp.choices[0].message
		turn1_content = msg.content or ""
		turn1_tool_calls = msg.tool_calls or []

		# Process Turn 1 tool calls
		for tc in turn1_tool_calls:
			tname = tc.function.name
			try:
				targs = json.loads(tc.function.arguments or "{}")
			except json.JSONDecodeError:
				targs = {}
			dispatch_tool_call(tname, targs, project, cfg, backend, PendingChanges())

		if turn1_content:
			append_event(
				"scripture.interpreted",
				{"interpretation": turn1_content if cfg.log_responses else "..."},
				project=project,
			)
			G.console.print(f"[dim]{turn1_content}[/]")

		messages.append({"role": "assistant", "content": turn1_content})

		# TURN 2: Question Clarification
		turn2_prompt = kantian_turn2_prompt(question)
		messages.append({"role": "user", "content": turn2_prompt})

		G.console.print(f"[dim]· Turn 2: clarifying question…[/]")
		try:
			resp = client.chat.completions.create(
				model=profile.served_model_name,
				messages=messages,
				tools=tools_full,
				tool_choice="auto",
				stream=False,
				timeout=float(profile.request_timeout_s),
			)
		except Exception as e:
			G.err.print(f"LLM error (Turn 2): {e}")
			append_event("error", {"where": "kantian_turn2", "msg": str(e)}, project=project)
			break

		msg = resp.choices[0].message
		turn2_content = msg.content or ""

		append_event(
			"question.clarified",
			{"clarification": turn2_content if cfg.log_responses else "..."},
			project=project,
		)
		G.console.print(f"[dim]{turn2_content}[/]")

		messages.append({"role": "assistant", "content": turn2_content})

		# TURN 3+: Kantian Iteration with Sandbox Testing
		pending = PendingChanges()
		iteration_num = 0
		ja_received = False

		try:
			while iteration_num < cfg.max_kantian_iterations:
				iteration_num += 1

				# Spell out the three Kantian questions LITERALLY
				iteration_prompt = (
					f"You are on iteration {iteration_num} of {cfg.max_kantian_iterations}.\n\n"
					f"Answer these three questions explicitly:\n\n"
					f"1. WHAT CAN I KNOW?\n"
					f"   - What does the filesystem reveal?\n"
					f"   - What tools are available?\n"
					f"   - What constraints exist (time, resources, environment)?\n"
					f"   - What can I test?\n\n"
					f"2. WHAT SHOULD I DO?\n"
					f"   - What is the user intent?\n"
					f"   - What follows from the logic of the problem?\n"
					f"   - What is the smallest correct step?\n"
					f"   - Is this reversible?\n\n"
					f"3. WHAT CAN I HOPE FOR?\n"
					f"   - Is the goal achievable?\n"
					f"   - Can I verify it works?\n"
					f"   - What is the success criterion?\n"
					f"   - What might go wrong?\n\n"
					f"After answering, call tools to test your answer in the VPS sandbox. "
					f"When you have verified the answer and are certain, end your message with: Ja"
				)

				messages.append({"role": "user", "content": iteration_prompt})

				G.console.print(f"[dim]· Turn 3.{iteration_num}: KNOW / SHOULD / HOPE…[/]")
				try:
					resp = client.chat.completions.create(
						model=profile.served_model_name,
						messages=messages,
						tools=tools_full,
						tool_choice="auto",
						stream=False,
						timeout=float(profile.request_timeout_s),
					)
				except Exception as e:
					G.err.print(f"LLM error (iteration {iteration_num}): {e}")
					append_event(
						"error",
						{"where": "kantian_iteration", "iteration": iteration_num, "msg": str(e)},
						project=project,
					)
					pending.discard()
					break

				msg = resp.choices[0].message
				iteration_content = msg.content or ""
				tool_calls = msg.tool_calls or []

				# Log iteration
				if iteration_content or tool_calls:
					payload: dict[str, Any] = {
						"iteration_num": iteration_num,
						"tools_called": [tc.function.name for tc in tool_calls],
					}
					if cfg.log_responses and iteration_content:
						payload["answer"] = iteration_content
					append_event("kantian.iteration", payload, project=project)

				# Process tool calls (including sandbox testing)
				if tool_calls:
					for tc in tool_calls:
						tname = tc.function.name
						try:
							targs = json.loads(tc.function.arguments or "{}")
						except json.JSONDecodeError:
							targs = {}

						# Log sandbox tests
						if tname == "run_in_sandbox":
							append_event(
								"sandbox.test.run",
								{"iteration": iteration_num, "code_length": len(targs.get("python_code", ""))},
								project=project,
							)

						result = dispatch_tool_call(
							tname, targs, project, cfg, backend, pending,
						)

						# Log sandbox results
						if tname == "run_in_sandbox":
							if "error" in result.lower() or "traceback" in result.lower():
								append_event(
									"sandbox.test.failed",
									{"iteration": iteration_num},
									project=project,
								)
							else:
								append_event(
									"sandbox.test.passed",
									{"iteration": iteration_num},
									project=project,
								)

						messages.append({
							"role": "tool",
							"tool_call_id": tc.id,
							"content": result,
						})

				# Add assistant message
				assistant_msg: dict[str, Any] = {
					"role": "assistant",
					"content": iteration_content,
				}
				if tool_calls:
					assistant_msg["tool_calls"] = [
						{
							"id": tc.id,
							"type": "function",
							"function": {
								"name": tc.function.name,
								"arguments": tc.function.arguments,
							},
						}
						for tc in tool_calls
					]
				messages.append(assistant_msg)

				# Check for Ja (the final answer)
				if G._message_ends_with_ja(iteration_content):
					ja_received = True
					append_event(
						"assistant.ja",
						{"iteration": iteration_num, "pending": len(pending.change_log), "answer": iteration_content if cfg.log_responses else "..."},
						project=project,
					)
					G.console.print(f"[green]Ja[/] [dim]— final answer received[/]")
					break

				# If no tools and no Ja, nudge
				if not tool_calls and not ja_received:
					G.console.print(f"[yellow]· iteration {iteration_num}: no Ja and no tools — nudging[/]")
					messages.append({
						"role": "system",
						"content": _NUDGE_NO_JA,
					})

		except KeyboardInterrupt:
			G.err.print(f"\nIteration {iteration_num}: aborted")
			append_event(
				"agent.aborted",
				{"iteration": iteration_num, "pending": len(pending.change_log)},
				project=project,
			)
			pending.discard()
			continue

		if not ja_received:
			pending.discard()
			continue

		# Auto-execute on Ja (always, no gate)
		if pending.change_log:
			summaries = commit_pending(project, pending)
			for s in summaries:
				append_event("tool.executed", {"summary": s["summary"]}, project=project)
				G.console.print(f"[green]executed[/] {s['summary']}")

		append_event(
			"michael.run.ended",
			{"iterations": iteration_num, "changes": len(pending.change_log)},
			project=project,
		)

		# Back to question prompt (or exit if user types exit)
		G.console.print("[dim]question answered. next question (or 'exit')[/]")

	append_event(
		"agent.ended",
		{"model": name, "kantian": True},
		project=project,
	)


def _run_agent_loop(
	"""Three-turn Kantian machine: scripture → target → iterate.

	Turn 1: Read scripture, interpret (read-only)
	Turn 2: Receive task, formulate target/goal/constraints
	Turn 3+: Iterate through Kantian questions with full toolset
	"""
	endpoint = _require_endpoint(profile, name)
	_ssh_preflight(cfg)

	client = llm_client(endpoint, profile.vllm_api_key)
	backend = make_backend(cfg)
	tools_read_only = _tools_for_mode("discussion")  # read-only tools for Turn 1
	tools_full = _tools_for_mode(mode)  # full toolset for Turn 3+
	base_prompt = cfg.resolved_system_prompt()

	backend_label = (
		"remote-podman (vps)" if cfg.vps_active()
		else ("local-podman" if isinstance(backend, LocalPodmanBackend)
		      else "no-sandbox")
	)
	G.console.print(
		f"[bold cyan]michael {verb_label}[/] [dim]project={project.slug}  "
		f"model={name}  mode={mode}  kantian=true  sandbox={backend_label}[/]"
	)
	G.console.print(
		f"[dim]Kantian machine (three-turn): scripture → target → iterate[/]"
	)

	session = PromptSession(
		history=FileHistory(str(G.REPL_HISTORY_PATH)),
		auto_suggest=AutoSuggestFromHistory(),
	)

	append_event(
		"agent.started",
		{
			"model": name,
			"served": profile.served_model_name,
			"mode": mode,
			"god": god_mode,
			"kantian": True,
			"sandbox": backend_label,
		},
		project=project,
	)

	while True:
		# Load scripture
		scripture = load_scripture()
		if scripture:
			append_event(
				"scripture.loaded",
				{
					"length": len(scripture),
					"lines": len(scripture.split("\n")),
				},
				project=project,
			)

		# TURN 1: Scripture Interpretation (read-only)
		turn1_prompt = kantian_turn1_prompt()
		if scripture:
			turn1_msg = f"{scripture}\n\n---\n\n{turn1_prompt}"
		else:
			turn1_msg = turn1_prompt

		header = build_header(project, base_prompt, mode=mode)
		messages: list[dict[str, Any]] = [
			{"role": "system", "content": header},
			{"role": "user", "content": turn1_msg},
		]

		try:
			G.console.print(f"[dim]· Turn 1: interpreting scripture…[/]")
			resp = client.chat.completions.create(
				model=profile.served_model_name,
				messages=messages,
				tools=tools_read_only,
				tool_choice="auto",
				stream=False,
				timeout=float(profile.request_timeout_s),
			)
		except Exception as e:
			G.err.print(f"LLM error (Turn 1): {e}")
			append_event(
				"error",
				{"where": "kantian_turn1", "msg": str(e)},
				project=project,
			)
			break

		msg = resp.choices[0].message
		turn1_content = msg.content or ""
		turn1_tool_calls = msg.tool_calls or []

		# Process any Turn 1 tool calls (read-only)
		for tc in turn1_tool_calls:
			tname = tc.function.name
			try:
				targs = json.loads(tc.function.arguments or "{}")
			except json.JSONDecodeError:
				targs = {}
			dispatch_tool_call(tname, targs, project, cfg, backend, PendingChanges())

		# Log Turn 1
		if turn1_content:
			append_event(
				"scripture.interpreted",
				{"interpretation": turn1_content if cfg.log_responses else "..."},
				project=project,
			)
		if not cfg.kantian_visible and god_mode:
			G.console.print(f"[dim]· Turn 1 complete (output suppressed in god mode)[/]")
		else:
			if turn1_content:
				G.console.print(f"[dim]{turn1_content}[/]")

		# Add Turn 1 response to messages
		messages.append({"role": "assistant", "content": turn1_content})
		if turn1_tool_calls:
			messages[-1]["tool_calls"] = [
				{
					"id": tc.id,
					"type": "function",
					"function": {
						"name": tc.function.name,
						"arguments": tc.function.arguments,
					},
				}
				for tc in turn1_tool_calls
			]

		# TURN 2: Task Reception & Target Formulation
		if god_mode:
			user_prompt = G._GOD_MODE_PROMPT
		else:
			try:
				user_prompt = session.prompt(">>> ")
			except (EOFError, KeyboardInterrupt):
				break
			user_prompt = (user_prompt or "").strip()
			if not user_prompt or user_prompt.lower() in ("quit", "exit"):
				break

		append_event(
			"prompt.sent",
			{
				"prompt": user_prompt,
				"model": name,
				"served": profile.served_model_name,
				"mode": mode,
				"turn": 2,
			},
			project=project,
		)

		turn2_prompt = kantian_turn2_prompt(user_prompt)
		messages.append({"role": "user", "content": turn2_prompt})

		try:
			G.console.print(f"[dim]· Turn 2: formulating target and goal…[/]")
			resp = client.chat.completions.create(
				model=profile.served_model_name,
				messages=messages,
				tools=tools_read_only,  # Still read-only for Turn 2
				tool_choice="auto",
				stream=False,
				timeout=float(profile.request_timeout_s),
			)
		except Exception as e:
			G.err.print(f"LLM error (Turn 2): {e}")
			append_event(
				"error",
				{"where": "kantian_turn2", "msg": str(e)},
				project=project,
			)
			break

		msg = resp.choices[0].message
		turn2_content = msg.content or ""

		# Parse target/goal/constraints from Turn 2
		target = ""
		goal = ""
		constraints = ""
		for line in turn2_content.split("\n"):
			if line.startswith("TARGET:"):
				target = line[7:].strip()
			elif line.startswith("GOAL:"):
				goal = line[5:].strip()
			elif line.startswith("CONSTRAINTS:"):
				constraints = line[12:].strip()

		append_event(
			"target.formulated",
			{
				"target": target,
				"goal": goal,
				"constraints": constraints,
			},
			project=project,
		)

		if not cfg.kantian_visible and god_mode:
			G.console.print(f"[dim]· Turn 2 complete (output suppressed in god mode)[/]")
		else:
			if turn2_content:
				G.console.print(f"[dim]{turn2_content}[/]")

		messages.append({"role": "assistant", "content": turn2_content})

		# TURN 3+: Kantian Iteration Loop (full toolset)
		pending = PendingChanges()
		iteration_num = 0
		ja_received = False

		try:
			while iteration_num < cfg.max_kantian_iterations:
				iteration_num += 1
				iteration_prompt = kantian_iteration_prompt(
					target, goal, iteration_num, cfg.max_kantian_iterations
				)
				messages.append({"role": "user", "content": iteration_prompt})

				G.console.print(f"[dim]· Turn 3.{iteration_num}: iterating Kantian questions…[/]")
				try:
					resp = client.chat.completions.create(
						model=profile.served_model_name,
						messages=messages,
						tools=tools_full,
						tool_choice="auto",
						stream=False,
						timeout=float(profile.request_timeout_s),
					)
				except Exception as e:
					G.err.print(f"LLM error (iteration {iteration_num}): {e}")
					append_event(
						"error",
						{
							"where": "kantian_iteration",
							"iteration": iteration_num,
							"msg": str(e),
						},
						project=project,
					)
					pending.discard()
					break

				msg = resp.choices[0].message
				iteration_content = msg.content or ""
				tool_calls = msg.tool_calls or []

				# Log iteration
				if iteration_content or tool_calls:
					payload: dict[str, Any] = {
						"iteration_num": iteration_num,
						"tools_called": [tc.function.name for tc in tool_calls],
					}
					if cfg.log_responses and iteration_content:
						payload["answer"] = iteration_content
					append_event(
						"kantian.iteration",
						payload,
						project=project,
					)

				# Process tool calls
				if tool_calls:
					for tc in tool_calls:
						tname = tc.function.name
						try:
							targs = json.loads(tc.function.arguments or "{}")
						except json.JSONDecodeError:
							targs = {}
						result = dispatch_tool_call(
							tname, targs, project, cfg, backend, pending,
						)
						messages.append({
							"role": "tool",
							"tool_call_id": tc.id,
							"content": result,
						})

				# Add assistant message to thread
				assistant_msg: dict[str, Any] = {
					"role": "assistant",
					"content": iteration_content,
				}
				if tool_calls:
					assistant_msg["tool_calls"] = [
						{
							"id": tc.id,
							"type": "function",
							"function": {
								"name": tc.function.name,
								"arguments": tc.function.arguments,
							},
						}
						for tc in tool_calls
					]
				messages.append(assistant_msg)

				# Check for Ja
				if G._message_ends_with_ja(iteration_content):
					ja_received = True
					append_event(
						"assistant.ja",
						{
							"iteration": iteration_num,
							"pending": len(pending.change_log),
						},
						project=project,
					)
					break

				# If no tool calls and no Ja, nudge
				if not tool_calls and not ja_received:
					G.console.print(
						f"[yellow]· iteration {iteration_num}: no Ja and no tools — nudging[/]"
					)
					messages.append({
						"role": "system",
						"content": _NUDGE_NO_JA,
					})

		except KeyboardInterrupt:
			G.err.print(f"\nIteration {iteration_num}: aborted by user")
			append_event(
				"agent.aborted",
				{"iteration": iteration_num, "pending": len(pending.change_log)},
				project=project,
			)
			pending.discard()
			if god_mode:
				break
			continue

		if not ja_received:
			pending.discard()
			if god_mode:
				break
			continue

		# Execution phase (same as stateless loop)
		if god_mode:
			if pending.change_log:
				summaries = commit_pending(project, pending)
				for s in summaries:
					G.console.print(f"[green]auto-applied[/] {s['summary']}")
			break
		else:
			approved = _present_pending_to_user(project, pending, iteration_content)
			if approved:
				summaries = commit_pending(project, pending)
				if summaries:
					G.console.print(f"[green]applied[/] {len(summaries)} change(s)")
			else:
				for entry in pending.change_log:
					append_event(
						"tool.rejected",
						{
							"tool": entry["tool"],
							"args": entry["args"],
							"summary": _summary_for(entry["tool"], entry["args"]),
							"delta": entry["delta"],
						},
						project=project,
					)
				pending.discard()
				G.console.print("[yellow]rejected[/] pending changes discarded")

	append_event(
		"agent.ended",
		{"model": name, "mode": mode, "god": god_mode, "kantian": True},
		project=project,
	)


def _run_agent_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    mode: str,
    *,
    verb_label: str,
    god_mode: bool = False,
    use_kantian: bool = True,
) -> None:
    """Shared agent-loop body for `run`, `new code`, `new discussion`, `nitro`.

    If use_kantian is True, runs the three-turn Kantian machine.
    Otherwise, uses the stateless loop.
    """
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)

    if use_kantian and cfg.use_stateful_kantian:
        return _run_kantian_loop(
            project, cfg, name, profile, mode,
            verb_label=verb_label, god_mode=god_mode
        )

    client = llm_client(endpoint, profile.vllm_api_key)
    backend = make_backend(cfg)
    tools = _tools_for_mode(mode)
    base_prompt = cfg.resolved_system_prompt()

    backend_label = (
        "remote-podman (vps)" if cfg.vps_active()
        else ("local-podman" if isinstance(backend, LocalPodmanBackend)
              else "no-sandbox")
    )
    G.console.print(
        f"[bold cyan]michael {verb_label}[/] [dim]project={project.slug}  "
        f"model={name}  mode={mode}  sandbox={backend_label}[/]"
    )
    G.console.print(
        f"[dim]empty line or 'quit' to exit · Ctrl-C aborts an in-flight "
        f"loop · LLM surfaces with the {G.JA_PASSPHRASE!r} passcode[/]"
    )

    session = PromptSession(
        history=FileHistory(str(G.REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    append_event(
        "agent.started",
        {
            "model": name,
            "served": profile.served_model_name,
            "mode": mode,
            "god": god_mode,
            "sandbox": backend_label,
        },
        project=project,
    )
    while True:
        if god_mode:
            user = G._GOD_MODE_PROMPT
        else:
            try:
                user = session.prompt(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            user = (user or "").strip()
            if not user or user.lower() in ("quit", "exit"):
                break

        append_event(
            "prompt.sent",
            {
                "prompt": user,
                "model": name,
                "served": profile.served_model_name,
                "mode": mode,
            },
            project=project,
        )

        header = build_header(project, base_prompt, mode=mode)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": header},
            {"role": "user", "content": user},
        ]
        pending = PendingChanges()
        turn = 0
        ja_received = False
        try:
            while True:
                turn += 1
                G.console.print(f"[dim]· turn {turn}: model thinking…[/]")
                try:
                    resp = client.chat.completions.create(
                        model=profile.served_model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        stream=False,
                        timeout=float(profile.request_timeout_s),
                    )
                except Exception as e:
                    G.err.print(f"LLM error: {e}")
                    append_event(
                        "error",
                        {"where": "agent_loop", "msg": str(e), "turn": turn},
                        project=project,
                    )
                    pending.discard()
                    break

                msg = resp.choices[0].message
                content = msg.content or ""
                if content:
                    payload: dict[str, Any] = {
                        "chars": len(content),
                        "model": name,
                        "served": profile.served_model_name,
                        "turn": turn,
                    }
                    if cfg.log_responses:
                        payload["text"] = content
                    append_event("assistant.message", payload, project=project)

                tool_calls = msg.tool_calls or []
                if tool_calls:
                    for tc in tool_calls:
                        G.console.print(f"[dim]· turn {turn}: tool {tc.function.name}[/]")
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if tool_calls:
                    for tc in tool_calls:
                        tname = tc.function.name
                        try:
                            targs = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            targs = {}
                        result = dispatch_tool_call(
                            tname, targs, project, cfg, backend, pending,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    continue

                if G._message_ends_with_ja(content):
                    ja_received = True
                    append_event(
                        "assistant.ja",
                        {"turn": turn, "pending": len(pending.change_log)},
                        project=project,
                    )
                    break
                G.console.print(
                    f"[yellow]· turn {turn}: no {G.JA_PASSPHRASE} and no tool "
                    f"calls — nudging the model back into the loop[/]"
                )
                messages.append({"role": "system", "content": _NUDGE_NO_JA})
                continue
        except KeyboardInterrupt:
            n_discarded = len(pending.change_log)
            G.err.print(f"\nturn {turn}: aborted by user; pending changes discarded")
            append_event(
                "agent.aborted",
                {"turn": turn, "pending": n_discarded},
                project=project,
            )
            if n_discarded:
                append_event(
                    "pending.discarded",
                    {
                        "turn": turn,
                        "count": n_discarded,
                        "tools": [e["tool"] for e in pending.change_log],
                    },
                    project=project,
                )
            pending.discard()
            if god_mode:
                break
            continue

        if not ja_received:
            pending.discard()
            if god_mode:
                break
            continue

        if god_mode:
            if content:
                G.console.print(
                    Panel(content, title="⚡ god — Ja", border_style="yellow")
                )
            if pending.change_log:
                summaries = commit_pending(project, pending)
                for s in summaries:
                    G.console.print(f"[green]auto-applied[/] {s['summary']}")
            break
        else:
            approved = _present_pending_to_user(project, pending, content)
            if approved:
                summaries = commit_pending(project, pending)
                if summaries:
                    G.console.print(f"[green]applied[/] {len(summaries)} change(s)")
            else:
                for entry in pending.change_log:
                    append_event(
                        "tool.rejected",
                        {
                            "tool": entry["tool"],
                            "args": entry["args"],
                            "summary": _summary_for(entry["tool"], entry["args"]),
                            "delta": entry["delta"],
                        },
                        project=project,
                    )
                pending.discard()
                G.console.print("[yellow]rejected[/] pending changes discarded")

    append_event(
        "agent.ended",
        {"model": name, "mode": mode, "god": god_mode},
        project=project,
    )
