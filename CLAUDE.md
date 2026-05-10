# Project Michael — Architecture & Deployment

Project Michael is an event-sourced, air-gapped AI control loop CLI. The phone (Termux or
Linux) is the control plane; a hardened VPS handles sandboxed code execution; a Vast.ai GPU
cluster runs the LLM inference. Every prompt rebuilds a four-header context package from the
project event log — the LLM is stateless, the log is its memory.

---

## Architecture

```
Phone (Termux / Linux)          VPS (Ubuntu 24.04, rootless podman)
  michael CLI ──────SSH──────▶  sandbox execution
       │                              ▲
       │ HTTP (OpenAI protocol)       │ staged code detonated here
       ▼                              │
  Vast.ai GPU cluster ─────────────▶ results back to CLI
  vLLM endpoint
```

**One model, full authority, always:**
There is a single model profile (`god`). It runs on whatever GPU cluster you rent — RTX 4090
for smaller models, H100s for the heavy ones. No tier switching, no mode selection.

**Four-header context package** (sent on every fresh LLM instance):
- H1 — user's prompts verbatim, in order
- H2 — live filesystem snapshot of the project
- H3 — every tool call executed in this project (causal chain)
- H4 — protocol Bible (the contract the LLM operates under)

**The Ja gate:** The LLM iterates privately with Michael — reading files, running the sandbox,
patching code — until it ends a message with the bareword `Ja`. On Ja, all staged changes are
**auto-committed immediately**. No Y/n. The prompt exits. What you get is what it gave.

**Dual filesystem zones:**
| Zone | Path | LLM tool access |
|------|------|-----------------|
| Central FS | `~/.michael/` | Read-only. Writes blocked at Python layer before any I/O. |
| Work FS | Everything else | Unrestricted — `write_file`/`apply_patch` accept absolute paths; `run_shell` has full system access. |

The Central FS holds all headers source data (events, config, state). Michael's application code
writes there freely; LLM tool calls are categorically blocked from doing so. Enforcement lives in
`michael/permissions.py` and is applied inside every write path in `michael/tools.py`.

---

## Deploy Checklist

### 1. Termux / Linux (control plane)

```bash
# Clone and install
git clone <repo> project_michael && cd project_michael
bash bootstrap_termux.sh          # Termux
# or: pip install -r requirements.txt && echo 'alias michael="python main.py"' >> ~/.bashrc

# Initialise config
michael init
michael config                    # fill in vast_api_key, instance ID, served_model_name
```

### 2. VPS (sandbox, run once as root)

```bash
git clone <repo> && bash bootstrap.sh
```

The script creates a `michael` user, hardens SSH, installs rootless podman, builds the
sandbox image, and creates `~/workspace`.

After bootstrap, copy your SSH public key to the VPS:
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub michael@<vps-ip>
michael ssh-test                  # verify roundtrip
```

### 3. Vast.ai GPU

1. Rent an instance — any GPU that fits your model (RTX 4090 for ≤32B, H100s for larger).
2. Note the instance ID from the Vast console.
3. Add to `~/.michael/config.json`:
   ```json
   "models": {
     "god": {
       "vast_instance_id": "12345",
       "served_model_name": "your-model-name"
     }
   }
   ```
4. Start inference: `michael up` (polls until vLLM is ready, caches the endpoint).

### 4. First run

```bash
michael new myproject             # creates project, sets it active
michael run fix the auth bug in login.py
```

The LLM reads your code, iterates in private, commits on Ja. Done.

---

## Config Keys

| Key | Description |
|-----|-------------|
| `vast_api_key` | Vast.ai console API key |
| `default_model` | Profile to use (default: `god`) |
| `models.god.vast_instance_id` | Numeric ID of the rented GPU instance |
| `models.god.served_model_name` | Matches `--served-model-name` on vLLM |
| `models.god.vllm_api_key` | Key vLLM was launched with (or empty) |
| `models.god.vllm_internal_port` | Container-internal port (default 8000) |
| `models.god.request_timeout_s` | LLM request timeout in seconds |
| `vps.host` | VPS public IP/hostname (empty = no remote sandbox) |
| `vps.user` | SSH user (default: `michael`) |
| `vps.ssh_key_path` | Path to private key (default: `~/.ssh/id_ed25519`) |
| `vps.workspace_dir` | Workspace dir on the VPS |
| `sandbox.image` | Tag of the sandbox image built by `bootstrap.sh` |
| `sandbox.memory_mb` | Sandbox memory cap in MB |
| `sandbox.cpus` | Sandbox CPU cap |
| `sandbox.pids` | Sandbox PID cap |
| `sandbox.timeout_s` | Default sandbox timeout in seconds |
| `system_prompt` | Default system prompt for the agent loop |
| `system_prompt_file` | If set, reads system prompt from this file |
| `log_responses` | If true (default), stores full LLM responses in events.jsonl |
| `boot_poll_s` | Poll interval while waiting for vLLM to come up |

---

## Command Reference

| Command | Description |
|---------|-------------|
| `michael init` | Write stub config if missing |
| `michael show` | List all projects |
| `michael new [name]` | Create a new project |
| `michael use <slug>` | Switch active project |
| `michael current` | Print active project |
| `michael config` | Open `config.json` in `$EDITOR` |
| `michael up` | Start Vast.ai instance, wait for vLLM |
| `michael down` | Stop Vast.ai instance |
| `michael status` | Derived state from event log |
| `michael ask "<prompt>"` | One-shot LLM call (no tool loop) |
| `michael run <prompt…>` | **Run the agent.** Everything after `run` is the prompt |
| `michael log [--tail N]` | Show event log (last 20 by default) |
| `michael sandbox <file.py>` | Run Python file in isolated sandbox |
| `michael undo [--list] [<id>]` | Restore the most recent (or named) change |
| `michael ssh-test` | Verify VPS reachability, report handshake time |

---

## How a Run Works

```
michael run refactor the parser to handle unicode edge cases
```

1. Michael packages H1–H4 (your prompts, filesystem, tool history, protocol) and sends it with
   your prompt to the vLLM endpoint.
2. The LLM iterates privately: reads files, patches code, runs the sandbox — as many turns as
   needed. You see dim status lines.
3. When the LLM ends a message with `Ja`, Michael auto-commits every staged change.
4. The terminal shows what was applied. The command exits.

No Y/n. No interactive loop. One prompt → concrete result.

---

## Tools Available to the LLM

| Tool | Behaviour |
|------|-----------|
| `write_file(path, content, expected_changes)` | Staged; auto-committed on Ja |
| `apply_patch(path, unified_diff, expected_changes)` | Same staging flow as write_file |
| `read_file(path)` | Auto-executes, no confirmation |
| `list_dir(path='.')` | Auto-executes, no confirmation |
| `search_memory(query)` | Auto-executes; searches stored LLM responses in this project |
| `run_in_sandbox(python_code)` | Isolated podman, auto-executes |
| `run_shell(cmd, timeout_s=60)` | Runs in project workspace, auto-executes |
| `fetch_page(url, selector='', timeout_s=30)` | Auto-executes; HTTP GET over internet — returns full page text, all links with anchor text, form fields, and raw JSON for API endpoints |

---

## State Layout

```
~/.michael/
  config.json                    # global config (chmod 600)
  events.jsonl                   # global event log (instance lifecycle)
  state.json                     # derived state (endpoint cache, etc.)
  projects/
    <slug>/
      config.json                # per-project stub
      events.jsonl               # per-project log (prompts, tool calls, LLM responses)
      trash/                     # pre-change snapshots for undo
  ssh-*.sock                     # SSH control-master sockets
  repl_history                   # REPL command history
```
