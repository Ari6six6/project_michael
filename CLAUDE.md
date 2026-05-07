# Project Michael — Architecture & Deployment

Project Michael is an event-sourced, air-gapped AI control loop CLI. The phone (Termux or
Linux) is the control plane; a hardened VPS handles sandboxed code execution; Vast.ai GPU
clusters run the LLM inference. Every prompt rebuilds a four-header context package from the
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

**Compute tiers:**
- `coder` / default model — 30B coder (RTX 4090 or similar, cheap, on constantly)
- `big` / instruct model — 30B instruct (same tier, separate GPU)
- `nitro` / big model — 235B+ (H100 cluster, expensive, spin up on demand)

**Four-header context package** (sent on every fresh LLM instance):
- H1 — user's prompts verbatim, in order
- H2 — live filesystem snapshot of the project
- H3 — every tool call executed in this project (causal chain)
- H4 — protocol Bible (the contract the LLM operates under)

**The Ja gate:** LLM signals "I am done; show the user" by ending a message with the bareword
`Ja`. Until then it iterates privately with Michael. On Ja, staged changes are presented and
the user approves or rejects with Y/n.

**God mode** (`nitro --god`): no user types anything. Michael fires a hardcoded
"burn this or let it be" prompt at the heavy model. On Ja, changes are **auto-committed**
without a Y/n gate. One shot. What you get is what you deserve.

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
michael config                    # fill in vast_api_key, instance IDs, vps.host
```

### 2. VPS (sandbox, run once as root)

```bash
curl https://<your-host>/init.sh | bash
# or: git clone <repo> && bash bootstrap.sh
```

The script creates a `michael` user, hardens SSH, installs rootless podman, builds the
sandbox image, and creates `~/workspace`.

After bootstrap, copy your Termux SSH public key to the VPS:
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub michael@<vps-ip>
michael ssh-test                  # verify roundtrip
```

### 3. Vast.ai GPU

1. Rent an instance (RTX 4090 for coder, H100 for nitro).
2. Note the instance ID from the Vast console.
3. Add to `~/.michael/config.json`:
   ```json
   "models": {
     "coder": { "vast_instance_id": "12345", "served_model_name": "qwen3-coder" },
     "nitro": { "vast_instance_id": "67890", "served_model_name": "qwen3-235b" }
   }
   ```
4. Start inference: `michael up` (polls until vLLM is ready, caches the endpoint).

### 4. First run

```bash
michael new project myproject     # creates project, sets it active
michael new code                  # enters agent loop on coder model
>>> hey                           # first prompt — the beginning
```

---

## Config Keys

| Key | Description |
|-----|-------------|
| `vast_api_key` | Vast.ai console API key |
| `default_model` | Profile used when no `--model` flag is passed |
| `models.<name>.vast_instance_id` | Numeric ID of the rented GPU instance |
| `models.<name>.served_model_name` | Matches `--served-model-name` on vLLM |
| `models.<name>.vllm_api_key` | Key vLLM was launched with (or empty) |
| `models.<name>.vllm_internal_port` | Container-internal port (default 8000) |
| `models.<name>.request_timeout_s` | LLM request timeout in seconds |
| `vps.host` | VPS public IP/hostname (empty = no remote sandbox) |
| `vps.user` | SSH user (default: `michael`) |
| `vps.ssh_key_path` | Path to private key (default: `~/.ssh/id_ed25519`) |
| `vps.workspace_dir` | Workspace dir on the VPS |
| `sandbox.image` | Tag of the sandbox image built by `bootstrap.sh` |
| `sandbox.memory_mb` | Sandbox memory cap in MB |
| `sandbox.cpus` | Sandbox CPU cap |
| `sandbox.pids` | Sandbox PID cap |
| `sandbox.timeout_s` | Default sandbox timeout in seconds |
| `system_prompt` | Default system prompt for agent loops |
| `system_prompt_file` | If set, reads system prompt from this file |
| `log_responses` | If true (default), stores full LLM responses in events.jsonl — required for `search_memory` |
| `boot_poll_s` | Poll interval while waiting for vLLM to come up |

---

## Command Reference

| Command | Description |
|---------|-------------|
| `michael init` | Write stub config if missing |
| `michael show` | List all projects |
| `michael new project [name]` | Create a new project |
| `michael new code [--model P]` | Fresh agent loop, full toolset |
| `michael new discussion [--model P]` | Fresh agent loop, read-only (no writes) |
| `michael nitro [--model P]` | Heavy model, same interactive loop as code |
| `michael nitro --god [--model P]` | **God mode** — see below |
| `michael use <slug>` | Switch active project |
| `michael current` | Print active project |
| `michael config` | Open `config.json` in `$EDITOR` |
| `michael up [--model P]` | Start Vast.ai instance, wait for vLLM |
| `michael down [--model P]` | Stop Vast.ai instance |
| `michael status` | Derived state from event log |
| `michael ask "<prompt>" [--model P]` | One-shot LLM call |
| `michael run [--model P]` | Alias for `new code` |
| `michael log [--tail N]` | Show event log (last 20 by default) |
| `michael sandbox <file.py>` | Run Python file in isolated sandbox |
| `michael undo [--list] [<id>]` | Restore the most recent (or named) change |
| `michael ssh-test` | Verify VPS reachability, report handshake time |

---

## God Mode

`michael nitro --god` engages the heavy model with full authority and no user gate.

**What it does:**
1. Fires the hardcoded prompt: *"Assess the full state of this project. Burn what is not
   working. Let stand what is righteous. Propose your changes."*
2. The model iterates privately with the full toolset — read, write, patch, sandbox — using
   the complete four-header context package.
3. When the model emits `Ja`, Michael auto-commits every staged change immediately.
4. The user sees what was applied. There is no Y/n. What you get is what it gave.

**When to use it:** When the scope of what needs fixing is too large to specify precisely, and
you trust the model's read of the project state more than your ability to describe it. The
heavy model sees everything at once — all your prompts, every file, every command that ever
ran. Point it at the problem and let it burn.

**Note:** Cold-start the heavy model first with `michael up --model nitro`. The first inference
after a cold start takes a few minutes (VRAM load). Subsequent calls are fast.

---

## Tools Available to the LLM

| Tool | Behaviour |
|------|-----------|
| `write_file(path, content, expected_changes)` | Staged, predicted-delta required, user confirms |
| `apply_patch(path, unified_diff, expected_changes)` | Same staging flow as write_file |
| `read_file(path)` | Auto-executes, no confirmation |
| `list_dir(path='.')` | Auto-executes, no confirmation |
| `search_memory(query)` | Auto-executes; searches stored LLM responses in this project |
| `run_in_sandbox(python_code)` | Isolated podman, requires user confirmation |
| `run_shell(cmd, timeout_s=60)` | Runs in project workspace, requires user confirmation |

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
