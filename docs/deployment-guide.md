# Project Michael — Full Deployment Guide

From a fresh Termux install with an old copy of Michael, through a clean VPS, to a running
Vast.ai GPU endpoint. Follow the phases in order — each one depends on the last.

---

## Prerequisites

Before you start, have these in hand:

| Item | Where to get it |
|------|----------------|
| Termux with storage access | F-Droid or Google Play |
| VPS public IP, root SSH access | Your VPS provider console |
| Vast.ai account + API key | console.vast.ai → Account → API Keys |

---

## Phase 1 — Termux: upgrade from old install

### 1.1 — Update Termux packages

```bash
pkg update && pkg upgrade -y
```

### 1.2 — Get the latest code

If you already have a clone, pull the latest:

```bash
cd ~/project_michael
git pull origin main
```

If your old clone is stale or broken, re-clone cleanly:

```bash
cd ~
rm -rf project_michael
git clone https://github.com/ari6six6/project_michael project_michael
cd project_michael
```

### 1.3 — Run the Termux bootstrap

```bash
bash bootstrap_termux.sh
```

This script is idempotent — safe to run over an existing install. It:
- Installs `python`, `openssh`, `git`, `rsync`, `coreutils`, `nano` via `pkg`
- Installs all Python deps from `requirements.txt` via pip (uses `httpx`, not the OpenAI SDK — no Rust compiler needed)
- Creates `~/.michael/` state directory
- Generates `~/.ssh/id_ed25519` if it does not exist (skips if it does)
- Installs the `michael` wrapper at `$PREFIX/bin/michael`
- Runs `michael init` to create a stub `~/.michael/config.json`

### 1.4 — Verify

```bash
michael --help
```

You should see the full command list. If you get `command not found`, restart your Termux session and try again.

### 1.5 — Copy your SSH public key

You will need this in Phase 2:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output line. It starts with `ssh-ed25519 AAAA…`.

> **Backup tip:** Also copy `~/.ssh/id_ed25519` (the private key) to a secure location.
> After Phase 2, SSH password auth is disabled on the VPS. Losing this key means losing access.

---

## Phase 2 — VPS: bootstrap from scratch

All commands in this phase run on the VPS as **root** unless noted.

### 2.1 — SSH in as root

```bash
ssh root@<vps-ip>
```

### 2.2 — Clone the repo

```bash
git clone https://github.com/ari6six6/project_michael /opt/project_michael
cd /opt/project_michael
```

### 2.3 — Run the VPS bootstrap

```bash
bash bootstrap.sh
```

This takes a few minutes. It runs 11 numbered steps:

| Step | What it does |
|------|-------------|
| 1 | Installs UFW, fail2ban, podman, python3, git, tmux, jq, chrony |
| 2 | Sets timezone, locale, NTP |
| 3 | Enables automatic security updates |
| 4 | Creates the `michael` user with sudo access |
| 5 | Configures UFW: deny-all-in, allow SSH only |
| 6 | Hardens SSH: pubkey-only, no passwords, no root login |
| 7 | Configures fail2ban for SSH rate-limiting |
| 8 | Applies kernel hardening (sysctl, AppArmor) |
| 9 | Builds the `michael-sandbox:alpine` container image |
| 10 | Creates Python venv, installs deps, installs `/usr/local/bin/michael` wrapper |
| 11 | Creates `/home/michael/workspace` for remote sandboxes |

> **Warning from the script:** It will print an alert if `/root/.ssh/authorized_keys` is
> missing or empty. Keep your root SSH session open until you have confirmed you can SSH in
> as `michael` with your key (step 2.5 below). Do not close the root session prematurely.

### 2.4 — Add your Termux public key to the `michael` user

Still as root on the VPS:

```bash
mkdir -p /home/michael/.ssh
echo "ssh-ed25519 AAAA...  (paste the full line from Phase 1 step 1.5)" \
  >> /home/michael/.ssh/authorized_keys
chmod 700 /home/michael/.ssh
chmod 600 /home/michael/.ssh/authorized_keys
chown -R michael:michael /home/michael/.ssh
```

### 2.5 — Verify SSH access from Termux

Back in Termux (keep root session open as fallback):

```bash
ssh michael@<vps-ip>
```

If you get a shell prompt without a password prompt, Phase 2 is complete. Exit back to Termux.

### 2.6 — Confirm the sandbox image was built

```bash
ssh michael@<vps-ip> 'podman images'
```

Expected output includes a line like:

```
localhost/michael-sandbox  alpine  <hash>  ...
```

If it is missing, re-run step 9 manually:

```bash
ssh michael@<vps-ip> "cd /opt/project_michael && podman build -t michael-sandbox:alpine -f Dockerfile.sandbox ."
```

---

## Phase 3 — Vast.ai: rent a GPU and configure vLLM

### 3.1 — Choose a GPU and model

| VRAM | GPU example | Recommended model |
|------|------------|-------------------|
| 24 GB | RTX 4090 | `Qwen/Qwen2.5-Coder-32B-Instruct` (GPTQ/Q4) or `meta-llama/Llama-3.1-8B-Instruct` |
| 48 GB | RTX A6000 / L40S | `Qwen/Qwen2.5-32B-Instruct` (fp16) |
| 80 GB | H100 | `Qwen/Qwen2.5-72B-Instruct` or `deepseek-ai/DeepSeek-Coder-V2-Instruct` |

### 3.2 — Rent an instance

1. Go to **console.vast.ai → Search**
2. Filter by your target GPU
3. Choose an instance that has a vLLM-compatible template, or select a bare PyTorch image
4. In the **On-Start Command** field (or equivalent), enter:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <HF-model-id> \
  --served-model-name god \
  --api-key <your-chosen-api-key> \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384
```

Replace `<HF-model-id>` with the full Hugging Face model ID (e.g. `Qwen/Qwen2.5-Coder-32B-Instruct`)
and `<your-chosen-api-key>` with any secret string you choose (you will paste this into Michael's config).

5. Click **Rent** — do NOT start it yet (or start and immediately stop it to note the instance ID)

### 3.3 — Note the instance ID

The numeric instance ID appears in the Vast console URL when you click on the instance:
`https://console.vast.ai/instances/<ID>/`

Write this number down — you need it in Phase 4.

### 3.4 — Stop the instance (save credits)

Stop or do not start the instance yet. Michael will start it for you in Phase 5 via `michael up`.

---

## Phase 4 — Termux: wire up the config

### 4.1 — Open config

```bash
michael config
```

This opens `~/.michael/config.json` in `$EDITOR` (defaults to `nano`).

### 4.2 — Fill in all required fields

Replace the stub with:

```json
{
  "vast_api_key": "<your Vast.ai API key>",
  "default_model": "god",
  "models": {
    "god": {
      "vast_instance_id": "<numeric instance ID from Phase 3 step 3.3>",
      "served_model_name": "god",
      "vllm_api_key": "<the api-key string from your on-start command>",
      "vllm_internal_port": 8000,
      "request_timeout_s": 120
    }
  },
  "vps": {
    "host": "<vps-ip>",
    "port": 22,
    "user": "michael",
    "ssh_key_path": "~/.ssh/id_ed25519",
    "workspace_dir": "/home/michael/workspace"
  }
}
```

Save and exit (`Ctrl+X` → `Y` → `Enter` in nano).

### 4.3 — Verify VPS connectivity

```bash
michael ssh-test
```

Expected: `✓ VPS reachable  handshake Xms`

If it fails: re-check `vps.host`, `vps.user`, and that your SSH key is in
`/home/michael/.ssh/authorized_keys` on the VPS.

---

## Phase 5 — Start the GPU and wait for vLLM

```bash
michael up
```

What happens:
1. Michael calls the Vast.ai API to start your instance
2. It polls `GET /v1/models` on the vLLM endpoint every 10–60 seconds (exponential backoff)
3. You see progress: `[Xs] polling endpoint…`
4. When vLLM responds 200 OK, Michael caches the endpoint in `~/.michael/config.json`

**Expected total time:**
- If the model is already cached on the Vast instance disk: ~3–6 minutes
- If the model needs downloading from Hugging Face: 5–25 minutes depending on model size

When ready you'll see:
```
✓ endpoint ready: http://<ip>:<port>/v1
```

> **Termux tip:** For long waits, run `termux-wake-lock` first to prevent Android from
> suspending the session.

---

## Phase 6 — Create a project and run the agent

### 6.1 — Create a project

```bash
michael new myproject
michael use myproject
michael current        # should print: myproject
```

### 6.2 — Run a basic prompt

```bash
michael run hello, what can you do?
```

What happens:
1. Michael packages the four-header context (H1: your prompts, H2: filesystem snapshot,
   H3: tool call history, H4: protocol bible) and sends it to vLLM
2. The LLM iterates privately — you see dim status lines for each turn
3. When the LLM ends a message with the bareword `Ja`, all staged changes are auto-committed
4. The terminal prints what was applied and exits

### 6.3 — Use the sandbox

```bash
michael run write a Python script that prints the first 20 primes, run it in the sandbox, and show me the output
```

The LLM will:
- Call `write_file` to stage the script
- Call `run_in_sandbox` — which SSHes to the VPS and runs it in the `michael-sandbox:alpine` container
- Read the output and decide if it's correct
- Say `Ja` once satisfied → script is committed

### 6.4 — View the event log

```bash
michael log           # last 20 events
michael log --tail 50 # last 50 events
```

---

## Phase 7 — Everyday lifecycle

| Goal | Command |
|------|---------|
| Start the GPU for a session | `michael up` |
| Stop the GPU (save credits) | `michael down` |
| Check current state | `michael status` |
| One-shot LLM question (no tool loop) | `michael ask "what does this function do?"` |
| Full agent run | `michael run <your prompt>` |
| Run Python in isolated sandbox | `michael sandbox script.py` |
| Revert last committed change | `michael undo` |
| See revertible changes | `michael undo --list` |
| List all projects | `michael show` |
| Switch project | `michael use <slug>` |
| Interactive REPL | `michael` (no args) |

---

## Troubleshooting

### `michael up` times out (600s)
- Check the Vast console — is the instance actually running?
- Check the on-start command for syntax errors (especially quotes)
- SSH into the Vast instance and check `nvidia-smi` and vLLM logs
- Try a smaller model if VRAM is insufficient

### `michael ssh-test` fails
- Confirm `vps.host` is the correct IP
- Confirm your pubkey is in `/home/michael/.ssh/authorized_keys` on the VPS
- Confirm UFW allows your SSH port: `ssh root@<vps> ufw status`

### `run_in_sandbox` fails with "sandbox unavailable"
- Verify `vps.host` is set in config
- Run `michael ssh-test` to confirm VPS is reachable
- Confirm the sandbox image exists: `ssh michael@<vps> podman images`

### LLM never says `Ja`
- The model may need more context or a more specific prompt
- Try `michael ask` for a quick sanity check that the endpoint is working
- Check `michael log` to see LLM responses

### Vast.ai endpoint changes after restart
- Run `michael up` after each instance restart — it re-polls and re-caches the endpoint
- The old cached endpoint in `config.json` becomes invalid when the instance stops

---

## Security notes

- `~/.michael/config.json` is created `chmod 600` — it contains your Vast API key
- SSH password auth is disabled on the VPS after bootstrap — pubkey only
- The sandbox container runs as an unprivileged user, with no network (by default), read-only
  root, memory/CPU/PID limits, and all capabilities dropped
- `Ja` is auto-commit with no confirmation prompt — review the LLM's stated plan in the dim
  status output before it finalises work
