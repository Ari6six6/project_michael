#!/usr/bin/env bash
# bootstrap_termux.sh — install Michael's runtime on Termux (Android).
#
# Run from the project directory (where main.py lives):
#     bash bootstrap_termux.sh
#
# Idempotent: re-running won't damage existing state.
# Does NOT install podman/docker/ufw/fail2ban — sandboxing happens on the VPS
# over SSH (configure vps.host in ~/.michael/config.json).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PUBKEY_PATH="${HOME}/.ssh/id_ed25519.pub"
KEY_PATH="${HOME}/.ssh/id_ed25519"

echo "==> michael termux bootstrap"
echo "    project: ${PROJECT_DIR}"

if [[ -z "${PREFIX:-}" ]] || [[ "${PREFIX}" != *com.termux* ]]; then
    echo "ERROR: this script is for Termux only." >&2
    echo "       \$PREFIX must be set to a Termux path (got: ${PREFIX:-unset})" >&2
    exit 1
fi

echo "==> [1/6] pkg update + dependencies"
pkg update
pkg upgrade -y
pkg install -y python openssh git rsync coreutils nano

echo "==> [2/6] python deps (openai SDK replaced by httpx — no Rust build required)"
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "==> [3/6] state directory"
install -d -m 0700 "${HOME}/.michael"

echo "==> [4/6] ssh key"
install -d -m 0700 "${HOME}/.ssh"
if [[ ! -f "${KEY_PATH}" ]]; then
    ssh-keygen -t ed25519 -N "" -f "${KEY_PATH}" -C "michael-on-termux"
    echo "    generated new ed25519 key at ${KEY_PATH}"
else
    echo "    ssh key already exists at ${KEY_PATH}"
fi

echo "==> [5/6] michael wrapper"
WRAPPER="${PREFIX}/bin/michael"
cat >"${WRAPPER}" <<EOF
#!/usr/bin/env bash
exec python "${PROJECT_DIR}/main.py" "\$@"
EOF
chmod +x "${WRAPPER}"
echo "    installed: ${WRAPPER}"

echo "==> [6/6] config stub"
michael init || true

echo
echo "============================================================"
echo "  michael bootstrap complete."
echo "============================================================"
echo
echo "  next steps"
echo "  ---------"
echo
echo "  1. Add this pubkey to michael@<your-vps>:~/.ssh/authorized_keys :"
echo
sed 's/^/       /' "${PUBKEY_PATH}"
echo
echo "     On the VPS, after running bootstrap.sh as root:"
echo "       echo 'PUBKEY_HERE' >> /home/michael/.ssh/authorized_keys"
echo "       chown michael:michael /home/michael/.ssh/authorized_keys"
echo "       chmod 600 /home/michael/.ssh/authorized_keys"
echo
echo "  2. Edit your config:"
echo "       michael config              # opens in nano"
echo
echo "     Required fields:"
echo "       vast_api_key                  Vast.ai console API key"
echo "       default_model                 e.g. 'coder'"
echo "       models.<name>.vast_instance_id"
echo "       models.<name>.served_model_name"
echo "       models.<name>.vllm_api_key"
echo
echo "     Optional (enables remote sandbox on the VPS):"
echo "       vps.host                      VPS public IP/hostname"
echo "       vps.user                      ssh user (default: michael)"
echo "       vps.ssh_key_path              ~/.ssh/id_ed25519"
echo "       vps.workspace_dir             /home/michael/workspace"
echo
echo "  3. Verify VPS reachability (only if vps.host is set):"
echo "       michael ssh-test"
echo
echo "  4. Boot the GPU and chat:"
echo "       michael up --model coder      # start the Vast.ai instance"
echo "       michael run --model coder     # interactive REPL"
echo "       michael down --model coder    # pause to stop GPU billing"
echo
echo "  5. Keep Michael alive in the background while you're elsewhere:"
echo "       termux-wake-lock              # release with: termux-wake-unlock"
echo
