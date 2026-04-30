#!/usr/bin/env bash
# init.sh — paste-into-VPS-web-console launcher.
#
# One-liner (run in the provider's web terminal as root):
#   curl -fsSL https://raw.githubusercontent.com/ari6six6/project_michael/claude/setup-deployment-terminal-OIP9X/init.sh | bash
#
# Clones the repo into /opt/project_michael and hands off to bootstrap.sh.
# Idempotent: re-running fast-forwards the checkout and re-runs bootstrap.

set -euo pipefail

REPO="${REPO:-https://github.com/ari6six6/project_michael.git}"
BRANCH="${BRANCH:-claude/setup-deployment-terminal-OIP9X}"
DEST="${DEST:-/opt/project_michael}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: init.sh must be run as root (paste with sudo or in a root web console)." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git ca-certificates curl

if [[ -d "${DEST}/.git" ]]; then
    git -C "${DEST}" remote set-url origin "${REPO}"
    git -C "${DEST}" fetch --depth=1 origin "${BRANCH}"
    git -C "${DEST}" checkout -B "${BRANCH}" "origin/${BRANCH}"
    git -C "${DEST}" reset --hard "origin/${BRANCH}"
else
    install -d -m 0755 "$(dirname "${DEST}")"
    git clone --depth=1 --branch "${BRANCH}" "${REPO}" "${DEST}"
fi

cd "${DEST}"
chmod +x bootstrap.sh
exec ./bootstrap.sh
