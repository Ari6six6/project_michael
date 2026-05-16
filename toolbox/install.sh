#!/usr/bin/env bash
# Install toolbox tools to ~/.michael/toolbox/ so the agent can call them in any project.
set -euo pipefail

DEST="${HOME}/.michael/toolbox"
SRC="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DEST"

count=0
for f in "$SRC"/*.py; do
    name="$(basename "$f")"
    cp -f "$f" "$DEST/$name"
    echo "installed: $name"
    ((count++))
done

echo ""
echo "toolbox: $count tool(s) installed to $DEST"
echo "These are now available to the michael agent in every project."
