#!/usr/bin/env bash
# bootstrap.sh — idempotent root-bootstrap for a fresh Ubuntu 24.04 LTS VPS.
# Provisions the host that runs `michael` (the air-gapped AI control loop CLI).
# Run as root once, then verify SSH login as the new user in a separate shell
# before closing the current session.

set -euo pipefail

USERNAME="${USERNAME:-michael}"
OPERATOR_USER="${OPERATOR_USER:-ari}"
SSH_PORT="${SSH_PORT:-22}"
TIMEZONE="${TIMEZONE:-UTC}"
TTYD_LOCAL_PORT="${TTYD_LOCAL_PORT:-7681}"
WEB_PORT="${WEB_PORT:-443}"
TTYD_VERSION="${TTYD_VERSION:-1.7.7}"
WEB_CRED_FILE="/root/.web-terminal-credentials"
TTYD_BIN="/usr/local/bin/ttyd"
WEB_TLS_DIR="/etc/ssl/web-terminal"
NGINX_HTPASSWD="/etc/nginx/web-terminal.htpasswd"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: bootstrap.sh must be run as root." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

APT_CONFOLD=(-o "Dpkg::Options::=--force-confold")

echo "[1/13] apt update + upgrade and install base packages"
apt-get update -y
apt-get "${APT_CONFOLD[@]}" -y upgrade
apt-get "${APT_CONFOLD[@]}" -y install \
    ufw fail2ban unattended-upgrades needrestart chrony apparmor-utils \
    ca-certificates curl gnupg lsb-release git jq podman uidmap slirp4netns \
    tmux htop \
    nginx apache2-utils openssl

echo "[2/13] timezone, chrony, locale"
timedatectl set-timezone "${TIMEZONE}"
systemctl enable --now chrony
if ! locale -a | grep -qiE '^en_US\.utf-?8$'; then
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen || true
    echo "en_US.UTF-8 UTF-8" >>/etc/locale.gen
    locale-gen en_US.UTF-8
fi
update-locale LANG=en_US.UTF-8

echo "[3/13] unattended-upgrades configuration"
cat >/etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Download-Upgradeable-Packages "1";
EOF
cat >/etc/apt/apt.conf.d/52unattended-upgrades-local <<'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}";
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
    "${distro_id}:${distro_codename}-updates";
};
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "04:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF
systemctl enable --now unattended-upgrades.service

echo "[4/13] non-root user ${USERNAME} + sudoers"
if ! id -u "${USERNAME}" >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" "${USERNAME}"
fi
usermod -aG sudo "${USERNAME}"

USER_HOME="$(getent passwd "${USERNAME}" | cut -d: -f6)"
install -d -m 0700 -o "${USERNAME}" -g "${USERNAME}" "${USER_HOME}/.ssh"
if [[ -s /root/.ssh/authorized_keys ]]; then
    install -m 0600 -o "${USERNAME}" -g "${USERNAME}" \
        /root/.ssh/authorized_keys "${USER_HOME}/.ssh/authorized_keys"
else
    echo "WARNING: /root/.ssh/authorized_keys missing or empty —" \
         "you must copy a pubkey to ${USER_HOME}/.ssh/authorized_keys before logging out." >&2
fi

# NOTE: michael runs containers via *rootless* podman (uidmap + slirp4netns
# are installed above). The application never shells out to sudo, so we
# deliberately do NOT grant NOPASSWD on podman/docker — that would be
# equivalent to root (sudo podman run -v /:/host …). Keep journalctl only,
# for read-only diagnostics.
SUDOERS_FILE="/etc/sudoers.d/10-${USERNAME}-agent"
cat >"${SUDOERS_FILE}" <<EOF
${USERNAME} ALL=(root) NOPASSWD: /usr/bin/journalctl
EOF
chmod 0440 "${SUDOERS_FILE}"
visudo -cf "${SUDOERS_FILE}"

echo "[5/13] UFW firewall"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow "${SSH_PORT}/tcp"
ufw allow "${WEB_PORT}/tcp"
ufw --force enable
ufw status verbose

echo "[6/13] SSH hardening"
CLOUD_INIT_DROPIN="/etc/ssh/sshd_config.d/50-cloud-init.conf"
if [[ -f "${CLOUD_INIT_DROPIN}" ]]; then
    sed -i 's/^[[:space:]]*PasswordAuthentication[[:space:]].*/# &/' "${CLOUD_INIT_DROPIN}"
fi
cat >/etc/ssh/sshd_config.d/99-hardening.conf <<EOF
Port ${SSH_PORT}
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
ClientAliveInterval 300
ClientAliveCountMax 2
MaxAuthTries 3
LoginGraceTime 20
AllowUsers ${USERNAME}
EOF
sshd -t
systemctl reload ssh

echo "[7/13] fail2ban"
cat >/etc/fail2ban/jail.d/sshd-local.conf <<EOF
[sshd]
enabled = true
port = ${SSH_PORT}
maxretry = 3
findtime = 10m
bantime = 1h
EOF
systemctl enable --now fail2ban
fail2ban-client status sshd || true

echo "[8/13] sysctl + needrestart + apparmor"
cat >/etc/sysctl.d/99-hardening.conf <<'EOF'
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
EOF
sysctl --system

install -d -m 0755 /etc/needrestart/conf.d
cat >/etc/needrestart/conf.d/50-autorestart.conf <<'EOF'
$nrconf{restart} = 'a';
$nrconf{kernelhints} = 0;
EOF

aa-status || true

echo "[9/13] michael state dir, sandbox image, CLI install"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICHAEL_DIR="${USER_HOME}/.michael"
install -d -m 0700 -o "${USERNAME}" -g "${USERNAME}" "${MICHAEL_DIR}"
if [[ -f "${PROJECT_DIR}/Dockerfile.sandbox" ]]; then
    install -m 0644 -o "${USERNAME}" -g "${USERNAME}" \
        "${PROJECT_DIR}/Dockerfile.sandbox" "${MICHAEL_DIR}/Dockerfile.sandbox"
    sudo -u "${USERNAME}" podman build \
        -t michael-sandbox:alpine \
        -f "${MICHAEL_DIR}/Dockerfile.sandbox" \
        "${MICHAEL_DIR}/"
else
    echo "NOTE: Dockerfile.sandbox not found; skipping sandbox image build." >&2
fi

# Install the michael CLI into a venv (Ubuntu 24.04 system Python is
# PEP 668-protected) and expose /usr/local/bin/michael as a thin wrapper.
apt-get "${APT_CONFOLD[@]}" -y install python3-venv python3-pip
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${PROJECT_DIR}/requirements.txt"

cat >/usr/local/bin/michael <<EOF
#!/bin/sh
exec "${VENV_DIR}/bin/python" "${PROJECT_DIR}/main.py" "\$@"
EOF
chmod 0755 /usr/local/bin/michael

echo "[10/13] operator user ${OPERATOR_USER} (web-terminal account, no SSH)"
if ! id -u "${OPERATOR_USER}" >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" "${OPERATOR_USER}"
fi
usermod -aG sudo "${OPERATOR_USER}"

# Set (or rotate) the operator's password and the basic-auth password to the
# same random secret. Persist it so it can be re-printed; not regenerated on
# re-runs unless WEB_REGEN=1.
if [[ -s "${WEB_CRED_FILE}" && "${WEB_REGEN:-0}" != "1" ]]; then
    WEB_PASSWORD="$(grep -E '^password=' "${WEB_CRED_FILE}" | head -n1 | cut -d= -f2-)"
fi
if [[ -z "${WEB_PASSWORD:-}" ]]; then
    WEB_PASSWORD="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)"
fi
echo "${OPERATOR_USER}:${WEB_PASSWORD}" | chpasswd

# operator gets passworded sudo (defense-in-depth on top of basic-auth).
SUDOERS_OP="/etc/sudoers.d/20-${OPERATOR_USER}-operator"
cat >"${SUDOERS_OP}" <<EOF
${OPERATOR_USER} ALL=(ALL:ALL) ALL
EOF
chmod 0440 "${SUDOERS_OP}"
visudo -cf "${SUDOERS_OP}"

# Build the sandbox image under the operator's rootless podman storage too,
# so `michael sandbox foo.py` works from the web terminal as ${OPERATOR_USER}.
# Each rootless podman user has its own image store.
OP_HOME="$(getent passwd "${OPERATOR_USER}" | cut -d: -f6)"
OP_MICHAEL_DIR="${OP_HOME}/.michael"
install -d -m 0700 -o "${OPERATOR_USER}" -g "${OPERATOR_USER}" "${OP_MICHAEL_DIR}"
if [[ -f "${PROJECT_DIR}/Dockerfile.sandbox" ]]; then
    install -m 0644 -o "${OPERATOR_USER}" -g "${OPERATOR_USER}" \
        "${PROJECT_DIR}/Dockerfile.sandbox" "${OP_MICHAEL_DIR}/Dockerfile.sandbox"
    sudo -u "${OPERATOR_USER}" podman build \
        -t michael-sandbox:alpine \
        -f "${OP_MICHAEL_DIR}/Dockerfile.sandbox" \
        "${OP_MICHAEL_DIR}/" || \
        echo "WARNING: sandbox build for ${OPERATOR_USER} failed (non-fatal)" >&2
fi

echo "[11/13] ttyd binary + loopback systemd unit"
if [[ ! -x "${TTYD_BIN}" ]] || ! "${TTYD_BIN}" --version 2>/dev/null | grep -q "${TTYD_VERSION}"; then
    curl -fsSL -o "${TTYD_BIN}" \
        "https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.x86_64"
    chmod 0755 "${TTYD_BIN}"
fi

# ttyd listens only on loopback; nginx terminates TLS and gates basic-auth.
# Auth+TLS are intentionally NOT in the ttyd cmdline: keeping the password
# out of /proc/<pid>/cmdline is the whole reason for fronting with nginx.
cat >/etc/systemd/system/ttyd.service <<EOF
[Unit]
Description=ttyd web terminal (loopback; fronted by nginx)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${OPERATOR_USER}
Group=${OPERATOR_USER}
WorkingDirectory=/home/${OPERATOR_USER}
ExecStart=${TTYD_BIN} \\
    --port ${TTYD_LOCAL_PORT} \\
    --interface 127.0.0.1 \\
    --writable \\
    --max-clients 4 \\
    /bin/bash -l
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
chmod 0644 /etc/systemd/system/ttyd.service
systemctl daemon-reload
systemctl enable --now ttyd.service
sleep 1
systemctl --no-pager --full status ttyd.service || true

echo "[12/13] self-signed TLS cert + htpasswd for nginx"
install -d -m 0755 "${WEB_TLS_DIR}"
PUBLIC_IP="$(curl -fsS --max-time 5 https://api.ipify.org || true)"
CERT_CN="${PUBLIC_IP:-michael-vps}"
if [[ ! -s "${WEB_TLS_DIR}/cert.pem" || ! -s "${WEB_TLS_DIR}/key.pem" || "${WEB_REGEN:-0}" == "1" ]]; then
    openssl req -x509 -nodes -newkey rsa:2048 \
        -days 3650 \
        -subj "/CN=${CERT_CN}" \
        -addext "subjectAltName=IP:${PUBLIC_IP:-127.0.0.1}" \
        -keyout "${WEB_TLS_DIR}/key.pem" \
        -out "${WEB_TLS_DIR}/cert.pem"
fi
chmod 0600 "${WEB_TLS_DIR}/key.pem"
chmod 0644 "${WEB_TLS_DIR}/cert.pem"
chown root:root "${WEB_TLS_DIR}/key.pem" "${WEB_TLS_DIR}/cert.pem"

# bcrypt-hashed htpasswd. -B = bcrypt, -b = take password from cmdline (this
# command is transient; only the resulting file persists). htpasswd file is
# readable by www-data only.
htpasswd -bcB "${NGINX_HTPASSWD}" "${OPERATOR_USER}" "${WEB_PASSWORD}"
chown root:www-data "${NGINX_HTPASSWD}"
chmod 0640 "${NGINX_HTPASSWD}"

# Persist credentials (root-only) so they can be re-printed later.
umask 077
cat >"${WEB_CRED_FILE}" <<EOF
url=https://${PUBLIC_IP:-<vps-ip>}/
username=${OPERATOR_USER}
password=${WEB_PASSWORD}
EOF
chmod 0400 "${WEB_CRED_FILE}"

echo "[13/13] nginx reverse proxy (HTTPS :${WEB_PORT} -> 127.0.0.1:${TTYD_LOCAL_PORT})"
rm -f /etc/nginx/sites-enabled/default
cat >/etc/nginx/sites-available/web-terminal <<EOF
server {
    listen ${WEB_PORT} ssl;
    listen [::]:${WEB_PORT} ssl;
    server_name _;

    ssl_certificate     ${WEB_TLS_DIR}/cert.pem;
    ssl_certificate_key ${WEB_TLS_DIR}/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    auth_basic           "web terminal";
    auth_basic_user_file ${NGINX_HTPASSWD};

    client_max_body_size 0;

    location / {
        proxy_pass         http://127.0.0.1:${TTYD_LOCAL_PORT};
        proxy_http_version 1.1;

        proxy_set_header Upgrade    \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       \$host;
        proxy_set_header X-Real-IP  \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # ttyd uses a long-lived WebSocket; keep the connection open.
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_buffering    off;
    }
}
EOF
ln -sf /etc/nginx/sites-available/web-terminal /etc/nginx/sites-enabled/web-terminal
nginx -t
systemctl enable --now nginx
systemctl reload nginx

cat <<EOF

================================================================
  bootstrap complete.

  WEB TERMINAL (use Safari over iCloud Private Relay):
      URL:      https://${PUBLIC_IP:-<vps-ip>}/
      User:     ${OPERATOR_USER}
      Password: ${WEB_PASSWORD}

  Traffic is TLS-encrypted (TLS 1.2/1.3). The cert is self-signed —
  Safari will show a "not private" warning on first visit because no
  public CA vouches for it (we have no domain). Encryption is real;
  trust is what's missing. Tap "Show Details" -> "visit this
  website" once per device to accept it.

  Architecture: nginx (:${WEB_PORT}, TLS + bcrypt htpasswd) -> ttyd
  (127.0.0.1:${TTYD_LOCAL_PORT}, no auth, no TLS, loopback only) -> bash -l
  as ${OPERATOR_USER}. Password is in ${NGINX_HTPASSWD} (mode 0640
  root:www-data), never on a process command line.

  Credentials also persisted at ${WEB_CRED_FILE} (root-only).
  To rotate them later: WEB_REGEN=1 ./bootstrap.sh

  michael CLI is on PATH for any user: just run \`michael\`.
  Next steps from the web terminal as ${OPERATOR_USER}:
      michael init
      \$EDITOR ~/.michael/config.json    # paste vast/vllm keys
      michael up                        # resumes the GPU instance

  SSH (kept available; key-only, no password, no root):
      ssh -p ${SSH_PORT} ${USERNAME}@${PUBLIC_IP:-<vps-ip>}

  BEFORE CLOSING THIS ROOT SESSION, open a NEW shell and verify
  EITHER the web terminal OR the SSH login works. Only then is
  it safe to log out as root — root login and password auth on
  SSH are disabled and cannot be recovered without console access.
================================================================
EOF
