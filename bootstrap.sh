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
TTYD_PORT="${TTYD_PORT:-443}"
TTYD_VERSION="${TTYD_VERSION:-1.7.7}"
WEB_CRED_FILE="/root/.web-terminal-credentials"
TTYD_BIN="/usr/local/bin/ttyd"
TTYD_ETC="/etc/ttyd"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: bootstrap.sh must be run as root." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

APT_CONFOLD=(-o "Dpkg::Options::=--force-confold")

echo "[1/12] apt update + upgrade and install base packages"
apt-get update -y
apt-get "${APT_CONFOLD[@]}" -y upgrade
apt-get "${APT_CONFOLD[@]}" -y install \
    ufw fail2ban unattended-upgrades needrestart chrony apparmor-utils \
    ca-certificates curl gnupg lsb-release git jq podman uidmap slirp4netns \
    tmux htop

echo "[2/12] timezone, chrony, locale"
timedatectl set-timezone "${TIMEZONE}"
systemctl enable --now chrony
if ! locale -a | grep -qiE '^en_US\.utf-?8$'; then
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen || true
    echo "en_US.UTF-8 UTF-8" >>/etc/locale.gen
    locale-gen en_US.UTF-8
fi
update-locale LANG=en_US.UTF-8

echo "[3/12] unattended-upgrades configuration"
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

echo "[4/12] non-root user ${USERNAME} + sudoers"
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

echo "[5/12] UFW firewall"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow "${SSH_PORT}/tcp"
ufw allow "${TTYD_PORT}/tcp"
ufw --force enable
ufw status verbose

echo "[6/12] SSH hardening"
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

echo "[7/12] fail2ban"
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

echo "[8/12] sysctl + needrestart + apparmor"
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

echo "[9/12] michael state dir + sandbox image"
MICHAEL_DIR="${USER_HOME}/.michael"
install -d -m 0700 -o "${USERNAME}" -g "${USERNAME}" "${MICHAEL_DIR}"
if [[ -f Dockerfile.sandbox ]]; then
    install -m 0644 -o "${USERNAME}" -g "${USERNAME}" \
        Dockerfile.sandbox "${MICHAEL_DIR}/Dockerfile.sandbox"
    sudo -u "${USERNAME}" podman build \
        -t michael-sandbox:alpine \
        -f "${MICHAEL_DIR}/Dockerfile.sandbox" \
        "${MICHAEL_DIR}/"
else
    echo "NOTE: Dockerfile.sandbox not found in CWD; skipping sandbox image build." >&2
fi

echo "[10/12] operator user ${OPERATOR_USER} (web-terminal account, no SSH)"
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

echo "[11/12] ttyd binary + self-signed TLS cert"
install -d -m 0755 "${TTYD_ETC}"
if [[ ! -x "${TTYD_BIN}" ]] || ! "${TTYD_BIN}" --version 2>/dev/null | grep -q "${TTYD_VERSION}"; then
    curl -fsSL -o "${TTYD_BIN}" \
        "https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.x86_64"
    chmod 0755 "${TTYD_BIN}"
fi

apt-get "${APT_CONFOLD[@]}" -y install openssl
PUBLIC_IP="$(curl -fsS --max-time 5 https://api.ipify.org || true)"
CERT_CN="${PUBLIC_IP:-michael-vps}"
if [[ ! -s "${TTYD_ETC}/cert.pem" || ! -s "${TTYD_ETC}/key.pem" || "${WEB_REGEN:-0}" == "1" ]]; then
    openssl req -x509 -nodes -newkey rsa:2048 \
        -days 3650 \
        -subj "/CN=${CERT_CN}" \
        -addext "subjectAltName=IP:${PUBLIC_IP:-127.0.0.1}" \
        -keyout "${TTYD_ETC}/key.pem" \
        -out "${TTYD_ETC}/cert.pem"
fi
chmod 0640 "${TTYD_ETC}/key.pem" "${TTYD_ETC}/cert.pem"
chgrp "${OPERATOR_USER}" "${TTYD_ETC}/key.pem" "${TTYD_ETC}/cert.pem"

# Persist credentials for the operator to retrieve.
umask 077
cat >"${WEB_CRED_FILE}" <<EOF
url=https://${PUBLIC_IP:-<vps-ip>}:${TTYD_PORT}/
username=${OPERATOR_USER}
password=${WEB_PASSWORD}
EOF
chmod 0400 "${WEB_CRED_FILE}"

echo "[12/12] ttyd systemd unit (HTTPS web terminal on :${TTYD_PORT})"
# Note: AmbientCapabilities lets the operator user bind :443 without root.
# We deliberately do NOT set NoNewPrivileges/CapabilityBoundingSet/ProtectSystem,
# because the user expects a full sudoer shell from the web terminal — those
# directives would block sudo (setuid) and writes to / outside /home.
cat >/etc/systemd/system/ttyd.service <<EOF
[Unit]
Description=ttyd web terminal (HTTPS, basic-auth) for ${OPERATOR_USER}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${OPERATOR_USER}
Group=${OPERATOR_USER}
AmbientCapabilities=CAP_NET_BIND_SERVICE
WorkingDirectory=/home/${OPERATOR_USER}
ExecStart=${TTYD_BIN} \\
    --port ${TTYD_PORT} \\
    --interface 0.0.0.0 \\
    --credential ${OPERATOR_USER}:${WEB_PASSWORD} \\
    --ssl \\
    --ssl-cert ${TTYD_ETC}/cert.pem \\
    --ssl-key ${TTYD_ETC}/key.pem \\
    --writable \\
    --max-clients 4 \\
    /bin/bash -l
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
chmod 0640 /etc/systemd/system/ttyd.service

systemctl daemon-reload
systemctl enable --now ttyd.service
sleep 1
systemctl --no-pager --full status ttyd.service || true

cat <<EOF

================================================================
  bootstrap complete.

  WEB TERMINAL (use Safari over iCloud Private Relay):
      URL:      https://${PUBLIC_IP:-<vps-ip>}:${TTYD_PORT}/
      User:     ${OPERATOR_USER}
      Password: ${WEB_PASSWORD}

  The TLS cert is self-signed. On first visit Safari will warn:
  tap "Show Details" -> "visit this website" -> confirm.
  Credentials also persisted at ${WEB_CRED_FILE} (root-only).
  To rotate them later: WEB_REGEN=1 ./bootstrap.sh

  SSH (kept available; key-only, no password, no root):
      ssh -p ${SSH_PORT} ${USERNAME}@${PUBLIC_IP:-<vps-ip>}

  BEFORE CLOSING THIS ROOT SESSION, open a NEW shell and verify
  EITHER the web terminal OR the SSH login works. Only then is
  it safe to log out as root — root login and password auth on
  SSH are disabled and cannot be recovered without console access.
================================================================
EOF
