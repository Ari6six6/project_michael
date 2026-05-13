#!/usr/bin/env bash
# bootstrap.sh — idempotent root-bootstrap for a fresh Ubuntu 24.04 LTS VPS.
# Provisions the host that runs `michael` (the air-gapped AI control loop CLI).
# Run as root once, then verify SSH login as the new user in a separate shell
# before closing the current session.

set -euo pipefail

USERNAME="${USERNAME:-michael}"
SSH_PORT="${SSH_PORT:-22}"
TIMEZONE="${TIMEZONE:-UTC}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: bootstrap.sh must be run as root." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

APT_CONFOLD=(-o "Dpkg::Options::=--force-confold")

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/11] apt update + upgrade and install base packages"
apt-get update -y
apt-get "${APT_CONFOLD[@]}" -y upgrade
apt-get "${APT_CONFOLD[@]}" -y install \
    ufw fail2ban unattended-upgrades needrestart chrony apparmor-utils \
    ca-certificates curl gnupg lsb-release git jq podman uidmap slirp4netns \
    patch python3-venv python3-pip tmux htop

echo "[2/11] timezone, chrony, locale"
timedatectl set-timezone "${TIMEZONE}"
systemctl enable --now chrony
if ! locale -a | grep -qiE '^en_US\.utf-?8$'; then
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen || true
    echo "en_US.UTF-8 UTF-8" >>/etc/locale.gen
    locale-gen en_US.UTF-8
fi
update-locale LANG=en_US.UTF-8

echo "[3/11] unattended-upgrades configuration"
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
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "04:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF
systemctl enable --now unattended-upgrades.service

echo "[4/11] non-root user ${USERNAME} + sudoers"
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

echo "[5/11] UFW firewall"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow "${SSH_PORT}/tcp"
ufw --force enable
ufw status verbose

echo "[6/11] SSH hardening"
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

echo "[7/11] fail2ban"
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

echo "[8/11] sysctl + needrestart + apparmor"
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

echo "[9/11] michael state dir + sandbox image"
MICHAEL_DIR="${USER_HOME}/.michael"
install -d -m 0700 -o "${USERNAME}" -g "${USERNAME}" "${MICHAEL_DIR}"

# Enable lingering so the systemd user instance starts on boot.
loginctl enable-linger "${USERNAME}"

# Rootless podman needs cgroupfs when there is no live dbus/systemd user session
# (e.g. during bootstrap where we're running via sudo rather than a real login).
CONTAINERS_CONF_DIR="${USER_HOME}/.config/containers"
install -d -m 0755 -o "${USERNAME}" -g "${USERNAME}" "${CONTAINERS_CONF_DIR}"
cat >"${CONTAINERS_CONF_DIR}/containers.conf" <<'EOF'
[engine]
cgroup_manager = "cgroupfs"
EOF
chown "${USERNAME}:${USERNAME}" "${CONTAINERS_CONF_DIR}/containers.conf"

if [[ -f "${PROJECT_DIR}/Dockerfile.sandbox" ]]; then
    install -m 0644 -o "${USERNAME}" -g "${USERNAME}" \
        "${PROJECT_DIR}/Dockerfile.sandbox" "${MICHAEL_DIR}/Dockerfile.sandbox"
    (cd "${MICHAEL_DIR}" && sudo -u "${USERNAME}" env HOME="${USER_HOME}" podman build \
        --cgroup-manager=cgroupfs \
        -t michael-sandbox:alpine \
        -f Dockerfile.sandbox \
        .)
else
    echo "NOTE: Dockerfile.sandbox not found in ${PROJECT_DIR}; skipping sandbox image build." >&2
fi

echo "[10/11] michael CLI venv + /usr/local/bin/michael wrapper"
# Ubuntu 24.04 system Python is PEP 668-protected, so install into a venv
# alongside the checkout. The wrapper makes `michael` runnable from anywhere
# (useful when the user SSHes in and wants to manage projects from the VPS).
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

echo "[11/11] workspace directory"
# Files written by remote Michael clients (over SSH) land here. Kept distinct
# from ~/.michael so the user can wipe state without losing project files.
install -d -m 0755 -o "${USERNAME}" -g "${USERNAME}" "${USER_HOME}/workspace"

cat <<EOF

================================================================
  bootstrap complete.

  BEFORE CLOSING THIS ROOT SESSION, open a NEW terminal and run:
      ssh -p ${SSH_PORT} ${USERNAME}@<this-host>

  Confirm the login works. Only then is it safe to log out as
  root — PasswordAuthentication and root login have been
  disabled and cannot be recovered without console access.

  After your phone has run bootstrap_termux.sh, append its pubkey to:
      ${USER_HOME}/.ssh/authorized_keys
  (chown ${USERNAME}:${USERNAME}, chmod 600).
================================================================
EOF
