#!/usr/bin/env bash
# infra/vm_setup.sh — bootstraps a fresh Ubuntu 22.04 Azure VM
# Called once by provision.sh via the Azure Custom Script Extension.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "==> Updating system packages"
apt-get update -qq
apt-get upgrade -y -qq

# ── NVIDIA driver ────────────────────────────────────────────────────────────
echo "==> Installing NVIDIA driver"
apt-get install -y -qq ubuntu-drivers-common
ubuntu-drivers install --gpgpu

# ── Docker Engine ────────────────────────────────────────────────────────────
echo "==> Installing Docker"
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
usermod -aG docker azureuser

# ── NVIDIA Container Toolkit ─────────────────────────────────────────────────
echo "==> Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ── Working directory ────────────────────────────────────────────────────────
echo "==> Creating ~/quiltify working directory"
sudo -u azureuser mkdir -p /home/azureuser/quiltify

echo "==> VM setup complete. Reboot may be required for NVIDIA driver to load."
