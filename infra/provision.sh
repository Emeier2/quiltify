#!/usr/bin/env bash
# infra/provision.sh — one-time Azure resource provisioning for Quiltify
#
# Usage:
#   export AZURE_SUBSCRIPTION_ID=<your-subscription-id>
#   bash infra/provision.sh
#
# Prerequisites:
#   az login        (Azure CLI, https://aka.ms/installazurecli)
#   ssh-keygen -t ed25519 -f ~/.ssh/quiltify_azure  (or reuse an existing key)
#
# What it creates:
#   - Resource group
#   - Azure Container Registry (Basic SKU)
#   - Ubuntu 22.04 VM with NVIDIA T4 GPU (Standard_NC4as_T4_v3)
#   - NVIDIA GPU driver + Docker + NVIDIA Container Toolkit on the VM
#   - NSG rule to open port 80 (HTTP)
set -euo pipefail

# ── Configuration — edit these before running ─────────────────────────────
RESOURCE_GROUP="${RESOURCE_GROUP:-quiltify-rg}"
LOCATION="${LOCATION:-eastus}"          # pick a region that has NC4as_T4_v3
ACR_NAME="${ACR_NAME:-quiltifyacr}"     # must be globally unique, lowercase
VM_NAME="${VM_NAME:-quiltify-vm}"
VM_SIZE="${VM_SIZE:-Standard_NC4as_T4_v3}"  # 4 vCPU, 28 GB RAM, 1× T4 (16 GB VRAM)
ADMIN_USER="${ADMIN_USER:-azureuser}"
SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-$(cat ~/.ssh/quiltify_azure.pub 2>/dev/null || cat ~/.ssh/id_ed25519.pub)}"
# ──────────────────────────────────────────────────────────────────────────

echo "==> Setting subscription"
az account set --subscription "${AZURE_SUBSCRIPTION_ID}"

echo "==> Creating resource group: ${RESOURCE_GROUP}"
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output none

echo "==> Creating Azure Container Registry: ${ACR_NAME}"
az acr create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --sku Basic \
  --admin-enabled true \
  --output none

ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query passwords[0].value -o tsv)

echo "   ACR login server : ${ACR_LOGIN_SERVER}"
echo "   ACR username     : ${ACR_USERNAME}"

echo "==> Creating VM: ${VM_NAME} (${VM_SIZE})"
az vm create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VM_NAME}" \
  --size "${VM_SIZE}" \
  --image Ubuntu2204 \
  --admin-username "${ADMIN_USER}" \
  --ssh-key-values "${SSH_PUBLIC_KEY}" \
  --public-ip-sku Standard \
  --output none

VM_IP=$(az vm show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VM_NAME}" \
  --show-details \
  --query publicIps -o tsv)
echo "   VM public IP: ${VM_IP}"

echo "==> Opening port 80"
az vm open-port \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VM_NAME}" \
  --port 80 \
  --output none

echo "==> Installing NVIDIA drivers, Docker, and NVIDIA Container Toolkit on VM"
az vm extension set \
  --resource-group "${RESOURCE_GROUP}" \
  --vm-name "${VM_NAME}" \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings "$(cat <<'JSON'
{
  "commandToExecute": "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y curl && curl -fsSL https://raw.githubusercontent.com/Emeier2/quiltify/master/infra/vm_setup.sh | bash"
}
JSON
)" \
  --output none

echo ""
echo "==================================================================="
echo " Provisioning complete!"
echo "==================================================================="
echo ""
echo "  VM IP address : ${VM_IP}"
echo "  SSH           : ssh ${ADMIN_USER}@${VM_IP} -i ~/.ssh/quiltify_azure"
echo "  App URL       : http://${VM_IP}"
echo ""
echo "  Add these secrets to your GitHub repository:"
echo "    VM_IP       = ${VM_IP}"
echo "    VM_SSH_KEY  = (contents of ~/.ssh/quiltify_azure)"
echo "    ACR_USERNAME= ${ACR_USERNAME}"
echo "    ACR_PASSWORD= ${ACR_PASSWORD}"
echo ""
echo "  Add this variable to your GitHub repository:"
echo "    ACR_NAME    = ${ACR_NAME}"
echo ""
echo "  Then set ALLOWED_ORIGINS in ~/quiltify/.env on the VM:"
echo "    ALLOWED_ORIGINS=http://${VM_IP}"
echo "==================================================================="
