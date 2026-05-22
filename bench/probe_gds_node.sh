#!/bin/bash
# Probe a node for GDS / InfiniBand / NIC / filesystem-mode capability.
set -u

echo "=== node: $(hostname) ==="
echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader 2>&1
echo
echo "nvidia-fs driver:"
[[ -e /proc/driver/nvidia-fs ]] && echo "  loaded (GDS direct path possible)" || echo "  NOT loaded (cuFile compat only)"
[[ -d /proc/driver/nvidia-fs ]] && ls -la /proc/driver/nvidia-fs/ 2>&1 | head -5
echo
echo "kernel modules:"
lsmod 2>/dev/null | grep -E 'nvidia_fs|nvidia_peermem|mlx5|ib_core|rdma' | head -10
echo
echo "InfiniBand / RDMA NICs:"
ibstatus 2>&1 | head -20 || echo "  ibstatus not available"
echo
echo "Network adapters:"
lspci 2>&1 | grep -iE 'mellanox|infiniband|ethernet controller' | head -10
echo
echo "Local NVMe disks:"
lsblk -d -o NAME,SIZE,MODEL,TRAN 2>&1 | grep -iE 'nvme|ssd' | head -10
echo
echo "VAST mount mode:"
mount | grep -E 'mydata|hpc|projects|scratch' | head -10
echo
echo "cuFile config (if present):"
[[ -f /etc/cufile.json ]] && grep -E 'compat_mode|posix_pool|properties|fs_compat' /etc/cufile.json | head -20 || echo "  no /etc/cufile.json"
echo
echo "=== end ==="
