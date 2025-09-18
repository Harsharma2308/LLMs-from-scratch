#!/usr/bin/env python3
"""Final CUDA debugging - check for runtime/driver mismatch"""
import os
import subprocess

print("=== CUDA Debugging Summary ===\n")

# 1. Check nvidia-smi reported versions
print("1. nvidia-smi versions:")
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Driver Version' in line:
        print(f"   {line.strip()}")
        
# 2. Check actual driver module version
print("\n2. Kernel module version:")
result = subprocess.run(['modinfo', 'nvidia'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if line.startswith('version:'):
        print(f"   Kernel module {line}")

# 3. Your specific situation
print("\n3. Analysis:")
print("   - Driver: 575.64.03 (from 2025)")
print("   - CUDA Runtime: 12.8")
print("   - PyTorch: Built for CUDA 12.8")
print("   - Error: cudaErrorUnknown (999)")
print("\n4. This appears to be a known issue with NVIDIA driver 575.x")
print("   The driver is too new and has compatibility issues.")
print("\n5. Solutions:")
print("   a) Downgrade NVIDIA driver to 550.x or 560.x:")
print("      sudo apt remove --purge nvidia-driver-575")
print("      sudo apt install nvidia-driver-560")
print("      sudo reboot")
print("\n   b) Or try the CUDA compatibility package:")
print("      sudo apt install cuda-compat-12-8")
print("\n   c) Or wait for PyTorch to catch up with driver 575.x")

# Try one workaround
print("\n6. Trying LD_PRELOAD workaround...")
try:
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libcuda.so.1'
    import torch
    print(f"   With LD_PRELOAD: CUDA available = {torch.cuda.is_available()}")
except Exception as e:
    print(f"   Workaround failed: {e}")

