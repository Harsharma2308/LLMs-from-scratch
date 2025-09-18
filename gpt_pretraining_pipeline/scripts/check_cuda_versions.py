#!/usr/bin/env python3
import subprocess
import re

print("=== CUDA Version Check ===\n")

# Check driver version
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    driver_match = re.search(r'Driver Version: (\d+\.\d+\.\d+)', result.stdout)
    cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
    if driver_match and cuda_match:
        print(f"NVIDIA Driver: {driver_match.group(1)}")
        print(f"CUDA Runtime API version (max supported): {cuda_match.group(1)}")
except:
    pass

# Check PyTorch expectations
print("\nPyTorch expectations:")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    
    # Check if it's a version mismatch
    if torch.version.cuda:
        cuda_major = int(torch.version.cuda.split('.')[0])
        print(f"\nPyTorch expects CUDA {cuda_major}.x")
        
except Exception as e:
    print(f"Error: {e}")

print("\n=== Recommendation ===")
print("Your driver (575.64.03) is VERY recent (2025). PyTorch 2.8.0 may not be")
print("compatible with such a new driver. Try:")
print("1. Downgrade NVIDIA driver to 550.x or 560.x series")
print("2. Or install PyTorch nightly that supports newer drivers:")
print("   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128")

