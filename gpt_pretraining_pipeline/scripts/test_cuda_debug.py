#!/usr/bin/env python3
"""Debug CUDA initialization issue in Cursor terminal"""

import os
import sys
import subprocess

print("=== CUDA Debug Information ===\n")

# 1. Check environment variables
print("1. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   LD_LIBRARY_PATH contains Cursor paths: {'/tmp/.mount_cursor' in os.environ.get('LD_LIBRARY_PATH', '')}")

# 2. Check nvidia-smi
print("\n2. nvidia-smi check:")
try:
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✓ GPU found: {result.stdout.strip()}")
    else:
        print(f"   ✗ nvidia-smi failed: {result.stderr}")
except Exception as e:
    print(f"   ✗ nvidia-smi error: {e}")

# 3. Try importing torch
print("\n3. PyTorch CUDA check:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA compiled version: {torch.version.cuda}")
    
    # Try to get more detailed error
    try:
        torch.cuda.init()
        print(f"   ✓ CUDA initialized successfully")
    except RuntimeError as e:
        print(f"   ✗ CUDA init error: {e}")
    
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
except Exception as e:
    print(f"   ✗ PyTorch import error: {e}")

print("\n4. Recommendations:")
print("   - This appears to be a Cursor IDE terminal environment issue")
print("   - The GPU is visible to nvidia-smi but PyTorch can't initialize CUDA")
print("   - Solution: Run your training script in an external terminal:")
print("     $ cd /home/hasharma/Workspace/Personal/LLMs-from-scratch")
print("     $ conda activate llms-scratch")
print("     $ python train_medium_dataset.py --epochs 5")

