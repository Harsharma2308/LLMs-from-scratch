#!/usr/bin/env python3
"""Test raw CUDA without PyTorch"""
import ctypes
import os

try:
    # Try to load the CUDA runtime library directly
    cuda = ctypes.CDLL("libcudart.so")
    
    # Get CUDA version
    version = ctypes.c_int()
    result = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
    
    if result == 0:
        major = version.value // 1000
        minor = (version.value % 1000) // 10
        print(f"✓ CUDA Runtime loaded successfully: {major}.{minor}")
        
        # Try to get device count
        device_count = ctypes.c_int()
        result = cuda.cudaGetDeviceCount(ctypes.byref(device_count))
        if result == 0:
            print(f"✓ CUDA devices found: {device_count.value}")
        else:
            print(f"✗ cudaGetDeviceCount failed with error: {result}")
    else:
        print(f"✗ cudaRuntimeGetVersion failed with error: {result}")
        
except Exception as e:
    print(f"✗ Failed to load CUDA runtime: {e}")

# Also test with nvidia-ml-py if available
try:
    import subprocess
    result = subprocess.run(['python', '-m', 'pip', 'show', 'nvidia-ml-py'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("\nInstalling nvidia-ml-py for additional testing...")
        subprocess.run(['pip', 'install', 'nvidia-ml-py'], check=True)
    
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"\n✓ nvidia-ml-py reports {device_count} GPU(s)")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        print(f"  GPU {i}: {name}")
except Exception as e:
    print(f"\nnvidia-ml-py test failed: {e}")
