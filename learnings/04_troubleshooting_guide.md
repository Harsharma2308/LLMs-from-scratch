# 🔨 Troubleshooting Guide

Common issues encountered and their solutions.

## 🖥️ CUDA/GPU Issues

### Problem: Training Running on CPU Instead of GPU
**Symptoms**: 
- Training extremely slow (0.009 steps/sec instead of 3.4)
- No GPU memory usage
- High CPU usage

**Solutions**:
1. Check CUDA availability:
   ```python
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

2. Verify PyTorch CUDA version:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. If still failing, try system reboot (worked in our case!)

4. Add explicit GPU checks in code:
   ```python
   if not torch.cuda.is_available():
       print("❌ ERROR: CUDA is not available!")
       sys.exit(1)
   ```

### Problem: CUDA Initialization Error
**Error**: `RuntimeError: CUDA error: initialization error`

**Solutions**:
1. Check nvidia-smi works: `nvidia-smi`
2. Update GPU drivers
3. Reinstall PyTorch with correct CUDA version
4. System reboot often fixes initialization issues

## 📝 Text/Unicode Issues

### Problem: Garbled Text in W&B
**Symptoms**: Seeing ""'s instead of quotes

**Solution**:
```python
def fix_unicode_for_display(text):
    replacements = {
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2014': '--',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
```

## 🏋️ Weight Loading Issues

### Problem: Shape Mismatch When Loading Weights
**Error**: `RuntimeError: Error(s) in loading state_dict`

**Causes**:
1. Wrong `qkv_bias` setting (OpenAI uses True)
2. Different context length (256 vs 1024)
3. Wrong model configuration

**Solution**:
```python
# Ensure configuration matches
config = {
    "vocab_size": 50257,
    "context_length": 1024,  # Must match pretrained
    "qkv_bias": True,        # Must be True for OpenAI
    # ... rest of config
}
```

### Problem: 404 Error Downloading Weights
**Error**: `urllib.error.HTTPError: HTTP Error 404`

**Solution**: Use correct URLs
```python
# Correct HuggingFace URL
url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth"
```

## 🔄 Training Issues

### Problem: Loss Not Decreasing
**Symptoms**: Loss stays around 10-11

**Checklist**:
1. Verify data is loaded correctly
2. Check learning rate (try 5e-4)
3. Ensure optimizer is stepping
4. Verify gradients are not zero
5. Check for gradient clipping too aggressive

### Problem: Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size (try 4 instead of 8)
2. Use gradient accumulation
3. Enable mixed precision training
4. Clear cache: `torch.cuda.empty_cache()`

## 🐛 Module Import Issues

### Problem: ModuleNotFoundError
**Error**: `ModuleNotFoundError: No module named 'previous_chapters'`

**Solution**: Add proper path imports
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

ch05_path = str(project_root / "ch05" / "01_main-chapter-code")
sys.path.append(ch05_path)
```

## ⚡ Performance Issues

### Problem: Training Suddenly Slows Down
**Symptoms**: Eval time jumps from 1.4s to 12.2s

**Possible Causes**:
1. GPU thermal throttling
2. Memory fragmentation
3. Background processes

**Solutions**:
1. Monitor GPU temperature: `nvidia-smi -l 1`
2. Add periodic cache clearing
3. Reduce evaluation frequency
4. Check for memory leaks

## 📊 Logging Issues

### Problem: W&B Step Counter Wrong
**Symptoms**: Steps showing 0, 1, 2 instead of 0, 100, 200

**Solution**: Use step parameter explicitly
```python
wandb.log(metrics, step=global_step)  # Don't add step to metrics dict
```

## 🔧 Environment Issues

### Problem: Conda Environment Not Activating
**Solution**:
```bash
# Ensure conda is initialized
conda init zsh  # or bash

# Activate environment
conda activate llms-scratch
```

### Problem: Package Version Conflicts
**Solution**: Create fresh environment
```bash
conda create -n llms-scratch python=3.10
conda activate llms-scratch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 💡 Pro Tips

1. **Always Test GPU First**:
   ```python
   # Quick GPU test
   torch.cuda.is_available() and print(torch.zeros(1).cuda())
   ```

2. **Use Smaller Test Runs**:
   ```bash
   python train.py --epochs 1 --eval-interval 10
   ```

3. **Enable Debug Mode**:
   ```python
   torch.autograd.set_detect_anomaly(True)  # For gradient debugging
   ```

4. **Profile Performance**:
   ```python
   with torch.profiler.profile() as prof:
       # training code
   print(prof.key_averages().table())
   ```

## 🚨 When All Else Fails

1. **Restart kernel/terminal**
2. **Clear all caches**: `rm -rf ~/.cache/torch`
3. **Reboot system** (surprisingly effective!)
4. **Check GitHub issues** for similar problems
5. **Start with minimal example** and add complexity

---

Remember: Most issues have been encountered by others. Don't hesitate to search error messages!
