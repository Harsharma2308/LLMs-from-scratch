# 🔧 Key Code Snippets & Patterns

This document contains important code snippets and patterns discovered during the project.

## 📦 Weight Loading

### 1. PyTorch State Dict (Recommended)
```python
# Download and load pretrained weights
url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth"
state_dict = torch.load("gpt2-small-124M.pth", weights_only=True)
model.load_state_dict(state_dict)
```

### 2. TensorFlow to PyTorch Conversion
```python
def assign(left, right):
    """Convert numpy array to PyTorch parameter"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, device=left.device))

# Split QKV weights
q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
```

### 3. Weight Mapping Pattern
```python
# OpenAI format → Our format
gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

# Attention weights need transposition
gpt.trf_blocks[b].att.W_query.weight = assign(
    gpt.trf_blocks[b].att.W_query.weight, q_w.T)
```

## 🏃‍♂️ Training Patterns

### 1. GPU Check Pattern
```python
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA is not available!")
    print("   Options:")
    print("   1. Fix your CUDA environment and try again")
    print("   2. Run with --force-cpu flag (not recommended)")
    sys.exit(1)
```

### 2. Efficient Loss Calculation
```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over data loader"""
    total_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if num_batches and i >= num_batches:
                break
            # ... calculate loss
    return total_loss / total_batches
```

### 3. Text Generation with Temperature
```python
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    for _ in range(max_new_tokens):
        # Get predictions
        logits = model(idx[:, -context_length:])
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, 
                               torch.tensor(float("-inf")).to(logits.device), 
                               logits)
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

## 📊 Logging Patterns

### 1. Dual Logging Setup
```python
# W&B initialization
wandb.init(
    project="LLMs-from-scratch",
    name=f"gpt-124m-{timestamp}",
    config=config,
    tags=["gpt-124m", "pretraining"]
)

# TensorBoard initialization
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/experiment_{timestamp}')
```

### 2. Unicode Fix for W&B
```python
def fix_unicode_for_display(text):
    """Fix Unicode characters for clean W&B display"""
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u2014': '--', # Em dash
        # ... more replacements
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return ''.join(char if ord(char) < 128 else '?' for char in text)
```

## 🎯 Model Configuration

### GPT-2 124M Configuration
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,    # Note: OpenAI uses 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True          # Important: OpenAI uses bias!
}
```

## 🔍 Debugging Patterns

### 1. Progress Tracking with tqdm
```python
from tqdm import tqdm

with tqdm(total=file_size, unit="iB", unit_scale=True, desc=filename) as pbar:
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        file.write(chunk)
        pbar.update(len(chunk))
```

### 2. Checkpoint Saving
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'global_step': global_step,
    'config': config,
    'train_loss': train_loss,
    'val_loss': val_loss,
}, checkpoint_path)
```

## 💡 Performance Tips

1. **Batch Size**: 8 works well for RTX 3500 (12GB VRAM)
2. **Evaluation**: Sample only 5 batches for quick validation
3. **Gradient Clipping**: Use 1.0 for stable training
4. **Learning Rate**: 5e-4 is a good starting point

## 🚨 Common Pitfalls

1. **Device Mismatch**: Always ensure tensors are on same device
2. **Context Length**: Our model uses 256, OpenAI uses 1024
3. **Bias Settings**: Must match when loading pretrained weights
4. **Weight Shapes**: Many need transposition from TF format
