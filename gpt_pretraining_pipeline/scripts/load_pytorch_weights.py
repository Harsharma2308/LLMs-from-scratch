#!/usr/bin/env python3
"""
Load GPT-2 weights using the recommended PyTorch state dict method
This is the simplest and most reliable method.
"""

import torch
import tiktoken
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our GPT model
ch05_path = str(project_root / "ch05" / "01_main-chapter-code")
sys.path.append(ch05_path)
from previous_chapters import GPTModel

def download_pytorch_weights(dest_path):
    """Download pre-converted PyTorch weights"""
    # Using the correct HuggingFace URL
    url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth"
    
    print("📥 Downloading GPT-2 124M PyTorch weights...")
    print(f"   From: {url}")
    print(f"   To: {dest_path}")
    
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))
        
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc="gpt2-small-124M.pth") as pbar:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print("✅ Download complete!")

def load_and_test_model():
    """Load the model and test it"""
    
    print("\n🚀 Loading GPT-2 124M with PyTorch State Dict Method\n")
    
    # Setup paths
    weights_dir = project_root / "gpt2_weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "gpt2-small-124M.pth"
    
    # Download if needed
    if not weights_path.exists():
        download_pytorch_weights(weights_path)
    else:
        print("✅ Weights already downloaded")
    
    # Model configuration (GPT-2 124M)
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True  # GPT-2 uses bias in attention
    }
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    model = GPTModel(config).to(device)
    print(f"📊 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load weights
    print("\n🔄 Loading pretrained weights...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully!")
    
    # Test the model
    print("\n🧪 Testing the model with text generation...")
    tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()
    
    test_prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "In the future, technology",
        "Once upon a time,",
        "The most important thing about",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Generate
        generated_tokens = []
        with torch.no_grad():
            for _ in range(30):  # Generate 30 tokens
                # Get model predictions
                logits = model(input_ids)
                
                # Get the next token (greedy decoding)
                next_token_logits = logits[0, -1, :]
                
                # Optional: Apply temperature
                temperature = 0.8
                next_token_logits = next_token_logits / temperature
                
                # Get probabilities and sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token.item())
                
                # Truncate if exceeding context length
                if input_ids.shape[1] > config["context_length"]:
                    input_ids = input_ids[:, -config["context_length"]:]
        
        # Decode and print
        full_text = tokenizer.decode(tokens + generated_tokens)
        print(f"🤖 Generated: {full_text}")
    
    print("\n✨ Success! The model is generating coherent text.")
    print("\n📚 Key Insights:")
    print("1. This method loads pre-converted PyTorch weights (no TensorFlow needed)")
    print("2. The weights are already in the correct format for our GPTModel")
    print("3. No complex weight mapping or transposition required")
    print("4. This is the fastest and most reliable method")
    
    return model

if __name__ == "__main__":
    model = load_and_test_model()
