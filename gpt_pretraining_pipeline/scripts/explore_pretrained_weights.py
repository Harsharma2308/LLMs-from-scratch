#!/usr/bin/env python3
"""
Explore OpenAI's pretrained GPT-2 weights and compare with our trained model
"""

import torch
import tiktoken
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add chapter 5 path for imports
ch05_path = str(project_root / "ch05" / "01_main-chapter-code")
sys.path.append(ch05_path)

# Import GPT model and utilities
import importlib.util
spec = importlib.util.spec_from_file_location(
    "previous_chapters", 
    str(project_root / "ch05" / "01_main-chapter-code" / "previous_chapters.py")
)
previous_chapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous_chapters)
GPTModel = previous_chapters.GPTModel

# Import download and loading utilities - we'll inline the functions we need
# to avoid the circular import issues
from gpt_download import download_and_load_gpt2

# Define the functions we need from gpt_generate
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # Ensure tensor is on the same device as the target parameter
    return torch.nn.Parameter(torch.tensor(right, device=left.device))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def compare_models():
    """Compare OpenAI's pretrained model with our trained model"""
    
    print("🔍 Exploring Pretrained GPT-2 Weights\n")
    
    # 1. Download OpenAI weights
    print("1️⃣ Downloading OpenAI GPT-2 124M weights...")
    settings, params = download_and_load_gpt2(
        model_size="124M", 
        models_dir=str(project_root / "gpt2_weights")
    )
    
    print("\n📊 OpenAI Model Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # 2. Initialize GPT model with OpenAI configuration
    print("\n2️⃣ Initializing GPT model with OpenAI configuration...")
    
    # Convert OpenAI settings to our config format
    config = {
        "vocab_size": settings["n_vocab"],
        "context_length": settings["n_ctx"],
        "emb_dim": settings["n_embd"],
        "n_heads": settings["n_head"],
        "n_layers": settings["n_layer"],
        "drop_rate": 0.1,  # Not in OpenAI settings
        "qkv_bias": True   # OpenAI uses bias in attention
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and load OpenAI weights
    openai_model = GPTModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in openai_model.parameters()):,}")
    
    # Load the weights
    print("\n3️⃣ Loading OpenAI weights into model...")
    load_weights_into_gpt(openai_model, params)
    print("✅ OpenAI weights loaded successfully!")
    
    # 3. Generate text with OpenAI model
    print("\n4️⃣ Generating text with OpenAI pretrained model...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    test_prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "Once upon a time, there was a",
        "The future of technology",
    ]
    
    openai_model.eval()
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Generate with OpenAI model
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = generate(
                openai_model, 
                input_ids, 
                max_new_tokens=50,
                context_size=config["context_length"],
                temperature=0.8,
                top_k=50
            )
        
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"🤖 OpenAI Model: {generated_text}")
    
    # 4. Load our trained model for comparison (if it exists)
    our_checkpoint = project_root / "gpt_pretraining_pipeline" / "model_checkpoints" / "model_step_1000.pt"
    if our_checkpoint.exists():
        print("\n\n5️⃣ Loading our trained model for comparison...")
        
        # Our model config (matching what we trained)
        our_config = {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False  # We didn't use bias
        }
        
        our_model = GPTModel(our_config).to(device)
        checkpoint = torch.load(our_checkpoint, map_location=device)
        our_model.load_state_dict(checkpoint['model_state_dict'])
        our_model.eval()
        
        print(f"✅ Loaded checkpoint from step {checkpoint['global_step']}")
        print(f"   Training loss: {checkpoint['train_loss']:.4f}")
        print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
        
        # Generate with our model
        print("\n📊 Comparing generations:")
        for prompt in test_prompts[:2]:  # Just compare first two
            print(f"\n📝 Prompt: '{prompt}'")
            
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            
            with torch.no_grad():
                generated = generate(
                    our_model,
                    input_ids,
                    max_new_tokens=50,
                    context_size=our_config["context_length"],
                    temperature=0.8,
                    top_k=50
                )
            
            generated_text = tokenizer.decode(generated[0].cpu().tolist())
            print(f"🔧 Our Model: {generated_text}")
    else:
        print(f"\n⚠️  No trained model checkpoint found at {our_checkpoint}")
    
    # 5. Analyze weight differences
    print("\n\n6️⃣ Analyzing weight conversion process:")
    print("\n🔄 Key transformations in load_weights_into_gpt:")
    print("  • Token embeddings: params['wte'] → tok_emb.weight")
    print("  • Position embeddings: params['wpe'] → pos_emb.weight")
    print("  • Attention weights: Split c_attn into separate Q, K, V matrices")
    print("  • Weight transposition: OpenAI uses different dimension ordering")
    print("  • Layer normalization: Different parameter names (g/b vs scale/shift)")
    print("  • Output projection: Shares weights with token embeddings")
    
    # Show example weight shapes
    print("\n📐 Example weight shapes:")
    print(f"  Token embeddings: {params['wte'].shape}")
    print(f"  Position embeddings: {params['wpe'].shape}")
    if 'blocks' in params and len(params['blocks']) > 0:
        print(f"  Attention (combined QKV): {params['blocks'][0]['attn']['c_attn']['w'].shape}")
        print(f"  Feed-forward (first layer): {params['blocks'][0]['mlp']['c_fc']['w'].shape}")
    
    print("\n✨ Done! Key insights:")
    print("  1. OpenAI's model uses combined QKV weights that need splitting")
    print("  2. Weights need transposition due to TF vs PyTorch conventions")
    print("  3. OpenAI includes bias in attention (we typically don't)")
    print("  4. The pretrained model generates much more coherent text!")

if __name__ == "__main__":
    compare_models()
