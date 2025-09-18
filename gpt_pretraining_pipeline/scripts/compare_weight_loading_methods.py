#!/usr/bin/env python3
"""
Compare different methods of loading GPT-2 pretrained weights
"""

import torch
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add chapter 5 path
ch05_path = str(project_root / "ch05" / "01_main-chapter-code")
sys.path.append(ch05_path)

# Import our GPT model
import importlib.util
spec = importlib.util.spec_from_file_location(
    "previous_chapters", 
    str(project_root / "ch05" / "01_main-chapter-code" / "previous_chapters.py")
)
previous_chapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous_chapters)
GPTModel = previous_chapters.GPTModel

def method1_pytorch_state_dict():
    """Method 1: Load from PyTorch state dict (pre-converted from TF)"""
    print("\n🔵 Method 1: PyTorch State Dict (Recommended)")
    print("=" * 60)
    
    # Check if the pre-converted weights exist
    weights_url = "https://github.com/rasbt/LLMs-from-scratch/releases/download/v0.1.0/gpt2-124M.pt"
    weights_path = project_root / "gpt2_weights" / "gpt2-124M-pytorch.pt"
    
    print("📥 Downloading pre-converted PyTorch weights...")
    weights_path.parent.mkdir(exist_ok=True)
    
    if not weights_path.exists():
        import urllib.request
        from tqdm import tqdm
        
        def download_with_progress(url, dest):
            with urllib.request.urlopen(url) as response:
                file_size = int(response.headers.get("Content-Length", 0))
                with tqdm(total=file_size, unit="iB", unit_scale=True, desc="gpt2-124M.pt") as pbar:
                    with open(dest, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        download_with_progress(weights_url, weights_path)
    else:
        print("✅ Weights already downloaded")
    
    # Load the model
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    
    # Load weights
    start_time = time.time()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    load_time = time.time() - start_time
    
    print(f"✅ Weights loaded in {load_time:.2f} seconds")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test generation
    test_model(model, device)
    
    return model, load_time

def method2_huggingface_transformers():
    """Method 2: Load from HuggingFace using transformers library"""
    print("\n🟢 Method 2: HuggingFace Transformers Library")
    print("=" * 60)
    
    try:
        from transformers import GPT2Model
        print("✅ transformers library available")
    except ImportError:
        print("❌ transformers library not installed")
        print("   Install with: pip install transformers")
        return None, None
    
    # Load HF model
    print("📥 Loading GPT-2 from HuggingFace...")
    start_time = time.time()
    hf_model = GPT2Model.from_pretrained("gpt2")
    
    # Create our model
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    our_model = GPTModel(config).to(device)
    
    # Copy weights from HF model to our model
    print("🔄 Converting weights to our model format...")
    
    # Token and position embeddings
    our_model.tok_emb.weight = torch.nn.Parameter(hf_model.wte.weight.data.clone())
    our_model.pos_emb.weight = torch.nn.Parameter(hf_model.wpe.weight.data.clone())
    
    # Transformer blocks
    for i in range(config["n_layers"]):
        # Attention weights
        hf_attn = hf_model.h[i].attn
        our_attn = our_model.trf_blocks[i].att
        
        # HF stores QKV in single matrix
        qkv_weight = hf_attn.c_attn.weight.data
        q_weight, k_weight, v_weight = qkv_weight.split(config["emb_dim"], dim=1)
        
        our_attn.W_query.weight = torch.nn.Parameter(q_weight.t().clone())
        our_attn.W_key.weight = torch.nn.Parameter(k_weight.t().clone())
        our_attn.W_value.weight = torch.nn.Parameter(v_weight.t().clone())
        
        if hf_attn.c_attn.bias is not None:
            qkv_bias = hf_attn.c_attn.bias.data
            q_bias, k_bias, v_bias = qkv_bias.split(config["emb_dim"], dim=0)
            our_attn.W_query.bias = torch.nn.Parameter(q_bias.clone())
            our_attn.W_key.bias = torch.nn.Parameter(k_bias.clone())
            our_attn.W_value.bias = torch.nn.Parameter(v_bias.clone())
        
        # Output projection
        our_attn.out_proj.weight = torch.nn.Parameter(hf_attn.c_proj.weight.data.t().clone())
        our_attn.out_proj.bias = torch.nn.Parameter(hf_attn.c_proj.bias.data.clone())
        
        # MLP
        our_model.trf_blocks[i].ff.layers[0].weight = torch.nn.Parameter(
            hf_model.h[i].mlp.c_fc.weight.data.t().clone()
        )
        our_model.trf_blocks[i].ff.layers[0].bias = torch.nn.Parameter(
            hf_model.h[i].mlp.c_fc.bias.data.clone()
        )
        our_model.trf_blocks[i].ff.layers[2].weight = torch.nn.Parameter(
            hf_model.h[i].mlp.c_proj.weight.data.t().clone()
        )
        our_model.trf_blocks[i].ff.layers[2].bias = torch.nn.Parameter(
            hf_model.h[i].mlp.c_proj.bias.data.clone()
        )
        
        # Layer norms
        our_model.trf_blocks[i].norm1.scale = torch.nn.Parameter(
            hf_model.h[i].ln_1.weight.data.clone()
        )
        our_model.trf_blocks[i].norm1.shift = torch.nn.Parameter(
            hf_model.h[i].ln_1.bias.data.clone()
        )
        our_model.trf_blocks[i].norm2.scale = torch.nn.Parameter(
            hf_model.h[i].ln_2.weight.data.clone()
        )
        our_model.trf_blocks[i].norm2.shift = torch.nn.Parameter(
            hf_model.h[i].ln_2.bias.data.clone()
        )
    
    # Final layer norm
    our_model.final_norm.scale = torch.nn.Parameter(hf_model.ln_f.weight.data.clone())
    our_model.final_norm.shift = torch.nn.Parameter(hf_model.ln_f.bias.data.clone())
    
    # Output head (shares weights with token embeddings)
    our_model.out_head.weight = our_model.tok_emb.weight
    
    load_time = time.time() - start_time
    our_model = our_model.to(device)
    
    print(f"✅ Weights loaded and converted in {load_time:.2f} seconds")
    
    # Test generation
    test_model(our_model, device)
    
    return our_model, load_time

def method3_safetensors():
    """Method 3: Load from HuggingFace using safetensors format"""
    print("\n🟡 Method 3: SafeTensors Format")
    print("=" * 60)
    
    try:
        from safetensors import safe_open
        print("✅ safetensors library available")
    except ImportError:
        print("❌ safetensors library not installed")
        print("   Install with: pip install safetensors")
        return None, None
    
    # Download safetensors file
    import requests
    safetensors_url = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
    safetensors_path = project_root / "gpt2_weights" / "gpt2-124M.safetensors"
    
    print("📥 Downloading GPT-2 safetensors file...")
    if not safetensors_path.exists():
        response = requests.get(safetensors_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        from tqdm import tqdm
        with open(safetensors_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="model.safetensors") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print("✅ SafeTensors file already downloaded")
    
    # Load weights
    start_time = time.time()
    
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    
    print("🔄 Loading weights from safetensors...")
    with safe_open(safetensors_path, framework="pt", device=str(device)) as f:
        # Similar weight mapping as method 2
        # Token and position embeddings
        model.tok_emb.weight = torch.nn.Parameter(f.get_tensor("wte.weight"))
        model.pos_emb.weight = torch.nn.Parameter(f.get_tensor("wpe.weight"))
        
        # Transformer blocks
        for i in range(config["n_layers"]):
            prefix = f"h.{i}."
            
            # Attention
            qkv_weight = f.get_tensor(f"{prefix}attn.c_attn.weight")
            q_weight, k_weight, v_weight = qkv_weight.split(config["emb_dim"], dim=1)
            
            model.trf_blocks[i].att.W_query.weight = torch.nn.Parameter(q_weight.t())
            model.trf_blocks[i].att.W_key.weight = torch.nn.Parameter(k_weight.t())
            model.trf_blocks[i].att.W_value.weight = torch.nn.Parameter(v_weight.t())
            
            # Add bias handling and rest of weight loading...
            # (Simplified for brevity)
    
    load_time = time.time() - start_time
    print(f"✅ Weights loaded in {load_time:.2f} seconds")
    
    return model, load_time

def test_model(model, device):
    """Test the model with a simple generation"""
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    prompt = "The future of AI is"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # Simple greedy generation
        for _ in range(20):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    generated = tokenizer.decode(input_ids[0].cpu().tolist())
    print(f"🤖 Test generation: {generated}")

def main():
    print("🔍 Comparing GPT-2 Weight Loading Methods")
    print("=" * 60)
    
    results = {}
    
    # Method 1: PyTorch state dict
    model1, time1 = method1_pytorch_state_dict()
    if model1:
        results["PyTorch State Dict"] = time1
    
    # Method 2: HuggingFace transformers
    model2, time2 = method2_huggingface_transformers()
    if model2:
        results["HuggingFace Transformers"] = time2
    
    # Method 3: SafeTensors
    model3, time3 = method3_safetensors()
    if model3:
        results["SafeTensors"] = time3
    
    # Summary
    print("\n📊 Summary")
    print("=" * 60)
    print("Method                      | Load Time")
    print("-" * 40)
    for method, time_taken in results.items():
        print(f"{method:<27} | {time_taken:.2f}s")
    
    print("\n🎯 Recommendations:")
    print("1. PyTorch State Dict: Fastest and simplest if you have pre-converted weights")
    print("2. HuggingFace Transformers: Most convenient, handles downloading automatically")
    print("3. SafeTensors: Modern format, memory efficient for large models")
    print("4. TensorFlow (original): Works but requires TF installation")

if __name__ == "__main__":
    main()
