"""
Training script for medium-sized dataset with comprehensive logging
"""

import torch
import tiktoken
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import os
import sys
from datetime import datetime
import argparse

# Add ch05 to path
sys.path.append('ch05/01_main-chapter-code')
from previous_chapters import GPTModel, create_dataloader_v1

# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-4,
    "num_epochs": 1,
    "eval_interval": 100,
    "save_interval": 1000,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over data loader"""
    total_loss = 0.
    total_batches = 0
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if num_batches and i >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else float('inf')

def generate_sample(model, tokenizer, device, start_text="The", max_tokens=50):
    """Generate sample text from the model"""
    model.eval()
    
    tokens = tokenizer.encode(start_text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / 0.8
            
            # Sample from distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
            
            # Truncate if exceeding context length
            if input_ids.shape[1] > GPT_CONFIG_124M["context_length"]:
                input_ids = input_ids[:, -GPT_CONFIG_124M["context_length"]:]
    
    return tokenizer.decode(input_ids[0].cpu().tolist())

def setup_wandb(config, training_config):
    """Initialize Weights & Biases with project configuration"""
    wandb.init(
        project="LLMs-from-scratch",
        name=f"gpt-124m-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **config,
            **training_config,
            "dataset": "5-classic-books",
            "model": "GPT-124M",
            "total_params": sum(p.numel() for p in GPTModel(config).parameters())
        },
        tags=["gpt-124m", "pretraining", "chapter5"]
    )
    print("  ✅ W&B logging enabled")

def setup_tensorboard(timestamp):
    """Initialize TensorBoard writer"""
    writer = SummaryWriter(f'runs/medium_dataset_{timestamp}')
    print(f"  ✅ TensorBoard logging enabled → runs/medium_dataset_{timestamp}")
    return writer

def log_metrics(writer, use_wandb, metrics, step, generated_text=None):
    """Log metrics to both TensorBoard and W&B"""
    # TensorBoard logging
    if writer:
        for key, value in metrics.items():
            if value is not None and isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
    
    # W&B logging
    if use_wandb:
        wandb_metrics = {}
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)):
                # Convert metric names for W&B (replace / with _)
                wandb_metrics[k.replace('/', '_')] = v
        
        # Add generated text if provided
        if generated_text is not None:
            # Use wandb.Text for proper text logging
            wandb_metrics['generated_text'] = wandb.Text(generated_text[:200])
        
        # Use W&B's step parameter instead of adding to metrics
        wandb.log(wandb_metrics, step=step)

def train_model(train_file, val_file, config, training_config, use_wandb=True, use_tensorboard=True):
    """Main training function with W&B (default) and TensorBoard logging
    
    Why both? 
    - TensorBoard: Great for local real-time monitoring during training
    - W&B: Better for experiment tracking, sharing results, and cloud storage
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize logging systems
    print("\n📊 Logging configuration:")
    if use_wandb:
        setup_wandb(config, training_config)
    else:
        print("  ❌ W&B logging disabled")
    
    # Load data
    print("\nLoading training data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_file, 'r', encoding='utf-8') as f:
        val_text = f.read()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create data loaders
    train_loader = create_dataloader_v1(
        train_text,
        batch_size=training_config["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True
    )
    
    val_loader = create_dataloader_v1(
        val_text,
        batch_size=training_config["batch_size"],
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = None
    if use_tensorboard:
        writer = setup_tensorboard(timestamp)
    else:
        print("  ❌ TensorBoard logging disabled")
    
    # Calculate initial losses
    print("\nCalculating initial losses...")
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Initial train loss: {train_loss:.4f}")
    print(f"Initial val loss: {val_loss:.4f}")
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    start_time = time.time()
    
    for epoch in range(training_config["num_epochs"]):
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if training_config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config["grad_clip"]
                )
            
            optimizer.step()
            
            # Log batch metrics
            log_metrics(writer, use_wandb, {
                'Loss/train_batch': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }, global_step)
            
            # Evaluation
            if global_step % training_config["eval_interval"] == 0:
                model.eval()
                
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
                
                # Generate sample
                sample_text = generate_sample(model, tokenizer, device)
                
                # Log evaluation metrics and generated text
                metrics = {
                    'Loss/train': train_loss,
                    'Loss/validation': val_loss,
                    'perplexity/train': torch.exp(torch.tensor(train_loss)).item(),
                    'perplexity/val': torch.exp(torch.tensor(val_loss)).item(),
                }
                log_metrics(writer, use_wandb, metrics, global_step, generated_text=sample_text)
                
                elapsed = time.time() - start_time
                print(f"Step {global_step} | Train loss: {train_loss:.4f} | "
                      f"Val loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
                print(f"Sample: {sample_text[:100]}...")
                print("-" * 80)
                
                model.train()
            
            # Save checkpoint
            if global_step % training_config["save_interval"] == 0 and global_step > 0:
                checkpoint_path = f"model_checkpoints/model_step_{global_step}.pt"
                os.makedirs("model_checkpoints", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': config,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            global_step += 1
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    final_train_loss = calc_loss_loader(train_loader, model, device)
    final_val_loss = calc_loss_loader(val_loader, model, device)
    
    print(f"\nTraining completed!")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Save final model
    final_path = "model_checkpoints/model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")
    
    if writer:
        writer.close()
    
    # Log final metrics
    if use_wandb:
        wandb.summary['final_train_loss'] = final_train_loss
        wandb.summary['final_val_loss'] = final_val_loss
        wandb.summary['final_perplexity_train'] = torch.exp(torch.tensor(final_train_loss)).item()
        wandb.summary['final_perplexity_val'] = torch.exp(torch.tensor(final_val_loss)).item()
        wandb.finish()
    
    return model, final_train_loss, final_val_loss

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GPT model on medium dataset')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    args = parser.parse_args()
    
    # Update config with command line args
    TRAINING_CONFIG['num_epochs'] = args.epochs
    TRAINING_CONFIG['batch_size'] = args.batch_size
    
    # Train on the medium dataset
    model, train_loss, val_loss = train_model(
        "training_data/train.txt",
        "training_data/val.txt",
        GPT_CONFIG_124M,
        TRAINING_CONFIG,
        use_wandb=not args.no_wandb,  # W&B enabled by default
        use_tensorboard=not args.no_tensorboard  # TensorBoard enabled by default
    )
