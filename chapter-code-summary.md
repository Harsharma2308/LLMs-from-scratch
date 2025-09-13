# LLMs from Scratch - Chapter Code Summary

## Chapter 2: Working with Text Data
**Main Focus**: Tokenization and data preparation for LLMs

### Core Components:
- **Tokenization**: Implementation of BPE (Byte Pair Encoding) tokenizer using tiktoken
- **Text Encoding/Decoding**: Converting text to token IDs and back
- **Data Loading**: Custom PyTorch DataLoader for text data
- **Sliding Window Approach**: Creating input-target pairs for next-token prediction
- **Vocabulary Building**: Understanding token vocabularies and special tokens

### Key Files:
- `ch02.ipynb`: Complete tokenization pipeline, BPE implementation
- `dataloader.ipynb`: PyTorch DataLoader for batching text sequences
- Bonus: BPE tokenizer from scratch implementation

### Code Highlights:
```python
# Tokenization with tiktoken
# Creating input-target pairs for training
# Batch processing with custom DataLoader
```

## Chapter 3: Coding Attention Mechanisms
**Main Focus**: Building the attention mechanism from scratch

### Core Components:
- **Self-Attention**: Single-head attention implementation
- **Scaled Dot-Product Attention**: Mathematical foundation of attention
- **Causal Attention Mask**: Preventing future token attention
- **Multi-Head Attention**: Parallel attention heads
- **Positional Encoding**: Adding position information to embeddings

### Key Files:
- `ch03.ipynb`: Complete attention implementation with visualizations
- `multihead-attention.ipynb`: Efficient multi-head attention variants
- PyTorch buffers for causal masks

### Code Highlights:
```python
# Attention score computation: Q @ K^T / sqrt(d_k)
# Causal masking for autoregressive generation
# Multi-head attention with concatenation
```

## Chapter 4: Implementing a GPT Model from Scratch
**Main Focus**: Complete GPT architecture assembly

### Core Components:
- **GPT Model Class**: Full transformer architecture
- **Layer Normalization**: Pre-norm architecture
- **Feed-Forward Networks**: MLP blocks with GELU activation
- **Residual Connections**: Skip connections for gradient flow
- **Model Configuration**: Parameterizable architecture (124M, 355M, etc.)
- **Text Generation**: Sampling strategies (greedy, temperature, top-k)

### Key Files:
- `ch04.ipynb`: Complete GPT implementation
- `gpt.py`: Modular GPT class for reuse
- Bonus: FLOPS analysis and KV-cache optimization

### Model Configurations:
```python
# GPT-124M configuration:
# - 12 layers, 12 heads, 768 embedding dim
# - ~124M parameters
# - Vocabulary size: 50,257 (GPT-2 tokenizer)
```

## Chapter 5: Pretraining on Unlabeled Data
**Main Focus**: Training loop implementation and model pretraining

### Core Components:
- **Training Loop**: Complete pretraining pipeline with:
  - Loss computation (cross-entropy)
  - Gradient accumulation
  - Learning rate scheduling (cosine, warmup)
  - Gradient clipping
  - Checkpointing
- **Model Evaluation**: Perplexity and generation quality metrics
- **Weight Loading**: Loading OpenAI GPT-2 pretrained weights
- **Data Pipeline**: Efficient text data streaming

### Key Files:
- `ch05.ipynb`: Full pretraining implementation
- `gpt_train.py`: Modular training script
- `gpt_generate.py`: Text generation utilities
- `gpt_download.py`: Download and convert OpenAI weights

### Training Features:
```python
# AdamW optimizer with weight decay
# Cosine learning rate schedule with warmup
# Mixed precision training support
# Model checkpointing every N steps
# Generation sampling during training
```

### Integration Points for W&B/TensorBoard:
- Training/validation loss logging
- Learning rate tracking
- Gradient norm monitoring
- Sample generation logging
- Model checkpoint saving

## Chapter 6: Finetuning for Classification
**Main Focus**: Adapting pretrained GPT for downstream tasks

### Core Components:
- **Classification Head**: Adding task-specific layers
- **Finetuning Strategies**: 
  - Full model finetuning
  - Last layer only
  - LoRA (Low-Rank Adaptation)
- **Data Processing**: Sentiment analysis dataset preparation
- **Evaluation Metrics**: Accuracy, confusion matrices

### Key Files:
- `ch06.ipynb`: Classification finetuning pipeline
- `gpt_class_finetune.py`: Modular finetuning script
- Bonus: IMDB 50k dataset experiments

## Chapter 7: Finetuning to Follow Instructions
**Main Focus**: Instruction-following and alignment

### Core Components:
- **Instruction Dataset**: Format and preparation
- **Supervised Finetuning**: Training on instruction-response pairs
- **Evaluation**: Response quality assessment
- **Preference Learning**: DPO (Direct Preference Optimization)

### Key Files:
- `ch07.ipynb`: Instruction finetuning implementation
- `gpt_instruction_finetuning.py`: Training script
- `ollama_evaluate.py`: Model evaluation utilities
- Bonus: DPO implementation from scratch

## Key Utilities Across Chapters

### Model Architecture Constants:
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 tokenizer
    "context_length": 1024,   # Maximum sequence length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer blocks
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
```

### Common Training Parameters:
```python
# Learning rate: 5e-4 (with warmup)
# Batch size: 8-64 (depending on GPU memory)
# Weight decay: 0.1
# Gradient clipping: 1.0
# Training steps: 5000-50000
```

## Bonus Materials Highlights
- **Alternative Architectures**: Llama, Qwen, Gemma implementations
- **Optimization**: KV-cache, memory-efficient loading
- **Extended Tokenizers**: Custom vocabulary extensions
- **Performance**: Training speed optimizations
- **UI**: Gradio/Streamlit interfaces for model interaction
