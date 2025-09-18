# GPT Pretraining Pipeline 🚀

A professional-grade implementation of GPT model pretraining from scratch, based on Sebastian Raschka's "Build a Large Language Model (From Scratch)".

## 🏗️ Architecture

This pipeline implements a **124M parameter GPT model** with:
- 12 transformer layers
- 12 attention heads  
- 768 embedding dimensions
- 50,257 vocabulary size (GPT-2 BPE tokenizer)
- 1024 context length

## 📊 Features

- **Distributed Training Support**: Efficient GPU utilization
- **Experiment Tracking**: Integrated W&B and TensorBoard logging
- **Modular Design**: Clean separation of configs, source code, and experiments
- **Automatic Mixed Precision**: FP16 training for faster convergence
- **Gradient Accumulation**: Effective batch size scaling
- **Learning Rate Scheduling**: Warmup + cosine annealing

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda activate llms-scratch
```

### 2. Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

### 3. Train Model
```bash
# Full training with W&B logging (default)
python src/train_gpt.py --epochs 5

# Quick test run
python src/train_gpt.py --epochs 1 --no-wandb

# Custom settings
python src/train_gpt.py --epochs 10 --batch-size 16
```

## 📁 Project Structure
```
gpt_pretraining_pipeline/
├── src/                     # Source code
│   └── train_gpt.py        # Main training script (includes model config)
├── experiments/            # Experiment artifacts
│   └── archive/           # Previous runs
├── scripts/               # Utility scripts
│   ├── prepare_dataset.py # Dataset preparation
│   └── test_cuda_*.py     # CUDA debugging tools
└── README.md              # This file
```

## 🎯 Training Results

Latest training run (5 epochs on 2.8MB dataset):
- **Initial Loss**: 10.99 → **Final Loss**: ~5.0
- **Training Time**: ~2.5 minutes per epoch (RTX 3500)
- **Perplexity**: 165 (approaching coherent text generation)

## 🔧 Advanced Usage

### Custom Configuration
```python
# Override config values
python src/train_gpt.py \
    --batch-size 16 \
    --learning-rate 0.001 \
    --grad-accumulation 4
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 src/train_gpt.py --distributed
```

### Hyperparameter Tuning
```bash
python scripts/hyperparam_search.py --config configs/sweep.yaml
```

## 📈 Monitoring

- **W&B Dashboard**: Track experiments at [wandb.ai/your-username/LLMs-from-scratch](https://wandb.ai)
- **TensorBoard**: `tensorboard --logdir experiments/logs`

## 🎓 Educational Notes

This implementation prioritizes clarity and educational value over performance. Key learning points:
- Self-attention mechanism implementation
- Positional encoding strategies  
- Layer normalization placement
- Gradient flow optimization
- Training stability techniques

## 🤝 Contributing

This is an educational project following the book's progression. Contributions that enhance clarity or add educational value are welcome!

## 📚 References

- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
- Original GPT paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT-2 paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
