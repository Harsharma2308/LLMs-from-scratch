# 📚 LLMs from Scratch - Learnings Hub

Welcome to the centralized learnings repository! This folder contains all the insights, guides, and lessons learned while building Large Language Models from scratch.

## 📖 Table of Contents

### 1. [Training Pipeline Learnings](01_training_pipeline_learnings.md)
**Key Topics**: GPU optimization, loss curves, perplexity understanding, debugging CUDA issues
- Training results and performance metrics
- Understanding perplexity as uncertainty metric
- GPU vs CPU performance comparison
- Common issues and solutions
- Best practices for training

### 2. [Pretrained Weights Guide](02_pretrained_weights_guide.md)
**Key Topics**: Loading OpenAI GPT-2 weights, weight conversion, model comparison
- Multiple methods for loading pretrained weights
- Weight transformation process (TF → PyTorch)
- Comparing your model vs pretrained models
- Recommended approaches

### 3. [Key Code Snippets](03_key_code_snippets.md)
**Key Topics**: Reusable code patterns, implementation details
- Weight loading patterns
- Training loop implementations
- Text generation functions
- Logging setups
- Common configurations

### 4. [Troubleshooting Guide](04_troubleshooting_guide.md)
**Key Topics**: Common problems and solutions
- CUDA/GPU issues and fixes
- Unicode/text display problems
- Weight loading errors
- Training issues
- Environment setup problems

## 🎯 Quick Reference

### Training Insights
- **GPU is 378x faster than CPU** for training
- **Perplexity** = effective branching factor (lower is better)
- **Loss reduction**: 10.99 → 3.35 (69% improvement) in 79 minutes
- **Key bottleneck**: Compute-bound, not memory-bound on RTX 3500

### Pretrained Weights
- **Best method**: PyTorch state dict from HuggingFace
- **URL**: `https://huggingface.co/rasbt/gpt2-from-scratch-pytorch`
- **Quality gap**: 777K tokens (ours) vs 40GB text (OpenAI) = huge difference
- **Key difference**: OpenAI uses `qkv_bias=True`

## 🚀 Key Takeaways

### 1. **Scale Matters**
- Our model: 777K tokens → fragmented output
- GPT-2: 40GB text → coherent, grammatical output
- Lesson: For production, always start with pretrained models

### 2. **GPU Setup is Critical**
- CUDA initialization failures can waste days
- Always verify GPU is actually being used
- Add explicit checks and `--force-cpu` flags for testing

### 3. **Monitoring is Essential**
- Use both W&B (cloud) and TensorBoard (local)
- Track loss, perplexity, and generated samples
- Log training loss every step, validation less frequently

### 4. **Weight Loading Complexity**
- TensorFlow → PyTorch requires careful transformation
- Weight transposition, QKV splitting, parameter remapping
- Pre-converted weights save significant time

## 📊 Performance Summary

| Metric | Our Training | OpenAI GPT-2 |
|--------|--------------|--------------|
| Data Size | 777K tokens | ~40GB text |
| Training Time | 79 minutes | Weeks |
| Parameters | 124M | 124M |
| Final Loss | 3.35 | ~2.5 |
| Coherence | Poor | Excellent |

## 🛠️ Practical Commands

```bash
# Train your own model
python gpt_pretraining_pipeline/src/train_gpt.py --epochs 5

# Load pretrained weights (recommended)
python gpt_pretraining_pipeline/scripts/load_pytorch_weights.py

# Compare models
python gpt_pretraining_pipeline/scripts/explore_pretrained_weights.py
```

## 📈 Learning Path

1. ✅ **Chapter 2-4**: Built GPT architecture from scratch
2. ✅ **Chapter 5**: Pretrained on small dataset (777K tokens)
3. ✅ **Weight Loading**: Loaded OpenAI's pretrained weights
4. 🔄 **Next**: Chapter 6 - Fine-tuning for specific tasks

## 💡 Wisdom Gained

> "The difference between a model trained on megabytes vs gigabytes of data is not just quantitative—it's qualitative. Coherent language understanding emerges from scale."

## 🔗 Quick Links

- [Original Book](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Main Repository](https://github.com/rasbt/LLMs-from-scratch)
- [HuggingFace Weights](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch)

---

*Last Updated: September 2025*
