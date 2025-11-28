# Hardware Requirements

## GPU Requirements

### Training

**A GPU is essential for practical training of Llama 3 models.**

#### Llama 3 8B Model

| Component | Minimum VRAM | Recommended VRAM |
|-----------|--------------|------------------|
| Fine-tuning (4-bit + LoRA) | 8GB | 16GB+ |
| DPO Training | 12GB | 16GB+ |
| Inference only | 4GB | 8GB+ |

**Example GPUs:**
- **Minimum**: NVIDIA RTX 3060 (12GB), RTX 3070 (8GB)
- **Recommended**: NVIDIA RTX 3090 (24GB), RTX 4090 (24GB), A100 (40GB/80GB)

#### Llama 3 70B Model

| Component | Minimum VRAM | Recommended VRAM |
|-----------|--------------|------------------|
| Fine-tuning (4-bit + LoRA) | 40GB | 80GB+ |
| DPO Training | 60GB | 80GB+ |
| Inference only | 40GB | 80GB+ |

**Required GPUs**: A100 (40GB/80GB), H100 (80GB), or multiple GPUs

## CPU Training

### Is it possible?

Technically yes, but **not practical**:
- Training on CPU will be **10-100x slower** than GPU
- A single epoch that takes 1 hour on GPU could take **10-100 hours on CPU**
- May run out of system RAM with larger models
- The code will automatically use CPU if no GPU is detected, but expect very long training times

### When CPU might be acceptable:

- **Inference only**: CPU inference is slow but feasible for occasional use
- **Very small datasets**: If you have <100 examples, CPU might be acceptable
- **Testing/Development**: For code testing without actual training

## Cloud GPU Options

If you don't have a local GPU, consider these cloud services:

### Free/Cheap Options

1. **Google Colab**
   - Free tier: T4 GPU (16GB VRAM) - limited hours
   - Pro: Better GPUs, more hours (~$10/month)
   - Good for: Getting started, small experiments

2. **Kaggle Notebooks**
   - Free: P100 GPU (16GB VRAM) - 30 hours/week
   - Good for: Learning and experimentation

3. **Hugging Face Spaces**
   - Free GPU hours available
   - Good for: Inference and demos

### Paid Cloud Services

1. **RunPod** / **Vast.ai**
   - Rent GPUs by the hour
   - RTX 3090: ~$0.30-0.50/hour
   - A100: ~$1-2/hour
   - Good for: Cost-effective training

2. **AWS SageMaker** / **Google Cloud** / **Azure**
   - Enterprise-grade GPU instances
   - More expensive but reliable
   - Good for: Production workloads

3. **Lambda Labs**
   - GPU cloud instances
   - Competitive pricing
   - Good for: Research and development

## Memory Optimization Tips

If you're close to VRAM limits:

1. **Reduce batch size**: Lower `per_device_train_batch_size` in config
2. **Increase gradient accumulation**: Compensate with `gradient_accumulation_steps`
3. **Use smaller LoRA rank**: Reduce `lora.r` in config (e.g., from 16 to 8)
4. **Shorter sequences**: Reduce `max_seq_length` if possible
5. **8-bit instead of 4-bit**: If you have more VRAM, 8-bit can be faster

## Checking Your Setup

### Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Monitor GPU usage during training:
```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Or use Python
pip install gpustat
gpustat -i 1
```

## Recommendations

- **For learning/experimentation**: Use Google Colab or Kaggle (free)
- **For serious training**: Get/rent a GPU with 16GB+ VRAM
- **For production**: Use cloud services or dedicated GPU servers
- **For inference only**: CPU is acceptable for occasional use, GPU recommended for frequent use


