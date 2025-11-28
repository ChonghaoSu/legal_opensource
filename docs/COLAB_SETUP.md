# Google Colab Setup Guide

This guide walks you through setting up and running the legal document analysis training on Google Colab.

## Prerequisites

1. **Google Account**: Sign in to [Google Colab](https://colab.research.google.com/)
2. **Hugging Face Account**: Get your token from [HF Settings](https://huggingface.co/settings/tokens)
3. **Project Files**: Have the project code ready (or clone from GitHub)

## Step-by-Step Setup

### 1. Open the Colab Notebook

- Option A: Upload `colab/legal_document_analysis_colab.ipynb` to Colab
- Option B: Create a new notebook and copy cells from the notebook file

### 2. Enable GPU

1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Choose **T4** (free) or **A100** (Colab Pro)
4. Click **Save**

### 3. Run Setup Cells

Execute the cells in order:

#### Cell 1: Check GPU
```python
# Verifies GPU is available
```

#### Cell 2: Install Dependencies
```python
# Installs all required packages
```

#### Cell 3: Mount Google Drive (Optional)
```python
# Mounts your Google Drive to save models
```

### 4. Set Up Project Files

You have two options:

#### Option A: Upload Files Manually
1. Use the file browser (📁 icon) on the left
2. Upload all project files:
   - `src/` directory
   - `configs/` directory  
   - `train_finetune.py`
   - `train_dpo.py`
   - Other necessary files

#### Option B: Clone from GitHub
If your project is on GitHub:
```python
!git clone https://github.com/yourusername/legal-document-analysis.git .
```

### 5. Set Environment Variables

**⚠️ Important**: Set your Hugging Face token:

```python
os.environ['HF_TOKEN'] = 'your_actual_token_here'
```

**Better option** (more secure): Use Colab secrets:
1. Go to **🔑 Secrets** in the left sidebar
2. Add a secret named `HF_TOKEN` with your token value
3. Use in code:
```python
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

### 6. Prepare Training Data

Upload your training data files:
- `data/train.jsonl` - Training examples
- `data/val.jsonl` - Validation examples
- `data/dpo_train.jsonl` - DPO training pairs
- `data/dpo_val.jsonl` - DPO validation pairs

Or use the sample data creation cell for testing.

### 7. Run Training

#### Fine-tuning
```python
!python train_finetune.py --config configs/finetune_config.yaml
```

**Expected time**: 1-4 hours depending on dataset size and GPU

#### DPO Training
```python
!python train_dpo.py --config configs/dpo_config.yaml
```

**Expected time**: 1-3 hours

### 8. Save Models

After training, save models to Google Drive:

```python
# Models will be copied to /content/drive/MyDrive/legal_document_models
```

## Colab-Specific Considerations

### Memory Management

Colab has limited resources:
- **Free tier**: ~15GB RAM, T4 GPU (16GB VRAM)
- **Pro tier**: More RAM, better GPUs

**Optimizations for Colab:**
- Use smaller batch sizes (already set in config)
- Reduce `max_seq_length` if you run out of memory
- Use gradient checkpointing (enabled by default)
- Clear cache between runs: `torch.cuda.empty_cache()`

### Session Limits

- **Free tier**: Sessions timeout after ~90 minutes of inactivity
- **Pro tier**: Longer sessions, but still limited

**Tips:**
- Save checkpoints frequently
- Use Google Drive to persist models
- Resume training from checkpoints if session ends

### File Persistence

Colab files are **temporary**:
- Files in `/content` are deleted when session ends
- **Always save to Google Drive** for persistence

### GPU Availability

Free tier GPUs:
- May not always be available
- Usage limits apply
- May disconnect during long training

**Solutions:**
- Use Colab Pro for better availability
- Save checkpoints frequently
- Resume from last checkpoint if disconnected

## Troubleshooting

### Out of Memory (OOM) Errors

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   per_device_train_batch_size: 1  # Reduce from 2
   gradient_accumulation_steps: 16  # Increase to compensate
   ```

2. Reduce sequence length:
   ```yaml
   max_seq_length: 1024  # Reduce from 2048
   ```

3. Use smaller LoRA rank:
   ```yaml
   lora:
     r: 8  # Reduce from 16
   ```

### Slow Training

**Possible causes:**
- CPU mode (no GPU): Check GPU is enabled
- Small batch size: Increase `gradient_accumulation_steps`
- Large dataset: Consider using `max_samples` to limit data

### Import Errors

**Solution:**
```python
# Reinstall packages
!pip install -q --upgrade transformers peft trl bitsandbytes
```

### Hugging Face Authentication

If you get authentication errors:
1. Verify token is set: `print(os.getenv('HF_TOKEN'))`
2. Request access to Llama 3 on Hugging Face
3. Accept the model license

## Best Practices

1. **Start Small**: Test with sample data first
2. **Monitor Progress**: Watch loss values and GPU usage
3. **Save Frequently**: Use `save_steps` to save checkpoints
4. **Use Drive**: Always save final models to Google Drive
5. **Document Changes**: Note any config changes you make
6. **Check Logs**: Review training logs for errors

## Loading Saved Models

To load a model saved to Drive in a new session:

```python
from google.colab import drive
drive.mount('/content/drive')

from src.models.llama3_model import Llama3Model

model = Llama3Model(
    model_name='/content/drive/MyDrive/legal_document_models/dpo_llama3',
    use_4bit=True
)
```

## Next Steps

After training:
1. Download models from Google Drive
2. Use locally with `main.py` for inference
3. Evaluate with `evaluate.py`
4. Deploy for production use

## Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab GPU Limits](https://colab.research.google.com/signup)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)


