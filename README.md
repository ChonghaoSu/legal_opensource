# Legal Document Analysis Tool with Llama 3

## Overview

This project implements an AI-powered legal document analysis tool that creates summaries and evidence maps with accurate citations. The system uses Llama 3, fine-tuned with Direct Preference Optimization (DPO) to minimize hallucinations and ensure citation accuracy.

## Problem Statement

Corporate legal teams spend excessive time searching through documents for evidence and verifying citations. This tool aims to:
- Speed up document analysis and verification
- Create accurate summaries with proper citations
- Prevent AI hallucinations
- Reduce legal costs and accelerate deal processes

## Features

- **Document Summarization**: Generate concise summaries of legal documents
- **Evidence Mapping**: Create evidence maps tied to specific citations
- **Citation Verification**: Ensure all claims are properly cited
- **Hallucination Prevention**: DPO training to minimize fabricated information
- **Fine-tuned Llama 3**: Customized for legal document analysis

## Architecture

- **Base Model**: Llama 3 (via Hugging Face Transformers)
- **Fine-tuning**: LoRA/QLoRA for efficient adaptation
- **DPO Training**: Direct Preference Optimization for preference alignment
- **Evaluation**: Citation accuracy and hallucination detection metrics

## Hardware Requirements

### GPU (Strongly Recommended)

**A GPU is highly recommended for training.** While the code can technically run on CPU, training Llama 3 without a GPU would be extremely slow (potentially days or weeks).

**Recommended GPU specifications:**
- **Minimum**: NVIDIA GPU with 8GB VRAM (e.g., RTX 3060, RTX 3070)
  - Can train Llama 3 8B with 4-bit quantization + LoRA
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)
  - Better performance and can handle larger batch sizes
- **For Llama 3 70B**: Requires 40GB+ VRAM (e.g., A100 40GB/80GB)

**Memory Requirements:**
- **Fine-tuning (8B model)**: ~8-12GB VRAM with 4-bit quantization
- **DPO Training (8B model)**: ~12-16GB VRAM (needs policy + reference model)
- **Inference only**: ~4-6GB VRAM

### CPU-Only Training (Not Recommended)

If you don't have a GPU, you have these options:

1. **Use Cloud Services:**
   - **Google Colab** (free tier with T4 GPU, Pro with better GPUs)
   - **Kaggle Notebooks** (free GPU hours)
   - **AWS SageMaker** / **Google Cloud** / **Azure** (pay-per-use)
   - **RunPod** / **Vast.ai** (cheaper GPU rentals)

2. **CPU Training** (extremely slow):
   - The code will automatically use CPU if no GPU is detected
   - Expect training to take 10-100x longer than on GPU
   - May run out of RAM with larger models

3. **Use Pre-trained Models**:
   - Skip training and use pre-trained checkpoints if available
   - Inference on CPU is more feasible (though still slow)

### Checking GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Set up environment variables (create `.env` file):
```bash
cp env_example .env
# Edit .env and add your HF_TOKEN
```

## Google Colab Setup

**Want to train on Google Colab?** See the detailed guide: [`docs/COLAB_SETUP.md`](docs/COLAB_SETUP.md)

Quick steps:
1. Open `colab/legal_document_analysis_colab.ipynb` in Google Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Set your `HF_TOKEN` in the notebook
4. Upload project files or clone from GitHub
5. Run the cells in order

The notebook includes:
- Automatic dependency installation
- GPU detection and setup
- Google Drive integration for saving models
- Optimized configs for Colab's T4 GPU
- Sample data creation

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp env_example .env
# Edit .env and add your HF_TOKEN
```

### 2. Create Sample Data (for testing)

```bash
python scripts/create_sample_data.py
```

This creates sample training data in the `data/` directory. For production use, you should prepare your own dataset from real legal documents.

### 3. Fine-tune Llama 3

```bash
python train_finetune.py --config configs/finetune_config.yaml
```

This will:
- Load Llama 3 with 4-bit quantization
- Apply LoRA for efficient fine-tuning
- Train on your legal document data
- Save the fine-tuned model to `models/finetuned_llama3`

### 4. Train with DPO

```bash
python train_dpo.py --config configs/dpo_config.yaml
```

This will:
- Load the fine-tuned model
- Train with Direct Preference Optimization to reduce hallucinations
- Save the DPO model to `models/dpo_llama3`

### 5. Use the Model

```bash
# Analyze a legal document
python main.py --input documents/legal_doc.pdf --output results/
```

### 6. Evaluate

```bash
python evaluate.py --model_path models/dpo_llama3 --test_data data/test/
```

## Project Structure

```
.
├── configs/              # Configuration files
├── data/                 # Training and test data
├── models/               # Saved model checkpoints
├── src/                  # Source code
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── data_processing/ # Data preprocessing
│   ├── evaluation/      # Evaluation metrics
│   └── inference/       # Inference pipeline
├── scripts/              # Utility scripts
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Important Notes

⚠️ **This tool does NOT provide legal advice, plan lawsuits, or replace lawyer judgment.**
It is designed to assist legal teams with document analysis and citation verification only.

## License

[Specify your license here]

