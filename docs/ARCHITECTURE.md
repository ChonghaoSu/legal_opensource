# Architecture Overview

## System Components

### 1. Model Integration (`src/models/llama3_model.py`)

The `Llama3Model` class provides:
- **Model Loading**: Loads Llama 3 with 4-bit quantization support
- **Tokenizer Management**: Handles Llama 3's tokenizer and chat template
- **LoRA Support**: Prepares model for parameter-efficient fine-tuning
- **Text Generation**: Provides interface for generating summaries

### 2. Data Processing (`src/data_processing/`)

#### Legal Document Processor
- Extracts text from PDF, DOCX, and TXT files
- Identifies legal citations using regex patterns
- Splits documents into manageable chunks
- Creates training examples with prompts and completions

#### Dataset Utils
- Formats data for fine-tuning (causal LM format)
- Formats data for DPO training (chosen/rejected pairs)
- Handles Llama 3's chat template format
- Creates Hugging Face datasets

### 3. Training Pipeline

#### Fine-tuning (`train_finetune.py`)
1. Loads Llama 3 with 4-bit quantization
2. Applies LoRA adapters for efficient training
3. Trains on legal document summarization task
4. Saves fine-tuned model checkpoint

#### DPO Training (`train_dpo.py`)
1. Loads fine-tuned model as policy model
2. Creates reference model (frozen copy)
3. Trains using Direct Preference Optimization
4. Optimizes for accurate citations and reduces hallucinations

### 4. Evaluation (`src/evaluation/citation_evaluator.py`)

Metrics:
- **Citation Precision**: Citations in output that exist in source
- **Citation Recall**: Relevant citations found in output
- **Citation F1**: Harmonic mean of precision and recall
- **Hallucination Rate**: Rate of fabricated citations

### 5. Inference (`main.py`)

The main application:
1. Processes legal documents
2. Generates chunk-level summaries
3. Creates overall document summary
4. Evaluates citation accuracy
5. Outputs results as JSON

## Training Flow

```
Base Llama 3
    ↓
[Fine-tuning with LoRA]
    ↓
Fine-tuned Model
    ↓
[DPO Training]
    ↓
DPO-optimized Model
    ↓
[Evaluation]
    ↓
Production Use
```

## Data Flow

### Fine-tuning
```
Legal Documents → Preprocessing → Training Examples → Fine-tuned Model
```

### DPO Training
```
Fine-tuned Model + Preference Pairs → DPO Training → DPO Model
```

### Inference
```
Legal Document → Processing → Chunking → Summarization → Evidence Map
```

## Key Technologies

- **Llama 3**: Meta's open-source LLM
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **DPO**: Direct Preference Optimization for alignment
- **4-bit Quantization**: Reduces memory requirements
- **Hugging Face Transformers**: Model framework
- **TRL**: Training library for DPO

## Memory Requirements

- **Base Model (8B)**: ~16GB (FP16) → ~4GB (4-bit)
- **Fine-tuning**: ~8-12GB with LoRA
- **DPO Training**: ~12-16GB (needs policy + reference model)

## Performance Considerations

1. **Quantization**: 4-bit reduces memory but may slightly impact quality
2. **LoRA**: Faster training, smaller checkpoints, but may need higher rank for complex tasks
3. **Chunking**: Large documents split for manageable processing
4. **Batch Size**: Adjusted based on available GPU memory

## Extension Points

- Add more citation patterns in `LegalDocumentProcessor`
- Customize evaluation metrics in `CitationEvaluator`
- Add new document formats in `extract_text()`
- Modify prompts in training data creation
- Add new DPO loss types in `train_dpo.py`


