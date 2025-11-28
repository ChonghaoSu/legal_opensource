"""
Dataset utilities for fine-tuning and DPO training
"""
import json
from typing import List, Dict, Any
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def format_instruction_prompt(instruction: str, input_text: str = None) -> str:
    """Format instruction following Llama 3 chat template"""
    if input_text:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful legal document analyst assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

Document:
{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful legal document analyst assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def preprocess_function_finetune(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict[str, Any]:
    """Preprocess function for fine-tuning"""
    prompts = []
    completions = []
    
    for i in range(len(examples['prompt'])):
        prompt = examples['prompt'][i]
        completion = examples['completion'][i] if 'completion' in examples else examples.get('summary', [''])[i]
        
        # Format with Llama 3 template
        full_text = format_instruction_prompt(prompt, None) + completion
        
        prompts.append(full_text)
        completions.append(completion)
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    labels = model_inputs["input_ids"].clone()
    
    # Mask prompt tokens in labels (only compute loss on completion)
    # Find where assistant response starts
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        if prompt_len < max_length:
            labels[i][:prompt_len] = -100
    
    model_inputs["labels"] = labels
    
    return model_inputs


def preprocess_function_dpo(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 512
) -> Dict[str, Any]:
    """Preprocess function for DPO training"""
    prompts = []
    chosen = []
    rejected = []
    
    for i in range(len(examples['prompt'])):
        prompt = examples['prompt'][i]
        chosen_text = examples['chosen'][i]
        rejected_text = examples['rejected'][i]
        
        # Format prompts
        chosen_prompt = format_instruction_prompt(prompt, None) + chosen_text
        rejected_prompt = format_instruction_prompt(prompt, None) + rejected_text
        
        prompts.append(prompt)
        chosen.append(chosen_prompt)
        rejected.append(rejected_prompt)
    
    # Tokenize prompts
    tokenized_prompts = tokenizer(
        prompts,
        max_length=max_prompt_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Tokenize chosen and rejected
    tokenized_chosen = tokenizer(
        chosen,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized_rejected = tokenizer(
        rejected,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized_prompts["input_ids"],
        "attention_mask": tokenized_prompts["attention_mask"],
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
    }


def create_dpo_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Create DPO dataset from examples"""
    # Format: each example should have 'prompt', 'chosen', 'rejected'
    dataset_dict = {
        'prompt': [ex['prompt'] for ex in examples],
        'chosen': [ex['chosen'] for ex in examples],
        'rejected': [ex['rejected'] for ex in examples]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset.map(
        lambda x: preprocess_function_dpo(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )


def create_finetune_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Create fine-tuning dataset from examples"""
    # Format: each example should have 'prompt' and 'completion' or 'summary'
    dataset_dict = {
        'prompt': [ex['prompt'] for ex in examples],
        'completion': [ex.get('completion', ex.get('summary', '')) for ex in examples]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset.map(
        lambda x: preprocess_function_finetune(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )


