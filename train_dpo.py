"""
DPO (Direct Preference Optimization) training script for Llama 3
"""
import os
import yaml
import argparse
from pathlib import Path
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel
import torch

from src.models.llama3_model import Llama3Model
from src.data_processing.dataset_utils import (
    load_jsonl,
    create_dpo_dataset
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DPO training for Llama 3")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_config = config['model']
    training_config = config['training']
    dpo_config = config['dpo']
    data_config = config['data']
    
    # Load base model (fine-tuned)
    print(f"Loading base model from {model_config['base_model_path']}...")
    base_model_path = model_config['base_model_path']
    
    # Load model
    llama_model = Llama3Model(
        model_name=base_model_path if Path(base_model_path).exists() else model_config.get('name', "meta-llama/Meta-Llama-3-8B-Instruct"),
        use_4bit=model_config['use_4bit'],
        bnb_4bit_compute_dtype=model_config['bnb_4bit_compute_dtype'],
        bnb_4bit_quant_type=model_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=model_config['bnb_4bit_use_double_quant']
    )
    
    # Load fine-tuned weights if available
    from peft import PeftModel
    if Path(base_model_path).exists():
        try:
            # Try to load as PEFT model
            model = PeftModel.from_pretrained(
                llama_model.get_model(),
                base_model_path,
                device_map="auto"
            )
            print("Loaded fine-tuned PEFT weights")
        except Exception as e:
            # If not PEFT, just use base model
            model = llama_model.get_model()
            print(f"Using base model (fine-tuned weights not found or incompatible: {e})")
    else:
        model = llama_model.get_model()
        print("Using base model (fine-tuned path not found)")
    
    # Create reference model (frozen copy for DPO)
    print("Creating reference model...")
    from transformers import AutoModelForCausalLM
    
    # Load reference model (same as base, but frozen)
    # For DPO, reference model should be the same as the policy model before DPO training
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config.get('name', "meta-llama/Meta-Llama-3-8B-Instruct"),
        quantization_config=llama_model.bnb_config if model_config['use_4bit'] else None,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16 if not model_config['use_4bit'] else None,
    )
    
    # If we loaded a fine-tuned model, also load the same fine-tuned weights for reference
    if Path(base_model_path).exists():
        try:
            from peft import PeftModel
            ref_model = PeftModel.from_pretrained(
                ref_model,
                base_model_path,
                device_map="auto"
            )
            print("Loaded fine-tuned weights for reference model")
        except:
            print("Using base model as reference (fine-tuned weights not compatible)")
    
    tokenizer = llama_model.get_tokenizer()
    
    # Load DPO data
    print(f"Loading DPO training data from {data_config['train_path']}...")
    train_examples = load_jsonl(data_config['train_path'])
    if data_config.get('max_samples'):
        train_examples = train_examples[:data_config['max_samples']]
    
    val_examples = []
    if data_config.get('val_path') and Path(data_config['val_path']).exists():
        print(f"Loading validation data from {data_config['val_path']}...")
        val_examples = load_jsonl(data_config['val_path'])
        if data_config.get('max_samples'):
            val_examples = val_examples[:data_config['max_samples']]
    
    # Create datasets
    print("Creating DPO datasets...")
    train_dataset = create_dpo_dataset(train_examples, tokenizer)
    eval_dataset = create_dpo_dataset(val_examples, tokenizer) if val_examples else None
    
    # DPO Training arguments
    dpo_training_args = DPOConfig(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'] if eval_dataset else None,
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        optim=training_config['optim'],
        max_length=training_config['max_length'],
        max_prompt_length=training_config['max_prompt_length'],
        beta=dpo_config['beta'],
        loss_type=dpo_config['loss_type'],
        label_smoothing=dpo_config.get('label_smoothing', 0.0),
        reference_free=dpo_config.get('reference_free', False),
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        run_name="llama3-legal-dpo",
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_training_args,
        beta=dpo_config['beta'],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=training_config['max_length'],
        max_prompt_length=training_config['max_prompt_length'],
    )
    
    # Train
    print("Starting DPO training...")
    dpo_trainer.train()
    
    # Save final model
    print(f"Saving DPO model to {training_config['output_dir']}...")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(training_config['output_dir'])
    
    print("DPO training completed!")


if __name__ == "__main__":
    main()

