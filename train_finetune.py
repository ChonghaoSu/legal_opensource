"""
Fine-tuning script for Llama 3 on legal documents
"""
import os
import yaml
import argparse
from pathlib import Path
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
import torch

from src.models.llama3_model import Llama3Model
from src.data_processing.dataset_utils import (
    load_jsonl,
    create_finetune_dataset,
    format_instruction_prompt
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 on legal documents")
    parser.add_argument("--config", type=str, default="configs/finetune_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_config = config['model']
    lora_config_dict = config['lora']
    training_config = config['training']
    data_config = config['data']
    
    # Load model
    print("Loading Llama 3 model...")
    llama_model = Llama3Model(
        model_name=model_config['name'],
        use_4bit=model_config['use_4bit'],
        bnb_4bit_compute_dtype=model_config['bnb_4bit_compute_dtype'],
        bnb_4bit_quant_type=model_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=model_config['bnb_4bit_use_double_quant']
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        target_modules=lora_config_dict['target_modules'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias=lora_config_dict['bias'],
        task_type=lora_config_dict['task_type']
    )
    
    # Prepare model for training
    model = llama_model.prepare_for_training(lora_config)
    tokenizer = llama_model.get_tokenizer()
    
    # Load data
    print(f"Loading training data from {data_config['train_path']}...")
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
    print("Creating datasets...")
    train_dataset = create_finetune_dataset(train_examples, tokenizer)
    eval_dataset = create_finetune_dataset(val_examples, tokenizer) if val_examples else None
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
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
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        run_name="llama3-legal-finetune",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {training_config['output_dir']}...")
    trainer.save_model()
    tokenizer.save_pretrained(training_config['output_dir'])
    
    print("Training completed!")


if __name__ == "__main__":
    main()


