"""
Script to create sample training data for fine-tuning and DPO
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def create_finetune_example(
    prompt: str,
    completion: str
) -> Dict[str, Any]:
    """Create a fine-tuning example"""
    return {
        'prompt': prompt,
        'completion': completion
    }


def create_dpo_example(
    prompt: str,
    chosen: str,
    rejected: str
) -> Dict[str, Any]:
    """Create a DPO example (chosen vs rejected response)"""
    return {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected
    }


def create_sample_data():
    """Create sample training data"""
    
    # Sample fine-tuning data
    finetune_examples = [
        create_finetune_example(
            prompt="""Analyze the following legal document excerpt and create a summary with accurate citations.

Document Excerpt:
The court held that the defendant's motion to dismiss should be granted. See Smith v. Jones, 123 F.3d 456 (2020). The plaintiff failed to state a claim upon which relief can be granted under Rule 12(b)(6) of the Federal Rules of Civil Procedure.

Citations found: Smith v. Jones, 123 F.3d 456, Rule 12(b)(6)

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:""",
            completion="""The court granted the defendant's motion to dismiss (Smith v. Jones, 123 F.3d 456 (2020)) because the plaintiff failed to state a claim under Federal Rule of Civil Procedure 12(b)(6)."""
        ),
        create_finetune_example(
            prompt="""Analyze the following legal document excerpt and create a summary with accurate citations.

Document Excerpt:
Under the Federal Rules of Evidence, hearsay is generally inadmissible unless it falls within an exception. See Fed. R. Evid. 802. However, statements made by a party opponent are admissible as non-hearsay. See Fed. R. Evid. 801(d)(2).

Citations found: Fed. R. Evid. 802, Fed. R. Evid. 801(d)(2)

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:""",
            completion="""Hearsay is generally inadmissible under Federal Rule of Evidence 802, but statements by party opponents are admissible as non-hearsay under Federal Rule of Evidence 801(d)(2)."""
        ),
    ]
    
    # Sample DPO data (chosen = good response with citations, rejected = bad response without citations or with hallucinations)
    dpo_examples = [
        create_dpo_example(
            prompt="""Analyze the following legal document excerpt and create a summary with accurate citations.

Document Excerpt:
The court held that the defendant's motion to dismiss should be granted. See Smith v. Jones, 123 F.3d 456 (2020). The plaintiff failed to state a claim upon which relief can be granted under Rule 12(b)(6) of the Federal Rules of Civil Procedure.

Citations found: Smith v. Jones, 123 F.3d 456, Rule 12(b)(6)

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:""",
            chosen="""The court granted the defendant's motion to dismiss (Smith v. Jones, 123 F.3d 456 (2020)) because the plaintiff failed to state a claim under Federal Rule of Civil Procedure 12(b)(6).""",
            rejected="""The court granted the motion to dismiss because the plaintiff's claim was insufficient. The court cited several cases including Brown v. White, 789 F.2d 123 (2019) in support of this decision."""
        ),
        create_dpo_example(
            prompt="""Analyze the following legal document excerpt and create a summary with accurate citations.

Document Excerpt:
Under the Federal Rules of Evidence, hearsay is generally inadmissible unless it falls within an exception. See Fed. R. Evid. 802. However, statements made by a party opponent are admissible as non-hearsay. See Fed. R. Evid. 801(d)(2).

Citations found: Fed. R. Evid. 802, Fed. R. Evid. 801(d)(2)

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:""",
            chosen="""Hearsay is generally inadmissible under Federal Rule of Evidence 802, but statements by party opponents are admissible as non-hearsay under Federal Rule of Evidence 801(d)(2).""",
            rejected="""Hearsay evidence is typically not allowed in court, but there are some exceptions. Party opponent statements can be used as evidence in certain circumstances."""
        ),
    ]
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save fine-tuning data
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    
    # Split examples (80/20)
    split_idx = int(len(finetune_examples) * 0.8)
    train_data = finetune_examples[:split_idx] if split_idx > 0 else finetune_examples
    val_data = finetune_examples[split_idx:] if split_idx < len(finetune_examples) else []
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Created {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Save DPO data
    dpo_train_path = data_dir / "dpo_train.jsonl"
    dpo_val_path = data_dir / "dpo_val.jsonl"
    
    dpo_split_idx = int(len(dpo_examples) * 0.8)
    dpo_train_data = dpo_examples[:dpo_split_idx] if dpo_split_idx > 0 else dpo_examples
    dpo_val_data = dpo_examples[dpo_split_idx:] if dpo_split_idx < len(dpo_examples) else []
    
    with open(dpo_train_path, 'w', encoding='utf-8') as f:
        for ex in dpo_train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(dpo_val_path, 'w', encoding='utf-8') as f:
        for ex in dpo_val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Created {len(dpo_train_data)} DPO training examples and {len(dpo_val_data)} DPO validation examples")
    print("\nNote: These are sample examples. For production use, you should create a larger dataset from real legal documents.")


if __name__ == "__main__":
    create_sample_data()


