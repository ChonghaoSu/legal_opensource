"""
Utility script to create DPO pairs from fine-tuning data
This script helps generate rejected examples by modifying chosen examples
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import re


def remove_citations(text: str) -> str:
    """Remove citations from text to create a rejected example"""
    # Common citation patterns
    patterns = [
        r'\([^)]*\d+\s+[A-Z][a-z]+\s+\d+[^)]*\)',  # (Smith v. Jones, 123 F.3d 456)
        r'See\s+[^.]*\.',  # See Smith v. Jones, 123 F.3d 456.
        r'\d+\s+F\.\d+d\s+\d+',  # 123 F.3d 456
        r'\d+\s+F\.\s+Supp\.\s+\d+',  # 123 F. Supp. 456
        r'[A-Z]\.\s+R\.\s+Evid\.\s+\d+',  # Fed. R. Evid. 802
        r'[A-Z]\.\s+R\.\s+Civ\.\s+P\.\s+\d+',  # Fed. R. Civ. P. 12
    ]
    
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def add_hallucinated_citation(text: str) -> str:
    """Add a fake citation to create a rejected example"""
    fake_citations = [
        "Brown v. White, 789 F.2d 123 (2019)",
        "Johnson v. Smith, 456 F.3d 789 (2021)",
        "See Fed. R. Civ. P. 56",
        "See 28 U.S.C. § 1331",
    ]
    
    # Add citation at the end if not present
    if not re.search(r'\d+\s+[A-Z]', text):
        text += f" ({random.choice(fake_citations)})"
    
    return text


def create_rejected_variants(chosen: str, variant_type: str = "remove_citations") -> str:
    """Create a rejected variant from a chosen response"""
    if variant_type == "remove_citations":
        return remove_citations(chosen)
    elif variant_type == "add_hallucinated":
        return add_hallucinated_citation(chosen)
    elif variant_type == "generic":
        # Make it more generic without specific details
        # Remove case names and specific numbers
        result = re.sub(r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+', 'the case', chosen, flags=re.IGNORECASE)
        result = re.sub(r'\d+\s+F\.\d+d\s+\d+', 'the court', result)
        return result
    else:
        # Just remove some citations
        return remove_citations(chosen)


def create_dpo_pairs_from_finetune(
    finetune_data_path: str,
    output_path: str,
    variant_types: List[str] = None
):
    """Create DPO pairs from fine-tuning data"""
    
    if variant_types is None:
        variant_types = ["remove_citations", "add_hallucinated", "generic"]
    
    # Load fine-tuning data
    print(f"Loading fine-tuning data from {finetune_data_path}...")
    finetune_examples = []
    with open(finetune_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                finetune_examples.append(json.loads(line))
    
    print(f"Loaded {len(finetune_examples)} examples")
    
    # Create DPO pairs
    dpo_examples = []
    for i, example in enumerate(finetune_examples):
        prompt = example.get('prompt', '')
        chosen = example.get('completion', example.get('summary', ''))
        
        if not chosen or not prompt:
            continue
        
        # Create rejected variant
        variant_type = random.choice(variant_types)
        rejected = create_rejected_variants(chosen, variant_type)
        
        # Ensure rejected is different from chosen
        if rejected.strip() == chosen.strip():
            rejected = remove_citations(chosen)  # Fallback
        
        dpo_examples.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(finetune_examples)} examples")
    
    # Save DPO data
    print(f"Saving {len(dpo_examples)} DPO examples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in dpo_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Created {len(dpo_examples)} DPO pairs")
    print("\nNote: These are automatically generated pairs. For best results, manually review and curate DPO pairs.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create DPO pairs from fine-tuning data")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to fine-tuning JSONL file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for DPO JSONL file")
    parser.add_argument("--variants", nargs="+",
                       choices=["remove_citations", "add_hallucinated", "generic"],
                       default=["remove_citations", "add_hallucinated"],
                       help="Types of rejected variants to create")
    
    args = parser.parse_args()
    
    create_dpo_pairs_from_finetune(
        args.input,
        args.output,
        args.variants
    )


if __name__ == "__main__":
    main()


