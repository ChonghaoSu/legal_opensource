"""
Evaluation script for fine-tuned and DPO models
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.models.llama3_model import Llama3Model
from src.data_processing.legal_document_processor import LegalDocumentProcessor
from src.evaluation.citation_evaluator import CitationEvaluator


def load_test_data(test_data_path: str) -> List[Dict[str, Any]]:
    """Load test data"""
    test_data = []
    test_path = Path(test_data_path)
    
    if test_path.is_file() and test_path.suffix == '.jsonl':
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    elif test_path.is_dir():
        # Load all JSONL files in directory
        for jsonl_file in test_path.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
    
    return test_data


def evaluate_model(
    model: Llama3Model,
    test_data: List[Dict[str, Any]],
    evaluator: CitationEvaluator
) -> Dict[str, Any]:
    """Evaluate model on test data"""
    results = []
    
    for i, example in enumerate(test_data):
        print(f"Evaluating example {i+1}/{len(test_data)}...")
        
        prompt = example.get('prompt', '')
        source_document = example.get('source_document', example.get('document_text', ''))
        ground_truth_citations = example.get('ground_truth_citations', None)
        
        # Generate summary
        generated_text = model.generate(
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract just the generated part (after prompt)
        if prompt in generated_text:
            generated_summary = generated_text.split(prompt)[-1].strip()
        else:
            generated_summary = generated_text
        
        # Evaluate
        metrics = evaluator.evaluate_summary(
            generated_summary,
            source_document,
            ground_truth_citations
        )
        
        results.append({
            'example_id': i,
            'generated_summary': generated_summary,
            **metrics
        })
    
    # Aggregate metrics
    avg_metrics = {
        'avg_citation_precision': sum(r['citation_precision'] for r in results) / len(results),
        'avg_citation_recall': sum(r['citation_recall'] for r in results) / len(results),
        'avg_citation_f1': sum(r['citation_f1'] for r in results) / len(results),
        'avg_hallucination_rate': sum(r['hallucination_rate'] for r in results) / len(results),
        'total_examples': len(results)
    }
    
    return {
        'individual_results': results,
        'aggregate_metrics': avg_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned/DPO model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data (file or directory)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = Llama3Model(
        model_name=args.model_path,
        use_4bit=args.use_4bit
    )
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize evaluator
    evaluator = CitationEvaluator()
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, test_data, evaluator)
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total Examples: {results['aggregate_metrics']['total_examples']}")
    print(f"Average Citation Precision: {results['aggregate_metrics']['avg_citation_precision']:.4f}")
    print(f"Average Citation Recall: {results['aggregate_metrics']['avg_citation_recall']:.4f}")
    print(f"Average Citation F1: {results['aggregate_metrics']['avg_citation_f1']:.4f}")
    print(f"Average Hallucination Rate: {results['aggregate_metrics']['avg_hallucination_rate']:.4f}")
    
    # Save CSV for detailed analysis
    csv_output = args.output.replace('.json', '.csv')
    df = pd.DataFrame(results['individual_results'])
    df.to_csv(csv_output, index=False)
    print(f"\nDetailed results saved to {csv_output}")


if __name__ == "__main__":
    main()


