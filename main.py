"""
Main application for legal document analysis
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.models.llama3_model import Llama3Model
from src.data_processing.legal_document_processor import LegalDocumentProcessor
from src.evaluation.citation_evaluator import CitationEvaluator


def analyze_document(
    model: Llama3Model,
    processor: LegalDocumentProcessor,
    document_path: str,
    output_dir: str = None
) -> Dict[str, Any]:
    """Analyze a legal document and generate summary with citations"""
    
    # Process document
    print(f"Processing document: {document_path}...")
    doc_data = processor.process_document(document_path)
    
    print(f"Found {doc_data['num_citations']} citations in {doc_data['num_chunks']} chunks")
    
    # Generate summaries for each chunk
    summaries = []
    all_citations = []
    
    for i, chunk in enumerate(doc_data['chunks']):
        print(f"Processing chunk {i+1}/{len(doc_data['chunks'])}...")
        
        # Create prompt
        chunk_citations = [c for c in doc_data['citations'] 
                          if chunk.find(c['text']) >= 0]
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a legal document analyst assistant. Analyze the following legal document excerpt and create a summary with accurate citations.<|eot_id|><|start_header_id|>user<|end_header_id|>

Document Excerpt:
{chunk}

Citations found: {', '.join([c['text'] for c in chunk_citations[:10]])}

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate summary
        generated = model.generate(
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract summary (text after the prompt)
        summary = generated.split("Summary:")[-1].strip()
        if "<|eot_id|>" in summary:
            summary = summary.split("<|eot_id|>")[0].strip()
        
        summaries.append({
            'chunk_id': i,
            'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
            'summary': summary,
            'citations': [c['text'] for c in chunk_citations]
        })
        
        all_citations.extend([c['text'] for c in chunk_citations])
    
    # Create overall summary
    print("Creating overall summary...")
    overall_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a legal document analyst assistant. Create a comprehensive summary of a legal document based on the following chunk summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Chunk Summaries:
{chr(10).join([f"Chunk {s['chunk_id']+1}: {s['summary']}" for s in summaries[:5]])}

Instructions:
1. Create a comprehensive summary that synthesizes all chunk summaries
2. Include all relevant citations
3. Organize the summary logically
4. Ensure every claim is properly cited

Overall Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    overall_generated = model.generate(
        overall_prompt,
        max_length=1024,
        temperature=0.7,
        top_p=0.9
    )
    
    overall_summary = overall_generated.split("Overall Summary:")[-1].strip()
    if "<|eot_id|>" in overall_summary:
        overall_summary = overall_summary.split("<|eot_id|>")[0].strip()
    
    # Evaluate citations
    evaluator = CitationEvaluator()
    citation_metrics = evaluator.evaluate_summary(
        overall_summary,
        doc_data['full_text']
    )
    
    result = {
        'document_path': document_path,
        'overall_summary': overall_summary,
        'chunk_summaries': summaries,
        'all_citations': list(set(all_citations)),
        'citation_metrics': citation_metrics,
        'document_stats': {
            'total_citations': doc_data['num_citations'],
            'total_chunks': doc_data['num_chunks'],
            'document_length': len(doc_data['full_text'])
        }
    }
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        doc_name = Path(document_path).stem
        output_file = output_path / f"{doc_name}_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Legal Document Analysis Tool")
    parser.add_argument("--input", type=str, required=True,
                       help="Input document path (PDF, DOCX, or TXT)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str, default="./models/dpo_llama3",
                       help="Path to model checkpoint (default: DPO model)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    if not Path(args.model_path).exists():
        print(f"Warning: Model path {args.model_path} not found. Using base Llama 3 model.")
        model = Llama3Model(use_4bit=args.use_4bit)
    else:
        model = Llama3Model(
            model_name=args.model_path,
            use_4bit=args.use_4bit
        )
    
    # Initialize processor
    processor = LegalDocumentProcessor()
    
    # Analyze document
    result = analyze_document(model, processor, args.input, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nDocument: {result['document_path']}")
    print(f"Total Citations Found: {result['document_stats']['total_citations']}")
    print(f"\nCitation Metrics:")
    print(f"  Precision: {result['citation_metrics']['citation_precision']:.4f}")
    print(f"  Recall: {result['citation_metrics']['citation_recall']:.4f}")
    print(f"  F1 Score: {result['citation_metrics']['citation_f1']:.4f}")
    print(f"  Hallucination Rate: {result['citation_metrics']['hallucination_rate']:.4f}")
    print(f"\nOverall Summary Preview:")
    print(result['overall_summary'][:500] + "..." if len(result['overall_summary']) > 500 else result['overall_summary'])


if __name__ == "__main__":
    main()


