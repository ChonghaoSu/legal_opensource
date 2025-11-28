"""
Evaluation metrics for citation accuracy and hallucination detection
"""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class CitationEvaluator:
    """Evaluate citation accuracy and detect hallucinations"""
    
    def __init__(self):
        # Citation patterns
        self.citation_patterns = [
            r'\d+\s+[A-Z][a-z]+\s+\d+',
            r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+',
            r'\d+\s+F\.\d+d\s+\d+',
            r'\d+\s+F\.\s+Supp\.\s+\d+',
            r'[A-Z]\.\s+R\.\s+Evid\.\s+\d+',
            r'[A-Z]\.\s+R\.\s+Civ\.\s+P\.\s+\d+',
        ]
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return list(set(citations))  # Remove duplicates
    
    def citation_precision(
        self,
        generated_text: str,
        source_document: str
    ) -> float:
        """Calculate precision: citations in generated text that exist in source"""
        generated_citations = self.extract_citations(generated_text)
        source_citations = self.extract_citations(source_document)
        
        if len(generated_citations) == 0:
            return 1.0 if len(source_citations) == 0 else 0.0
        
        correct_citations = sum(1 for cit in generated_citations if cit in source_citations)
        return correct_citations / len(generated_citations)
    
    def citation_recall(
        self,
        generated_text: str,
        source_document: str,
        relevant_citations: List[str] = None
    ) -> float:
        """Calculate recall: relevant citations found in generated text"""
        generated_citations = self.extract_citations(generated_text)
        
        if relevant_citations:
            target_citations = relevant_citations
        else:
            target_citations = self.extract_citations(source_document)
        
        if len(target_citations) == 0:
            return 1.0 if len(generated_citations) == 0 else 0.0
        
        found_citations = sum(1 for cit in target_citations if cit in generated_citations)
        return found_citations / len(target_citations)
    
    def citation_f1(
        self,
        generated_text: str,
        source_document: str,
        relevant_citations: List[str] = None
    ) -> float:
        """Calculate F1 score for citations"""
        precision = self.citation_precision(generated_text, source_document)
        recall = self.citation_recall(generated_text, source_document, relevant_citations)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def detect_hallucinated_citations(
        self,
        generated_text: str,
        source_document: str
    ) -> List[str]:
        """Detect citations in generated text that don't exist in source"""
        generated_citations = self.extract_citations(generated_text)
        source_citations = self.extract_citations(source_document)
        
        hallucinated = [cit for cit in generated_citations if cit not in source_citations]
        return hallucinated
    
    def hallucination_rate(
        self,
        generated_text: str,
        source_document: str
    ) -> float:
        """Calculate rate of hallucinated citations"""
        generated_citations = self.extract_citations(generated_text)
        
        if len(generated_citations) == 0:
            return 0.0
        
        hallucinated = self.detect_hallucinated_citations(generated_text, source_document)
        return len(hallucinated) / len(generated_citations)
    
    def evaluate_summary(
        self,
        generated_summary: str,
        source_document: str,
        ground_truth_citations: List[str] = None
    ) -> Dict[str, float]:
        """Comprehensive evaluation of generated summary"""
        metrics = {
            'citation_precision': self.citation_precision(generated_summary, source_document),
            'citation_recall': self.citation_recall(generated_summary, source_document, ground_truth_citations),
            'citation_f1': self.citation_f1(generated_summary, source_document, ground_truth_citations),
            'hallucination_rate': self.hallucination_rate(generated_summary, source_document),
            'num_citations_generated': len(self.extract_citations(generated_summary)),
            'num_citations_source': len(self.extract_citations(source_document)),
        }
        
        return metrics


