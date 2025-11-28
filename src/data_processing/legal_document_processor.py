"""
Legal Document Processing and Preprocessing
"""
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import PyPDF2
from docx import Document
import spacy
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class LegalDocumentProcessor:
    """Process legal documents and extract text with citations"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Citation patterns (common legal citation formats)
        self.citation_patterns = [
            r'\d+\s+[A-Z][a-z]+\s+\d+',  # "123 U.S. 456"
            r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+',  # "Smith v. Jones"
            r'\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
            r'\d+\s+F\.\s+Supp\.\s+\d+',  # Federal Supplement
            r'[A-Z]\.\s+R\.\s+Evid\.\s+\d+',  # Federal Rules of Evidence
            r'[A-Z]\.\s+R\.\s+Civ\.\s+P\.\s+\d+',  # Federal Rules of Civil Procedure
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from document based on file extension"""
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif path.suffix.lower() in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def find_citations(self, text: str) -> List[Dict[str, Any]]:
        """Find all citations in the text"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        return citations
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_words + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_training_example(
        self,
        document_text: str,
        chunk_text: str,
        citations: List[Dict[str, Any]],
        summary: str = None
    ) -> Dict[str, Any]:
        """Create a training example for fine-tuning"""
        # Find citations in the chunk
        chunk_citations = [c for c in citations if c['start'] >= 0]
        
        # Create prompt
        prompt = f"""You are a legal document analyst. Analyze the following legal document excerpt and create a summary with accurate citations.

Document Excerpt:
{chunk_text}

Citations found: {', '.join([c['text'] for c in chunk_citations[:5]])}

Instructions:
1. Create a concise summary of the key points
2. Include all relevant citations in your summary
3. Ensure every factual claim is tied to a specific citation
4. Do not make up information or citations

Summary:"""
        
        return {
            'prompt': prompt,
            'document_text': document_text[:500],  # First 500 chars for context
            'chunk_text': chunk_text,
            'citations': [c['text'] for c in chunk_citations],
            'summary': summary
        }
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a complete document"""
        text = self.extract_text(file_path)
        citations = self.find_citations(text)
        chunks = self.split_into_chunks(text)
        
        return {
            'file_path': file_path,
            'full_text': text,
            'citations': citations,
            'chunks': chunks,
            'num_citations': len(citations),
            'num_chunks': len(chunks)
        }


