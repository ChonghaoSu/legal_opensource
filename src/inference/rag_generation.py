# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:55:28 2025

@author: rtvid
"""

import os
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel
from configs.rag_config import RAGConfig

class CitationGenerator:
    """
    Loads the SFT/DPO adapter and generates the JSON evidence map.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading base model: {RAGConfig.BASE_MODEL_ID}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                RAGConfig.BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(RAGConfig.BASE_MODEL_ID)
        except Exception as e:
            print(f"CRITICAL ERROR loading base model: {e}")
            raise e

        # Load SFT/DPO Adapter
        if os.path.exists(RAGConfig.ADAPTER_PATH):
            print(f"✅ Loading SFT/DPO adapter from {RAGConfig.ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(self.model, RAGConfig.ADAPTER_PATH)
        else:
            print(f"⚠️  WARNING: Adapter path '{RAGConfig.ADAPTER_PATH}' not found.")
            print("   Running with BASE MODEL only. Results will lack strict JSON formatting.")

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Constructs the prompt and generates the JSON output."""
        context_text = "\n\n".join([f"Source ({c['source']}): {c['content']}" for c in context_chunks])
        
        prompt = f"""### Instruction:
You are a legal evidence assistant. Analyze the provided context documents and answer the user's question.
Output strictly in JSON format with the following schema:
{{
  "claim": "The answer to the question",
  "evidence": [
    {{
      "text": "Exact quote from text",
      "source": "Source document name"
    }}
  ]
}}
If the context does not contain the answer, return: {{ "claim": "Insufficient Context", "evidence": [] }}

### Context:
{context_text}

### Question:
{query}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=RAGConfig.MAX_NEW_TOKENS,
                temperature=RAGConfig.TEMPERATURE, 
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response_clean = response.split("### Response:")[-1].strip()
        else:
            response_clean = response
            
        return response_clean
