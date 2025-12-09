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
        
        # --- 1. GUARDRAIL: Handle Empty Context ---
        if not context_chunks:
            print("(!) DEBUG: No context chunks provided to generator.")
            return '{ "claim": "Insufficient Context", "evidence": [] }'

        context_text = "\n\n".join([f"Source ({c['source']}): {c['content']}" for c in context_chunks])
        
        # --- 2. DEBUGGING: Print what the model is reading ---
        print("\n--- [DEBUG] RETRIEVED CONTEXT ---")
        print(context_text[:500] + "..." if len(context_text) > 500 else context_text)
        print("---------------------------------\n")

        # --- 3. PROMPT ENGINEERING: One-Shot Example ---
        # We give the model an example interaction so it knows exactly what JSON looks like.
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a legal evidence assistant. Analyze the provided context documents and answer the user's question.
Output strictly in JSON format. Do not output conversational text.

Example Format:
Question: "What is the termination notice period?"
Context: "Source (Contract.pdf): Either party may terminate this agreement with 30 days prior written notice."
Output: {{ "claim": "30 days", "evidence": [ {{ "text": "terminate this agreement with 30 days prior written notice", "source": "Contract.pdf" }} ] }}

If the context does not contain the answer, return: {{ "claim": "Insufficient Context", "evidence": [] }}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context_text}

Question:
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
        
        # Clean up Llama-3 specific generation artifacts
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response_clean = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response_clean = response
            
        return response_clean
