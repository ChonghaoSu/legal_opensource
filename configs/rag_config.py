# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:52:41 2025

@author: rtvid
"""

import os

class RAGConfig:
    # --- PATHS ---
    # Automatically finds the project root (assuming this file is in legal_opensource/configs/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_SOURCE_DIR = os.path.join(DATA_DIR, "sec_filings")
    INDEX_SAVE_PATH = os.path.join(DATA_DIR, "faiss_legal_index")
    
    # Model directories
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # CRITICAL: Point this to your actual DPO output folder name
    # If your folder is named differently (e.g., "dpo_model"), change "final_dpo_adapter" below.
    ADAPTER_PATH = os.path.join(MODEL_DIR, "train_dpo")

    # --- MODEL SETTINGS ---
    BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- RAG PARAMETERS ---
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 3
    
    # --- GENERATION PARAMETERS ---
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.1 

    @classmethod
    def ensure_directories(cls):
        """Creates necessary data directories if they don't exist."""
        os.makedirs(cls.PDF_SOURCE_DIR, exist_ok=True)

        os.makedirs(cls.MODEL_DIR, exist_ok=True)
