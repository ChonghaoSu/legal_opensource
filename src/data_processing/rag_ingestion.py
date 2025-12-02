# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:54:44 2025

@author: rtvid
"""

import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Import config from the configs package
from configs.rag_config import RAGConfig

class LegalKnowledgeBase:
    """
    Handles ingestion, chunking, and indexing of legal documents.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=RAGConfig.EMBEDDING_MODEL)
        self.vector_store = None

    def ingest_and_index(self):
        """Loads PDFs, chunks them, and builds/saves the FAISS index."""
        print(f"Loading documents from {RAGConfig.PDF_SOURCE_DIR}...")
        
        loader = DirectoryLoader(
            RAGConfig.PDF_SOURCE_DIR,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        if not documents:
            print(f"(!) No PDF documents found in {RAGConfig.PDF_SOURCE_DIR}")
            return False

        # RecursiveSplitter is best for legal text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(texts)} chunks.")

        print("Embedding chunks and building FAISS index...")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        self.vector_store.save_local(RAGConfig.INDEX_SAVE_PATH)
        print(f"Index saved to {RAGConfig.INDEX_SAVE_PATH}")
        return True

    def load_index(self):
        """Loads an existing FAISS index from disk."""
        if os.path.exists(RAGConfig.INDEX_SAVE_PATH):
            self.vector_store = FAISS.load_local(
                RAGConfig.INDEX_SAVE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Loaded existing FAISS index.")
            return True
        else:
            print("(!) Index not found.")
            return False

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieves the top-k relevant chunks."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or build index first.")
            
        docs = self.vector_store.similarity_search(query, k=RAGConfig.TOP_K_RETRIEVAL)
        return [{"content": d.page_content, "source": d.metadata.get("source", "Unknown")} for d in docs]