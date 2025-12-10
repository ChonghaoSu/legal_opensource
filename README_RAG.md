# Legal RAG: Retrieval-Augmented Generation Pipeline

This module implements a robust **Retrieval-Augmented Generation (RAG)** system designed specifically for legal document analysis. It is optimized for **Google Colab (T4 GPU)** environments, featuring a "fail-safe" data ingestion engine and aggressive memory management to prevent "Out of Memory" (OOM) errors on free-tier resources.

## 🚀 Key Features

* **Fail-Safe Ingestion:** Automatically detects if user documents are missing and generates synthetic legal contracts (NDAs, MSAs) for immediate testing and demos.
* **Transient Memory Management:** Implements a "Load-and-Release" architecture where the LLM (Qwen 2.5) is only loaded into VRAM during generation and immediately flushed afterwards.
* **Hybrid Document Support:** Native support for ingesting both `.pdf` and `.txt` files.
* **Legal-Specific Chunking:** Uses recursive splitting with overlap to preserve the context of long legal clauses (e.g., ensuring "Section 4.1" stays with its paragraph).
* **Local Vector Store:** Uses ChromaDB locally, requiring no external API keys for storage.

## 🛠️ Prerequisites

* **Google Account** (for Google Colab)
* **Hugging Face Account** (with a valid Access Token)
* **GPU Runtime:** T4 (Standard Free Tier) or higher.

## ⚙️ Setup & Installation

### 1. Environment Setup
Run the "Master Setup Cell" at the start of your notebook. This handles:
* Mounting Google Drive (for model persistence).
* Installing dependencies (`langchain`, `chromadb`, `transformers`, `unstructured`).
* **Interactive Authentication:** Securely logs into Hugging Face to access gated models.

### 2. Document Ingestion
You have two options for data:

* **Option A (Real Data):** Drag and drop your legal PDFs into the `rag_documents/` folder in the Colab sidebar.
* **Option B (Test Mode):** Leave the folder empty. The system will auto-generate `synthetic_nda.txt` and `synthetic_msa.txt` to ensure the pipeline never crashes during demos.

## 🖥️ Usage Guide

### Step 1: Build the Knowledge Base
Run the ingestion cell to process documents. The system logic:
1.  **Load:** Scans `rag_documents/` for files.
2.  **Split:** Chunks text into 1000-character segments (200-char overlap).
3.  **Embed:** Uses `sentence-transformers/all-MiniLM-L6-v2` to create vector embeddings.
4.  **Index:** Stores vectors in a local `ChromaDB` instance.

### Step 2: Query the System
Use the transient bot function to save GPU memory:

```python
# This loads the model, generates an answer, and immediately clears VRAM
answer = ask_lawyer_bot_transient("What are the termination conditions?")
print(answer)
