# Legal RAG Analysis System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChonghaoSu/legal_opensource/blob/main/legal_rag.ipynb)

> **Note:** Click the badge above to launch the interactive RAG pipeline directly in your browser.

## 📋 Project Overview

This repository contains a robust **Retrieval-Augmented Generation (RAG)** system designed specifically for legal document analysis. It is optimized for **Google Colab (T4 GPU)** environments, featuring a "fail-safe" data ingestion engine and aggressive memory management to prevent "Out of Memory" (OOM) errors on free-tier resources.

## 🚀 Key Features

* **Fail-Safe Ingestion:** Automatically detects if user documents are missing and generates synthetic legal contracts (NDAs, MSAs) for immediate testing and demos.
* **Transient Memory Management:** Implements a "Load-and-Release" architecture where the LLM (Qwen 2.5 or Llama 3) is only loaded into VRAM during generation and immediately flushed afterwards.
* **Hybrid Document Support:** Native support for ingesting both `.pdf` and `.txt` files.
* **Legal-Specific Chunking:** Uses recursive splitting with overlap to preserve the context of long legal clauses (e.g., ensuring "Section 4.1" stays with its paragraph).
* **Local Vector Store:** Uses ChromaDB locally, requiring no external API keys for storage.

## 🛠️ Prerequisites

* **Google Account** (for Google Colab)
* **Hugging Face Account** (with a valid Access Token)
* **GPU Runtime:** T4 (Standard Free Tier) or higher.

## ⚙️ Setup & Installation

### Option 1: One-Click Launch (Recommended)
Simply click the **Open in Colab** badge at the top of this file. The notebook `legal_rag.ipynb` comes pre-configured with:
* **Automatic Dependency Installation:** Installs `langchain`, `chromadb`, and `transformers` on launch.
* **Fail-Safe Data:** Automatically generates synthetic contracts if no real PDFs are uploaded.
* **GPU Optimization:** Auto-configures the T4 GPU environment.

### Option 2: Manual Clone
If you prefer to run the scripts locally or on a different server:

```bash
git clone [https://github.com/ChonghaoSu/legal_opensource.git](https://github.com/ChonghaoSu/legal_opensource.git)
cd legal_opensource
pip install -r requirements.txt
