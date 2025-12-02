#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:14:41 2025

@author: ralphvidaurri
"""

## pip install reportlab

import os
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configuration matching your RAG pipeline
DATA_OUTPUT_DIR = "./data/sec_filings"

# Mock Legal Content (The "Truth" for your RAG system)
MOCK_DOCUMENTS = [
    {
        "filename": "Cloud_Service_Agreement_v1.pdf",
        "title": "MASTER CLOUD SERVICES AGREEMENT",
        "content": [
            "SECTION 4.2: LIMITATION OF LIABILITY.",
            "Except for checks and wire transfers, the total cumulative liability of Provider shall not exceed the total fees paid by Customer to Provider in the twelve (12) month period preceding the claim.",
            "This limitation applies to all claims, including breach of contract, negligence, and strict liability.",
            "SECTION 5: INDEMNIFICATION.",
            "Provider agrees to indemnify Customer against any third-party claims alleging that the Service infringes upon any patent or copyright."
        ]
    },
    {
        "filename": "Apple_10K_Filing_2024.pdf",
        "title": "FORM 10-K: APPLE INC.",
        "content": [
            "ITEM 1A. RISK FACTORS.",
            "Global economic conditions could materially adversely affect the Company's business, results of operations, financial condition, and growth.",
            "The Company is subject to complex and changing laws and regulations worldwide, which exposes the Company to potential liabilities, increased costs, and other adverse effects on the Company's business.",
            "Data privacy and security: The Company handles significant amounts of personal data and is subject to evolving privacy laws (GDPR, CCPA)."
        ]
    },
    {
        "filename": "Vendor_NonDisclosure_Agreement.pdf",
        "title": "MUTUAL NON-DISCLOSURE AGREEMENT",
        "content": [
            "1. DEFINITION OF CONFIDENTIAL INFORMATION.",
            "Confidential Information means any non-public information disclosed by one party to the other, including trade secrets, source code, and customer lists.",
            "2. EXCLUSIONS.",
            "Confidential Information shall not include information that (a) is now or subsequently becomes generally known or available by publication, commercial use or otherwise through no fault of the Recipient."
        ]
    },
    {
        "filename": "Tesla_10K_Filing_Excerpt.pdf",
        "title": "FORM 10-K: TESLA, INC.",
        "content": [
            "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS.",
            "We may face competition from traditional automotive manufacturers who are entering the electric vehicle market.",
            "Our specific risks include supply chain constraints for lithium-ion battery cells and semiconductor chips.",
            "Regulatory credits: We earn tradable credits in the operation of our business under various regulations related to zero-emission vehicles."
        ]
    },
    {
        "filename": "Employment_Contract_Template.pdf",
        "title": "EXECUTIVE EMPLOYMENT AGREEMENT",
        "content": [
            "ARTICLE 3: COMPENSATION.",
            "Base Salary: The Executive shall receive a base salary of $250,000 per annum, payable in accordance with the Company's standard payroll practices.",
            "Severance: Upon termination without Cause, the Company shall pay Executive a lump sum equal to six (6) months of Base Salary.",
            "Non-Compete: For a period of one (1) year following termination, Executive shall not directly compete with the Company."
        ]
    }
]

def create_pdf(file_path, title, content_lines):
    """Generates a simple text-based PDF."""
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    
    # Body
    c.setFont("Helvetica", 12)
    y_position = height - 80
    
    for line in content_lines:
        # Simple text wrapping logic (very basic)
        words = line.split()
        current_line = ""
        for word in words:
            if c.stringWidth(current_line + word, "Helvetica", 12) < (width - 100):
                current_line += word + " "
            else:
                c.drawString(50, y_position, current_line)
                y_position -= 20
                current_line = word + " "
        
        # Draw the last part of the line
        c.drawString(50, y_position, current_line)
        y_position -= 30  # Extra space between paragraphs
        
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    c.save()
    print(f"Generated: {file_path}")

def main():
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)
        print(f"Created directory: {DATA_OUTPUT_DIR}")

    # 2. Generate PDFs
    print(f"Generating {len(MOCK_DOCUMENTS)} mock legal PDFs...")
    for doc in MOCK_DOCUMENTS:
        full_path = os.path.join(DATA_OUTPUT_DIR, doc['filename'])
        create_pdf(full_path, doc['title'], doc['content'])
    
    print("\n✅ Success! PDF generation complete.")
    print(f"Run 'legal_rag_pipeline.py' now to ingest these files.")

if __name__ == "__main__":
    main()
