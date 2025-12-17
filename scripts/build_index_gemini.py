"""
Build Gemini Index Script
==========================
Generate embeddings using Gemini and build FAISS index.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding_engine_gemini import GeminiEmbeddingEngine


def main():
    """Build FAISS index using Gemini embeddings."""
    print("=" * 70)
    print("BUILDING GEMINI-POWERED EMBEDDINGS AND FAISS INDEX")
    print("=" * 70)

    # Load scraped data
    data_file = "data/shl_catalog.json"
    print(f"\nStep 1: Loading data from {data_file}...")

    with open(data_file, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"Loaded {len(assessments)} assessments")

    # Create embedding engine
    print("\nStep 2: Initializing Gemini embedding engine...")
    engine = GeminiEmbeddingEngine()

    # Build index
    print("\nStep 3: Generating Gemini embeddings and building FAISS index...")
    print("(This will make API calls to Gemini - may take 1-2 minutes)")
    engine.build_index(assessments)

    print("\n" + "=" * 70)
    print("GEMINI INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(assessments)} assessments")
    print(f"Index saved to: data/embeddings/faiss_index_gemini.bin")
    print(f"Metadata saved to: data/embeddings/metadata_gemini.json")
    print("\nReady for Gemini-powered retrieval!")


if __name__ == "__main__":
    main()
