"""
Build Enhanced Gemini Index
============================
Richer text representation (300-500 words vs 50-100).
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding_engine_gemini_enhanced import EnhancedGeminiEmbeddingEngine


def main():
    """Build enhanced Gemini index with 3-5x richer text."""
    print("=" * 70)
    print("BUILDING ENHANCED GEMINI INDEX (RICH TEXT)")
    print("=" * 70)

    data_file = "data/shl_catalog.json"
    print(f"\nStep 1: Loading data from {data_file}...")

    with open(data_file, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"Loaded {len(assessments)} assessments")

    print("\nStep 2: Initializing Enhanced Gemini engine...")
    print("(Creates 3-5x richer text: 300-500 words vs 50-100)")
    engine = EnhancedGeminiEmbeddingEngine()

    print("\nEnhancements:")
    print("  - Expanded test type descriptions")
    print("  - Inferred skills from assessment names")
    print("  - Added use cases and features")
    print("  - Enriched job role context")
    print("  - Target: 300-500 words per document")

    print("\nStep 3: Building index with Gemini API...")
    print("(This may take 1-2 minutes)")
    engine.build_index(assessments)

    print("\n" + "=" * 70)
    print("ENHANCED GEMINI INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(assessments)} assessments")
    print("Files created:")
    print("  - data/embeddings/faiss_index_gemini_enhanced.bin")
    print("  - data/embeddings/metadata_gemini_enhanced.json")
    print("\nReady for improved retrieval!")


if __name__ == "__main__":
    main()
