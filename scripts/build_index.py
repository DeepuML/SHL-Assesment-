"""
Build Index Script
==================
Process data and build FAISS index.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor import DataProcessor
from src.embedding_engine import EmbeddingEngine


def main():
    """Main workflow."""
    print("=" * 70)
    print("BUILDING EMBEDDINGS AND FAISS INDEX")
    print("=" * 70)
    
    # Step 1: Process data
    print("\nStep 1: Processing scraped data...")
    processor = DataProcessor(
        input_file="data/shl_catalog.json",
        output_file="data/processed/shl_assessments_clean.json"
    )
    assessments = processor.process()
    
    # Step 2: Generate embeddings and build index
    print("\nStep 2: Generating embeddings and building FAISS index...")
    engine = EmbeddingEngine()
    engine.process("data/processed/shl_assessments_clean.json")
    
    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(assessments)} assessments")
    print("Ready for retrieval!")


if __name__ == "__main__":
    main()
