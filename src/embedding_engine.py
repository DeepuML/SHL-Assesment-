""" Embedding Engine : Generate embeddings and build FAISS index for semantic search."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generate embeddings and manage FAISS index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/embeddings/faiss_index.bin",
        metadata_path: str = "data/embeddings/metadata.json",
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: SentenceTransformer model name
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.assessments = []

    def load_model(self):
        """Load SentenceTransformer model."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(
            f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def create_document_text(self, assessment: Dict) -> str:
        """
        Create rich text representation for embedding.

        Args:
            assessment: Assessment dictionary

        Returns:
            Combined text for embedding
        """
        parts = []

        # Assessment name (most important)
        if assessment.get("assessment_name"):
            parts.append(assessment["assessment_name"])

        # Test type
        test_type_map = {
            "K": "Knowledge and Skills",
            "P": "Personality and Behavior",
            "C": "Cognitive Ability",
            "A": "Assessment",
        }
        test_type = assessment.get("test_type", "A")
        parts.append(test_type_map.get(test_type, "Assessment"))

        # Category
        if assessment.get("category"):
            parts.append(assessment["category"])

        # Description
        if assessment.get("description"):
            parts.append(assessment["description"])

        # Skills
        if assessment.get("skills"):
            parts.append(f"Skills: {assessment['skills']}")

        # Job roles
        if assessment.get("job_roles"):
            parts.append(f"Roles: {assessment['job_roles']}")

        return " ".join(parts)

    def generate_embeddings(
        self, assessments: List[Dict], batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for all assessments.
        Args:
            assessments: List of assessment dictionaries
            batch_size: Batch size for encoding
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(assessments)} assessments...")

        # Create document texts
        texts = [self.create_document_text(a) for a in assessments]

        # Generate embeddings
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings.
        Args:
            embeddings: Numpy array of embeddings
        """
        logger.info("Building FAISS index...")

        dimension = embeddings.shape[1]

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        logger.info(f"Saved FAISS index to: {self.index_path}")

        # Save metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {self.metadata_path}")

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        logger.info(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.assessments = json.load(f)

        logger.info(
            f"Loaded index with {self.index.ntotal} vectors and {len(self.assessments)} assessments"
        )

    def process(self, assessments_file: str):
        """Main processing workflow.
        Args:
            assessments_file: Path to processed assessments JSON
        """
        logger.info("=" * 60)
        logger.info("EMBEDDING GENERATION STARTED")
        logger.info("=" * 60)

        # Load assessments
        logger.info(f"Loading assessments from: {assessments_file}")
        with open(assessments_file, "r", encoding="utf-8") as f:
            self.assessments = json.load(f)
        logger.info(f"Loaded {len(self.assessments)} assessments")

        # Load model
        self.load_model()

        # Generate embeddings
        embeddings = self.generate_embeddings(self.assessments)

        # Build index
        self.build_index(embeddings)

        # Save
        self.save_index()

        logger.info("=" * 60)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 60)


def main():
    """Run embedding generation."""
    engine = EmbeddingEngine()
    engine.process("data/processed/shl_assessments_clean.json")


if __name__ == "__main__":
    main()
