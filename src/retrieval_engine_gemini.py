""" Gemini Retrieval Engine:Semantic search using Gemini embeddings + FAISS."""

import logging
import os
from typing import Dict, List

import faiss
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

from src.embedding_engine_gemini import GeminiEmbeddingEngine

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiRetrievalEngine:
    """Retrieval engine using Gemini embeddings."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "models/text-embedding-004",
        index_path: str = "data/embeddings/faiss_index_gemini.bin",
        metadata_path: str = "data/embeddings/metadata_gemini.json",
    ):
        """
        Initialize Gemini retrieval engine.

        Args:
            api_key: Gemini API key
            model_name: Gemini embedding model
            index_path: Path to FAISS index
            metadata_path: Path to metadata
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Configure Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)

        self.embedding_engine = GeminiEmbeddingEngine(
            api_key=api_key,
            model_name=model_name,
            index_path=index_path,
            metadata_path=metadata_path,
        )

        # Auto-load on initialization
        self.load()

    def load(self) -> None:
        """Load FAISS index and metadata."""
        logger.info("Loading Gemini retrieval engine...")
        self.embedding_engine.load()
        logger.info(f"Loaded {len(self.embedding_engine.metadata)} assessments")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using Gemini API.

        Args:
            query: Query string

        Returns:
            Query embedding vector
        """
        result = genai.embed_content(
            model=self.model_name, content=query, task_type="retrieval_query"
        )

        # Convert to numpy and normalize
        embedding = np.array([result["embedding"]], dtype=np.float32)
        faiss.normalize_L2(embedding)

        return embedding

    def detect_query_intent(self, query: str) -> Dict[str, float]:
        """
        Detect query intent for K/P/C distribution.

        Args:
            query: Query string

        Returns:
            Intent scores dict
        """
        query_lower = query.lower()

        # Keywords for each type
        k_keywords = [
            "python",
            "java",
            "sql",
            "programming",
            "coding",
            "technical",
            "developer",
            "engineer",
            "skills",
            "javascript",
            "react",
            "node",
        ]
        p_keywords = [
            "personality",
            "leadership",
            "communication",
            "teamwork",
            "behavioral",
            "culture",
            "soft skills",
            "sales",
            "manager",
        ]
        c_keywords = [
            "cognitive",
            "aptitude",
            "reasoning",
            "numerical",
            "verbal",
            "logical",
            "analytical",
            "problem solving",
        ]

        # Count matches
        k_score = sum(1 for kw in k_keywords if kw in query_lower)
        p_score = sum(1 for kw in p_keywords if kw in query_lower)
        c_score = sum(1 for kw in c_keywords if kw in query_lower)

        # Normalize
        total = k_score + p_score + c_score + 1e-6

        return {"K": k_score / total, "P": p_score / total, "C": c_score / total}

    def balance_results(
        self, results: List[Dict], intent: Dict[str, float], k: int = 10
    ) -> List[Dict]:
        """
        Balance results by test type based on intent.

        Args:
            results: List of retrieved assessments
            intent: Intent scores
            k: Number of results to return

        Returns:
            Balanced results
        """
        # Separate by type
        by_type = {"K": [], "P": [], "C": [], "A": []}
        for r in results:
            test_type = r.get("test_type", "A")
            by_type[test_type].append(r)

        # Determine distribution
        if intent["K"] > 0.5:
            # Technical query
            target = {"K": 6, "P": 3, "C": 1}
        elif intent["P"] > 0.5:
            # Behavioral query
            target = {"K": 2, "P": 6, "C": 2}
        elif intent["C"] > 0.5:
            # Cognitive query
            target = {"K": 2, "P": 2, "C": 6}
        else:
            # Balanced
            target = {"K": 4, "P": 4, "C": 2}

        # Select from each type
        balanced = []
        for test_type, count in target.items():
            balanced.extend(by_type[test_type][:count])

        # Fill remaining with any type
        remaining = k - len(balanced)
        if remaining > 0:
            all_remaining = [r for r in results if r not in balanced]
            balanced.extend(all_remaining[:remaining])

        return balanced[:k]

    def recommend(self, query: str, k: int = 10) -> List[Dict]:
        """
        Get recommendations for query.

        Args:
            query: Query string
            k: Number of recommendations

        Returns:
            List of recommended assessments
        """
        # Embed query
        query_embedding = self.embed_query(query)

        # Search FAISS (retrieve more for balancing)
        distances, indices = self.embedding_engine.index.search(query_embedding, k * 3)

        # Get results
        results = []
        for idx in indices[0]:
            if idx < len(self.embedding_engine.metadata):
                results.append(self.embedding_engine.metadata[idx])

        # Detect intent
        intent = self.detect_query_intent(query)

        # Balance results
        balanced = self.balance_results(results, intent, k)

        return balanced


if __name__ == "__main__":
    # Test Gemini retrieval engine
    logging.basicConfig(level=logging.INFO)

    engine = GeminiRetrievalEngine()

    # Test query
    results = engine.recommend("Python developer assessment", k=5)

    print(f"\nTop 5 recommendations:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['assessment_name']}")
