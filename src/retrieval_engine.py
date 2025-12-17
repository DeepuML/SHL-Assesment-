"""
Retrieval Engine
================
Semantic search with intelligent test-type balancing.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Semantic retrieval with balancing logic."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/embeddings/faiss_index.bin",
        metadata_path: str = "data/embeddings/metadata.json"
    ):
        """
        Initialize retrieval engine.
        
        Args:
            model_name: SentenceTransformer model name
            index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.assessments = []
        
        # Auto-load on initialization
        self.load()
        
    def load(self):
        """Load model, index, and metadata."""
        logger.info("Loading retrieval engine...")
        
        # Load model
        self.model = SentenceTransformer(self.model_name)
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        
        logger.info(f"Loaded {len(self.assessments)} assessments")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query to embedding vector.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding[0]
    
    def search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of (index, score) tuples
        """
        query_embedding = self.encode_query(query)
        
        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        return list(zip(indices[0], scores[0]))
    
    def detect_query_intent(self, query: str) -> Dict[str, float]:
        """
        Detect whether query needs technical (K), behavioral (P), or cognitive (C) assessments.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with intent scores
        """
        query_lower = query.lower()
        
        # Technical/Knowledge keywords
        technical_keywords = [
            'developer', 'engineer', 'programmer', 'coding', 'software',
            'technical', 'skill', 'python', 'java', 'sql', 'programming',
            'data', 'analyst', 'accountant', 'financial'
        ]
        
        # Behavioral/Personality keywords
        behavioral_keywords = [
            'leadership', 'team', 'communication', 'personality', 'behavior',
            'soft skill', 'motivation', 'culture fit', 'work style', 'manager'
        ]
        
        # Cognitive keywords
        cognitive_keywords = [
            'reasoning', 'problem solving', 'cognitive', 'ability', 'aptitude',
            'analytical', 'logical', 'numerical', 'verbal'
        ]
        
        # Count matches
        k_score = sum(1 for kw in technical_keywords if kw in query_lower)
        p_score = sum(1 for kw in behavioral_keywords if kw in query_lower)
        c_score = sum(1 for kw in cognitive_keywords if kw in query_lower)
        
        total = k_score + p_score + c_score
        
        if total == 0:
            # Default: balanced
            return {'K': 0.4, 'P': 0.4, 'C': 0.2}
        
        return {
            'K': k_score / total,
            'P': p_score / total,
            'C': c_score / total
        }
    
    def balance_results(
        self,
        results: List[Tuple[int, float]],
        query_intent: Dict[str, float],
        final_k: int = 10
    ) -> List[Dict]:
        """
        Balance results by test type based on query intent.
        
        Args:
            results: List of (index, score) tuples
            query_intent: Query intent scores
            final_k: Number of final recommendations
            
        Returns:
            Balanced list of assessments
        """
        # Group results by test type
        by_type = {'K': [], 'P': [], 'C': [], 'A': []}
        
        for idx, score in results:
            assessment = self.assessments[idx].copy()
            assessment['_score'] = float(score)
            assessment['_index'] = idx
            test_type = assessment.get('test_type', 'A')
            by_type[test_type].append(assessment)
        
        # Determine target counts based on intent
        if query_intent['K'] > 0.5 or query_intent['P'] > 0.5:
            # Strong intent for one type
            if query_intent['K'] > query_intent['P']:
                target_counts = {'K': int(final_k * 0.6), 'P': int(final_k * 0.3), 'C': int(final_k * 0.1)}
            else:
                target_counts = {'K': int(final_k * 0.3), 'P': int(final_k * 0.6), 'C': int(final_k * 0.1)}
        else:
            # Balanced query
            target_counts = {'K': int(final_k * 0.4), 'P': int(final_k * 0.4), 'C': int(final_k * 0.2)}
        
        # Select assessments
        balanced_results = []
        
        for test_type in ['K', 'P', 'C', 'A']:
            target = target_counts.get(test_type, 0)
            available = by_type[test_type]
            
            # Take top N by score
            selected = sorted(available, key=lambda x: x['_score'], reverse=True)[:target]
            balanced_results.extend(selected)
        
        # If we don't have enough, add more from highest scoring
        if len(balanced_results) < final_k:
            all_results = [self.assessments[idx].copy() for idx, score in results]
            for r, (idx, score) in zip(all_results, results):
                r['_score'] = float(score)
            
            # Add missing ones
            existing_ids = {r['assessment_url'] for r in balanced_results}
            for r in sorted(all_results, key=lambda x: x['_score'], reverse=True):
                if r['assessment_url'] not in existing_ids:
                    balanced_results.append(r)
                    if len(balanced_results) >= final_k:
                        break
        
        # Sort by score and take top k
        balanced_results = sorted(balanced_results, key=lambda x: x['_score'], reverse=True)[:final_k]
        
        # Remove internal fields
        for r in balanced_results:
            r.pop('_score', None)
            r.pop('_index', None)
        
        return balanced_results
    
    def recommend(self, query: str, k: int = 10) -> List[Dict]:
        """
        Get recommendations for a query.
        
        Args:
            query: Search query
            k: Number of recommendations
            
        Returns:
            List of recommended assessments
        """
        # Detect query intent
        intent = self.detect_query_intent(query)
        logger.info(f"Query intent: K={intent['K']:.2f}, P={intent['P']:.2f}, C={intent['C']:.2f}")
        
        # Search
        results = self.search(query, k=min(50, k * 5))
        
        # Balance results
        recommendations = self.balance_results(results, intent, final_k=k)
        
        return recommendations


def main():
    """Test retrieval engine."""
    engine = RetrievalEngine()
    engine.load()
    
    # Test query
    query = "Python developer with leadership skills"
    recommendations = engine.recommend(query, k=10)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['assessment_name']} (Type: {rec['test_type']})")
        print(f"   URL: {rec['assessment_url']}")


if __name__ == "__main__":
    main()
