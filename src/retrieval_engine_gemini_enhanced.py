""" Enhanced Gemini Retrieval Engine :  Uses rich text representation for better matching."""

import logging
import numpy as np
from typing import List, Dict
import google.generativeai as genai
import faiss
import os
from dotenv import load_dotenv
from src.embedding_engine_gemini_enhanced import EnhancedGeminiEmbeddingEngine

load_dotenv()

logger = logging.getLogger(__name__)


class EnhancedGeminiRetrievalEngine:
    """Enhanced retrieval with richer query representation."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "models/text-embedding-004",
        index_path: str = "data/embeddings/faiss_index_gemini_enhanced.bin",
        metadata_path: str = "data/embeddings/metadata_gemini_enhanced.json"
    ):
        """Initialize enhanced Gemini retrieval engine."""
        self.model_name = model_name
        
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        
        self.embedding_engine = EnhancedGeminiEmbeddingEngine(
            api_key=api_key,
            model_name=model_name,
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        self.load()
    
    def load(self) -> None:
        """Load enhanced index."""
        logger.info("Loading Enhanced Gemini retrieval engine...")
        self.embedding_engine.load()
        logger.info(f"Loaded {len(self.embedding_engine.metadata)} assessments")
    
    def expand_query(self, query: str) -> str:
        """
        Expand short query to match richer document representation.
        """
        query_lower = query.lower()
        
        # Add context based on query type
        expanded_parts = [query]
        
        # Technical queries
        if any(word in query_lower for word in ['python', 'java', 'javascript', 'developer', 'programmer', 'coding', 'sql']):
            expanded_parts.append("Technical skills assessment for programming and software development")
        
        # Personality/behavioral
        if any(word in query_lower for word in ['personality', 'leadership', 'communication', 'teamwork', 'behavioral']):
            expanded_parts.append("Personality and behavioral assessment for soft skills and work style")
        
        # Cognitive
        if any(word in query_lower for word in ['cognitive', 'aptitude', 'reasoning', 'analytical', 'problem solving']):
            expanded_parts.append("Cognitive ability and aptitude assessment for reasoning and analytical thinking")
        
        # Sales/Business
        if any(word in query_lower for word in ['sales', 'business', 'account', 'customer']):
            expanded_parts.append("Sales and business development assessment")
        
        # Add role context
        if 'senior' in query_lower or 'experienced' in query_lower:
            expanded_parts.append("For experienced professionals and senior-level candidates")
        elif 'junior' in query_lower or 'graduate' in query_lower or 'entry' in query_lower:
            expanded_parts.append("For entry-level candidates and new graduates")
        
        expanded_query = '. '.join(expanded_parts)
        
        # Log expansion
        if len(expanded_query) > len(query):
            logger.info(f"Query expanded: {len(query)} → {len(expanded_query)} chars")
        
        return expanded_query
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with expansion."""
        # Expand query for better matching
        expanded_query = self.expand_query(query)
        
        result = genai.embed_content(
            model=self.model_name,
            content=expanded_query,
            task_type="retrieval_query"
        )
        
        embedding = np.array([result['embedding']], dtype=np.float32)
        faiss.normalize_L2(embedding)
        
        return embedding
    
    def detect_query_intent(self, query: str) -> Dict[str, float]:
        """Detect query intent."""
        query_lower = query.lower()
        
        k_keywords = ['python', 'java', 'sql', 'programming', 'coding', 'technical', 
                      'developer', 'engineer', 'skills', 'javascript', 'react', 'node']
        p_keywords = ['personality', 'leadership', 'communication', 'teamwork', 
                      'behavioral', 'culture', 'soft skills', 'sales', 'manager']
        c_keywords = ['cognitive', 'aptitude', 'reasoning', 'numerical', 'verbal', 
                      'logical', 'analytical', 'problem solving']
        
        k_score = sum(1 for kw in k_keywords if kw in query_lower)
        p_score = sum(1 for kw in p_keywords if kw in query_lower)
        c_score = sum(1 for kw in c_keywords if kw in query_lower)
        
        total = k_score + p_score + c_score + 1e-6
        
        return {
            'K': k_score / total,
            'P': p_score / total,
            'C': c_score / total
        }
    
    def balance_results(self, results: List[Dict], intent: Dict[str, float], k: int = 10) -> List[Dict]:
        """Balance results by test type."""
        by_type = {'K': [], 'P': [], 'C': [], 'A': []}
        for r in results:
            test_type = r.get('test_type', 'A')
            by_type[test_type].append(r)
        
        if intent['K'] > 0.5:
            target = {'K': 6, 'P': 3, 'C': 1}
        elif intent['P'] > 0.5:
            target = {'K': 2, 'P': 6, 'C': 2}
        elif intent['C'] > 0.5:
            target = {'K': 2, 'P': 2, 'C': 6}
        else:
            target = {'K': 4, 'P': 4, 'C': 2}
        
        balanced = []
        for test_type, count in target.items():
            balanced.extend(by_type[test_type][:count])
        
        remaining = k - len(balanced)
        if remaining > 0:
            all_remaining = [r for r in results if r not in balanced]
            balanced.extend(all_remaining[:remaining])
        
        return balanced[:k]
    
    def recommend(
        self, 
        query: str, 
        k: int = 10,
        k_candidates_multiplier: int = 3,
        balance_ratios: Dict = None,
        intent_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Get recommendations with enhanced matching and optimized parameters.
        
        Args:
            query: Search query
            k: Number of recommendations to return
            k_candidates_multiplier: Multiplier for initial candidates (k * multiplier)
            balance_ratios: Custom balance ratios for K/P/C types
            intent_threshold: Threshold for strong intent detection
        """
        query_embedding = self.embed_query(query)
        
        # Use optimized candidate count
        n_candidates = k * k_candidates_multiplier
        distances, indices = self.embedding_engine.index.search(query_embedding, n_candidates)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.embedding_engine.metadata):
                results.append(self.embedding_engine.metadata[idx])
        
        # Detect intent with custom threshold
        intent = self.detect_query_intent(query)
        
        # Use custom balance ratios if provided
        if balance_ratios:
            balanced = self.balance_results_optimized(
                results, intent, k, balance_ratios, intent_threshold
            )
        else:
            balanced = self.balance_results(results, intent, k)
        
        return balanced
    
    def balance_results_optimized(
        self, 
        results: List[Dict], 
        intent: Dict[str, float], 
        k: int,
        balance_ratios: Dict,
        intent_threshold: float
    ) -> List[Dict]:
        """Balance results with optimized ratios."""
        by_type = {'K': [], 'P': [], 'C': [], 'A': []}
        for r in results:
            test_type = r.get('test_type', 'A')
            by_type[test_type].append(r)
        
        # Determine query type with threshold
        if intent['K'] > intent_threshold:
            target_ratios = balance_ratios['k_heavy']
        elif intent['P'] > intent_threshold:
            target_ratios = balance_ratios['p_heavy']
        elif intent['C'] > intent_threshold:
            target_ratios = balance_ratios['c_heavy']
        else:
            target_ratios = balance_ratios['mixed']
        
        # Convert ratios to counts
        target = {
            'K': int(k * target_ratios['K']),
            'P': int(k * target_ratios['P']),
            'C': int(k * target_ratios['C'])
        }
        
        # Ensure we get exactly k results
        total = sum(target.values())
        if total < k:
            target['K'] += (k - total)
        
        balanced = []
        for test_type, count in target.items():
            balanced.extend(by_type[test_type][:count])
        
        # Fill remaining spots
        remaining = k - len(balanced)
        if remaining > 0:
            all_remaining = [r for r in results if r not in balanced]
            balanced.extend(all_remaining[:remaining])
        
        return balanced[:k]
