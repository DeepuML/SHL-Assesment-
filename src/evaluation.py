""" Evaluation Module : Compute Mean Recall@10 on labeled training set."""
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate recommendation quality using Recall@K."""
    
    def __init__(self, retrieval_engine):
        """ Initialize evaluator.
        Args:
            retrieval_engine: RetrievalEngine instance
        """
        self.engine = retrieval_engine
        
    def recall_at_k(self, recommended: List[str], relevant: List[str], k: int = 10) -> float:
        """ Compute Recall@K.   
        
        Args:
            recommended: List of recommended assessment URLs
            relevant: List of relevant assessment URLs
            k: K value for recall
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        # Normalize URLs (handle /solutions/products/ vs /products/)
        def normalize_url(url):
            return url.replace('/solutions/products/', '/products/')
        
        # Take top K recommendations
        recommended_k = set([normalize_url(url) for url in recommended[:k]])
        relevant_set = set([normalize_url(url) for url in relevant])
        
        # Intersection
        intersection = recommended_k & relevant_set
        
        # Recall = |intersection| / |relevant|
        recall = len(intersection) / len(relevant_set)
        
        return recall
    
    def evaluate(self, test_file: str, k: int = 10) -> Dict:
        """ Evaluate on labeled test set.
        Args:
            test_file: Path to test file with queries and labels
            k: K value for recall
        Returns:
            Evaluation metrics
        """
        logger.info("=" * 60)
        logger.info(f"EVALUATION STARTED (Recall@{k})")
        logger.info("=" * 60)
        
        # Load test data
        logger.info(f"Loading test data from: {test_file}")
        
        # Check file format (JSON or CSV)
        if test_file.endswith('.csv'):
            df = pd.read_csv(test_file)
            test_data = []
            for _, row in df.iterrows():
                relevant_urls = row['relevant_assessment_urls'].split('|')
                test_data.append({
                    'query': row['query'],
                    'relevant_urls': relevant_urls
                })
        else:
            # Assume format: {query_id, query, relevant_urls}
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} test queries")
        
        # Evaluate each query
        recall_scores = []
        results = []
        
        for item in test_data:
            query = item['query']
            relevant_urls = item['relevant_urls']
            
            # Get recommendations
            recommendations = self.engine.recommend(query, k=k)
            recommended_urls = [r['assessment_url'] for r in recommendations]
            
            # Compute recall
            recall = self.recall_at_k(recommended_urls, relevant_urls, k=k)
            recall_scores.append(recall)
            
            results.append({
                'query': query,
                'recall': recall,
                'recommended': recommended_urls[:k],
                'relevant': relevant_urls
            })
            
            logger.info(f"Query: {query[:50]}... | Recall@{k}: {recall:.4f}")
        
        # Compute mean recall
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        logger.info("=" * 60)
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"  Mean Recall@{k}: {mean_recall:.4f}")
        logger.info(f"  Queries evaluated: {len(recall_scores)}")
        logger.info("=" * 60)
        
        return {
            'mean_recall': mean_recall,
            'num_queries': len(recall_scores),
            'recall_scores': recall_scores,
            'detailed_results': results
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to: {output_file}")


def main():
    """Run evaluation."""
    from src.retrieval_engine import RetrievalEngine
    
    # Load retrieval engine
    engine = RetrievalEngine()
    engine.load()
    
    # Create evaluator
    evaluator = Evaluator(engine)
    
    # Evaluate
    results = evaluator.evaluate("data/train/train_set.json", k=10)
    
    # Save results
    evaluator.save_results(results, "data/evaluation_results.json")


if __name__ == "__main__":
    main()
