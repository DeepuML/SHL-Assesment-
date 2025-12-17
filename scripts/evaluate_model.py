"""Evaluate Model Script`=====================
Run evaluation on labeled training set.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval_engine import RetrievalEngine
from src.evaluation import Evaluator


def main():
    """Run evaluation."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Load retrieval engine
    print("\nLoading retrieval engine...")
    engine = RetrievalEngine()
    engine.load()
    
    # Create evaluator
    evaluator = Evaluator(engine)
    
    # Run evaluation
    print("\nRunning evaluation on training set...")
    results = evaluator.evaluate("data/train/train.csv", k=10)
    
    # Save results
    evaluator.save_results(results, "data/evaluation_results.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mean Recall@10: {results['mean_recall']:.4f}")
    print(f"Queries evaluated: {results['num_queries']}")
    print("\nDetailed results saved to: data/evaluation_results.json")


if __name__ == "__main__":
    main()
