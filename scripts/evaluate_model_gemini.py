"""
Evaluate Gemini Model Script
=============================
Run evaluation on labeled training set using Gemini embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import Evaluator
from src.retrieval_engine_gemini import GeminiRetrievalEngine


def main():
    """Run evaluation with Gemini."""
    print("=" * 70)
    print("GEMINI MODEL EVALUATION")
    print("=" * 70)

    # Load Gemini retrieval engine
    print("\nLoading Gemini retrieval engine...")
    engine = GeminiRetrievalEngine()

    # Create evaluator
    evaluator = Evaluator(engine)

    # Run evaluation
    print("\nRunning evaluation on training set with Gemini embeddings...")
    results = evaluator.evaluate("data/train/train.csv", k=10)

    # Save results
    evaluator.save_results(results, "data/evaluation_results_gemini.json")

    # Print summary
    print("\n" + "=" * 70)
    print("GEMINI EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"Mean Recall@10: {results['mean_recall']:.4f} ({results['mean_recall']*100:.2f}%)"
    )
    print(f"Queries evaluated: {results['num_queries']}")
    print("\nDetailed results saved to: data/evaluation_results_gemini.json")

    # Compare with baseline if exists
    try:
        import json

        with open("data/evaluation_results.json", "r") as f:
            baseline = json.load(f)
        baseline_recall = baseline["mean_recall"]
        improvement = (
            (results["mean_recall"] - baseline_recall) / baseline_recall
        ) * 100

        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE (MiniLM)")
        print("=" * 70)
        print(f"Baseline (MiniLM):  {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
        print(
            f"Gemini:             {results['mean_recall']:.4f} ({results['mean_recall']*100:.2f}%)"
        )
        print(f"Improvement:        {improvement:+.1f}%")
    except:
        pass


if __name__ == "__main__":
    main()
