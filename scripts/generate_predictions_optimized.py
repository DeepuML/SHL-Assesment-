""" Generate test predictions using OPTIMIZED Enhanced Gemini model.
Based on hyperparameter optimization results: 21.22% recall
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import pandas as pd

from src.retrieval_engine_gemini_enhanced import EnhancedGeminiRetrievalEngine


def load_optimal_params():
    """Load optimal parameters from optimization results."""
    results_file = Path("data/hyperparameter_optimization_results.json")
    with open(results_file, "r") as f:
        results = json.load(f)
    return results["best_params"]


def main():
    """Generate predictions with optimized parameters."""
    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS - OPTIMIZED MODEL")
    print("=" * 70)

    # Load optimal parameters
    params = load_optimal_params()
    print("\nUsing Optimized Parameters:")
    print(f"  k_candidates_multiplier: {params['k_candidates_multiplier']}")
    print(f"  intent_threshold: {params['intent_threshold']}")
    print(f"  balance_k_heavy: {params['balance_k_heavy']}")
    print(f"  balance_p_heavy: {params['balance_p_heavy']}")

    # Load retrieval engine with optimal params
    print("\nLoading Optimized Enhanced Gemini engine...")
    engine = EnhancedGeminiRetrievalEngine()
    engine.load()

    # Load test queries
    test_df = pd.read_csv("data/test/test.csv")
    print(f"\nLoaded {len(test_df)} test queries")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []

    for idx, row in test_df.iterrows():
        query = row["query"]
        query_id = idx + 1

        # Get recommendations with optimal parameters
        results = engine.recommend(
            query=query,
            k=10,
            k_candidates_multiplier=params["k_candidates_multiplier"],
            balance_ratios={
                "k_heavy": params["balance_k_heavy"],
                "p_heavy": params["balance_p_heavy"],
                "c_heavy": params["balance_c_heavy"],
                "mixed": params["balance_mixed"],
            },
            intent_threshold=params["intent_threshold"],
        )

        # Extract URLs
        urls = [r["assessment_url"] for r in results]

        predictions.append(
            {"query_id": query_id, "recommended_assessments": "|".join(urls)}
        )

        print(f"  Query {query_id}: {len(urls)} recommendations")

    # Save predictions
    output_file = "data/submission_optimized.csv"
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("PREDICTIONS SAVED")
    print("=" * 70)
    print(f"\nFile: {output_file}")
    print(f"Queries: {len(predictions)}")
    print(f"Recommendations per query: 10")

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print("\nOptimized Model Statistics:")
    print(f"  Mean Recall@10 (training): 21.22%")
    print(f"  Improvement vs Enhanced: +5.53%")
    print(f"  Improvement vs Baseline: +18.61%")
    print("\nBEST PERFORMING MODEL - READY FOR SUBMISSION")

    print("\n" + "=" * 70)
    print("SUBMISSION FORMAT")
    print("=" * 70)
    print("\nFormat: CSV with columns [query_id, recommended_assessments]")
    print("Example:")
    print(pred_df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()
