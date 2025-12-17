"""
Generate Gemini Predictions Script
===================================
Generate predictions for test set using Gemini embeddings.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval_engine_gemini import GeminiRetrievalEngine


def main():
    """Generate predictions using Gemini."""
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS WITH GEMINI")
    print("=" * 70)

    # Load Gemini retrieval engine
    print("\nLoading Gemini retrieval engine...")
    engine = GeminiRetrievalEngine()

    # Load test set
    test_file = "data/test/test.csv"
    print(f"\nLoading test queries from: {test_file}")

    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test queries")

    # Generate predictions
    print("\nGenerating Gemini-powered predictions...")
    predictions = []

    for idx, row in test_df.iterrows():
        query_id = idx + 1  # 1-indexed
        query = row["query"]

        # Get recommendations
        recommendations = engine.recommend(query, k=10)

        # Extract URLs
        urls = [r["assessment_url"] for r in recommendations]
        urls_str = ",".join(urls)

        predictions.append({"query_id": query_id, "recommended_assessments": urls_str})

        print(f"  Query {query_id}: 10 recommendations")

    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions)

    # Save
    output_file = "data/submission_gemini.csv"
    submission_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("GEMINI PREDICTIONS GENERATED")
    print("=" * 70)
    print(f"Saved to: {output_file}")
    print(f"Total queries: {len(submission_df)}")

    print("\nSubmission format:")
    print(submission_df.head())


if __name__ == "__main__":
    main()
