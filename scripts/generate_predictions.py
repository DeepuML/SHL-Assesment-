"""
Generate Predictions Script
===========================
Generate predictions for test set and create submission CSV.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval_engine import RetrievalEngine


def main():
    """Generate predictions."""
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)
    
    # Load retrieval engine
    print("\nLoading retrieval engine...")
    engine = RetrievalEngine()
    engine.load()
    
    # Load test set
    test_file = "data/test/test.csv"
    print(f"\nLoading test queries from: {test_file}")
    
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test queries")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        query_id = idx + 1  # 1-indexed
        query = row['query']
        
        # Get recommendations
        recommendations = engine.recommend(query, k=10)
        
        # Extract URLs
        recommended_urls = [rec['assessment_url'] for rec in recommendations]
        
        # Format as comma-separated string
        recommended_str = ",".join(recommended_urls)
        
        predictions.append({
            'query_id': query_id,
            'recommended_assessments': recommended_str
        })
        
        print(f"  Query {query_id}: {len(recommendations)} recommendations")
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Save to CSV
    output_file = "data/submission.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("PREDICTIONS GENERATED")
    print("=" * 70)
    print(f"Saved to: {output_file}")
    print(f"Total queries: {len(predictions)}")
    print("\nSubmission format:")
    print(df.head())


if __name__ == "__main__":
    main()
