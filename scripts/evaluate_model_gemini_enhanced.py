"""
Evaluate Enhanced Gemini Model
===============================
Test improved performance with richer text.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval_engine_gemini_enhanced import EnhancedGeminiRetrievalEngine
from src.evaluation import Evaluator


def main():
    """Evaluate enhanced Gemini model."""
    print("=" * 70)
    print("ENHANCED GEMINI MODEL EVALUATION")
    print("=" * 70)
    
    print("\nLoading Enhanced Gemini retrieval engine...")
    engine = EnhancedGeminiRetrievalEngine()
    
    evaluator = Evaluator(engine)
    
    print("\nRunning evaluation with ENHANCED embeddings...")
    print("(Richer text: 300-500 words vs 50-100)")
    results = evaluator.evaluate("data/train/train.csv", k=10)
    
    evaluator.save_results(results, "data/evaluation_results_gemini_enhanced.json")
    
    print("\n" + "=" * 70)
    print("ENHANCED GEMINI EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mean Recall@10: {results['mean_recall']:.4f} ({results['mean_recall']*100:.2f}%)")
    print(f"Queries evaluated: {results['num_queries']}")
    
    # Compare with both baseline and original Gemini
    try:
        import json
        
        with open("data/evaluation_results.json", 'r') as f:
            baseline = json.load(f)
        baseline_recall = baseline['mean_recall']
        
        try:
            with open("data/evaluation_results_gemini.json", 'r') as f:
                gemini = json.load(f)
            gemini_recall = gemini['mean_recall']
            has_gemini = True
        except:
            has_gemini = False
        
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"MiniLM (Baseline):        {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
        
        if has_gemini:
            print(f"Gemini (Original):        {gemini_recall:.4f} ({gemini_recall*100:.2f}%)")
        
        print(f"Gemini (Enhanced):        {results['mean_recall']:.4f} ({results['mean_recall']*100:.2f}%)")
        
        improvement = ((results['mean_recall'] - baseline_recall) / baseline_recall) * 100
        print(f"\nImprovement vs Baseline:  {improvement:+.1f}%")
        
        if has_gemini:
            gemini_improvement = ((results['mean_recall'] - gemini_recall) / gemini_recall) * 100
            print(f"Improvement vs Original:  {gemini_improvement:+.1f}%")
        
    except Exception as e:
        print(f"\nNote: Could not load comparison data: {e}")
    
    print("\nDetailed results saved to: data/evaluation_results_gemini_enhanced.json")


if __name__ == "__main__":
    main()
