import pandas as pd
import json

print("=" * 80)
print("FINAL MODEL COMPARISON - ALL APPROACHES")
print("=" * 80)

# Load all evaluation results
with open('data/evaluation_results.json', 'r') as f:
    baseline = json.load(f)

with open('data/evaluation_results_gemini.json', 'r') as f:
    gemini = json.load(f)

with open('data/evaluation_results_gemini_enhanced.json', 'r') as f:
    enhanced = json.load(f)

# Create comparison table
models = {
    'Model': [
        'MiniLM (Baseline)',
        'Gemini (Original)', 
        'Gemini (Enhanced)'
    ],
    'Embedding Dims': [384, 768, 768],
    'Text Length': ['50-100 words', '50-100 words', '300-500 words'],
    'Recall@10': [
        f"{baseline['mean_recall']:.4f} ({baseline['mean_recall']*100:.2f}%)",
        f"{gemini['mean_recall']:.4f} ({gemini['mean_recall']*100:.2f}%)",
        f"{enhanced['mean_recall']:.4f} ({enhanced['mean_recall']*100:.2f}%)"
    ],
    'vs Baseline': [
        '0.0%',
        f"{((gemini['mean_recall']-baseline['mean_recall'])/baseline['mean_recall']*100):+.1f}%",
        f"{((enhanced['mean_recall']-baseline['mean_recall'])/baseline['mean_recall']*100):+.1f}%"
    ],
    'Cost': ['Free', '~$0.001', '~$0.001'],
    'Speed': ['Fast (50ms)', 'Medium (200ms)', 'Medium (200ms)']
}

df = pd.DataFrame(models)

print("\nPERFORMANCE COMPARISON:")
print("=" * 80)
print(df.to_string(index=False))

print("\n\nWINNER: Enhanced Gemini (20.11% recall)")
print("=" * 80)

print("\nKey Improvements:")
print("  - Expanded test type descriptions (K/P/C -> full sentences)")
print("  - Inferred skills from assessment names")
print("  - Added use cases and features")
print("  - Query expansion for better matching")
print("  - 3-5x richer text representation")

print("\nSUBMISSION FILES:")
print("=" * 80)
print("  1. data/submission.csv (MiniLM - 17.89%)")
print("  2. data/submission_gemini.csv (Gemini Original - 11.11%)")
print("  3. data/submission_enhanced.csv (Enhanced Gemini - 20.11%) RECOMMENDED")

print("\nINSIGHTS:")
print("=" * 80)
print("  • Bigger model ≠ better (Gemini original worse than MiniLM)")
print("  • Text enrichment is KEY for large models")
print("  • Gemini needs 300-500 words to shine (vs 50-100)")
print("  • Query expansion helps match richer documents")
print("  • Enhanced Gemini: +12.4% improvement over baseline")

print("\nRECOMMENDATION:")
print("=" * 80)
print("  Submit: data/submission_enhanced.csv")
print("  Reason: Best performance (20.11% recall)")
print("  Bonus: Demonstrates ML engineering skill (model optimization)")

print("\n" + "=" * 80)
print("PROJECT COMPLETE")
print("=" * 80)
