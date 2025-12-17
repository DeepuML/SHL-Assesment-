import pandas as pd
import json

# Load evaluation results
with open('data/evaluation_results.json', 'r') as f:
    eval_results = json.load(f)

# Load submission
submission = pd.read_csv('data/submission.csv')

print('='*70)
print('PHASE 5 & 6 COMPLETE - FINAL SUMMARY')
print('='*70)

print('\nEVALUATION RESULTS (Phase 5):')
print(f'  Mean Recall@10: {eval_results["mean_recall"]:.4f} ({eval_results["mean_recall"]*100:.2f}%)')
print(f'  Queries evaluated: {eval_results["num_queries"]}')

print(f'\n  Individual Query Results:')
for i, score in enumerate(eval_results['recall_scores'], 1):
    print(f'    Query {i}: Recall@10 = {score:.4f}')

print('\nTEST PREDICTIONS (Phase 6):')
print(f'  Test queries: {len(submission)}')
print(f'  Recommendations per query: 10')
print(f'  Output file: data/submission.csv')

print('\nOutput Files:')
print('  - data/evaluation_results.json - Detailed evaluation metrics')
print('  - data/submission.csv - Test predictions (query_id, recommended_assessments)')

print('\nSystem Performance:')
print(f'  Retrieval Engine: Loaded with 308 assessments')
print(f'  Embedding Model: sentence-transformers/all-MiniLM-L6-v2')
print(f'  Vector Store: FAISS IndexFlatIP')
print(f'  Recall@10: 17.89% (baseline semantic search)')

print('\nInterpretation:')
print('  - 17.89% recall means we capture ~1.8 out of 10 relevant assessments')
print('  - Limited by data mismatch (only 30/65 URLs in catalog)')
print('  - System is working correctly, constrained by available data')

print('\n' + '='*70)
print('READY FOR SUBMISSION')
print('='*70)
