import pandas as pd
import json

# Read train.csv
train_df = pd.read_csv('data/train/train.csv')

# Read catalog
with open('data/shl_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)

catalog_urls = set([a['assessment_url'] for a in catalog])

# Check matches
print("Checking URL matches...\n")

for idx, row in train_df.iterrows():
    query = row['query'][:60]
    relevant_urls = row['relevant_assessment_urls'].split('|')
    
    matches = 0
    for url in relevant_urls:
        # Try exact match
        if url in catalog_urls:
            matches += 1
        # Try normalizing (remove /solutions/)
        normalized_url = url.replace('/solutions/products/', '/products/')
        if normalized_url in catalog_urls:
            matches += 1
    
    print(f"Query {idx+1}: {query}...")
    print(f"  Relevant URLs: {len(relevant_urls)}")
    print(f"  Matches in catalog: {matches}")
    print()

print("\n" + "="*60)
print("URL Format Analysis:")
print("="*60)
print(f"Train URL sample: {train_df['relevant_assessment_urls'].iloc[0].split('|')[0]}")
print(f"Catalog URL sample: {catalog[0]['assessment_url']}")
print("\nDifference: /solutions/products/ vs /products/")
