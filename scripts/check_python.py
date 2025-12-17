import json

with open('data/shl_catalog.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

python_assessments = [a for a in data if 'python' in a['assessment_name'].lower() or 'python' in a['description'].lower()]

print(f"Found {len(python_assessments)} Python-related assessments:\n")
for a in python_assessments:
    print(f"- {a['assessment_name']}")
    print(f"  URL: {a['assessment_url']}")
    print(f"  Type: {a['test_type']}")
    print()

# Also check for Node.js, Java, and other common programming languages
print("\n" + "="*50)
print("Other programming languages found:")
print("="*50)

languages = ['java', 'javascript', 'node', 'react', 'angular', 'c++', 'c#', 'ruby', 'php', 'swift']
for lang in languages:
    count = len([a for a in data if lang in a['assessment_name'].lower() or lang in a['description'].lower()])
    if count > 0:
        print(f"{lang.upper()}: {count} assessments")
