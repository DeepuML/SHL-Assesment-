"""Check for remaining emojis in project files."""
import re
from pathlib import Path

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000027BF"  # misc symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\u2700-\u27BF"  # dingbats
    "\u2600-\u26FF"  # misc symbols
    "]+", 
    flags=re.UNICODE
)

project_root = Path(__file__).parent.parent
found_emojis = []

# Check .py and .md files
for ext in ['*.py', '*.md']:
    for file_path in project_root.rglob(ext):
        # Skip venv and hidden folders
        if any(part.startswith('.') or part == 'venv' for part in file_path.parts):
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8')
            matches = emoji_pattern.findall(content)
            if matches:
                found_emojis.append((str(file_path.relative_to(project_root)), matches))
        except Exception as e:
            pass

if found_emojis:
    print("Found emojis in the following files:")
    for file, emojis in found_emojis:
        print(f"\n{file}:")
        print(f"  Emojis: {set(emojis)}")
else:
    print("No emojis found in the project!")
