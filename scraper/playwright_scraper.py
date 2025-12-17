"""
Playwright-based SHL Scraper
Uses browser automation to handle dynamic content
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = Path("data/shl_catalog.json")

def classify_test_type(text):
    """Classify assessment type from text"""
    text_lower = text.lower()
    
    k_kw = ['skill', 'technical', 'programming', 'java', 'python', 'sql', 'coding', 'developer']
    c_kw = ['cognitive', 'ability', 'reasoning', 'numerical', 'verbal', 'aptitude']
    p_kw = ['personality', 'behavior', 'motivat', 'competenc', 'leadership']
    
    k_score = sum(1 for kw in k_kw if kw in text_lower)
    c_score = sum(1 for kw in c_kw if kw in text_lower)
    p_score = sum(1 for kw in p_kw if kw in text_lower)
    
    if k_score > c_score and k_score > p_score:
        return 'K'
    elif c_score > p_score:
        return 'C'
    else:
        return 'P'

def scrape_assessment_details(page, url):
    """Scrape details from individual assessment page"""
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        time.sleep(2)
        
        soup = BeautifulSoup(page.content(), "html.parser")
        
        # Extract name
        name = None
        for tag in soup.find_all(['h1', 'h2']):
            text = tag.get_text(strip=True)
            if text and len(text) > 3:
                name = text
                break
        
        if not name:
            name = url.split('/')[-2].replace('-', ' ').title()
        
        # Extract description
        description = ""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            description = meta.get('content', '')[:500]
        
        # Get full text for classification
        full_text = soup.get_text()
        test_type = classify_test_type(full_text)
        
        return {
            'assessment_name': name,
            'assessment_url': url,
            'description': description,
            'test_type': test_type,
            'category': '',
            'skills': '',
            'job_roles': ''
        }
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None

def scrape():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    # Load existing assessments
    existing = []
    seen_urls = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            seen_urls = {a['assessment_url'] for a in existing}
            print(f"Loaded {len(existing)} existing assessments\n")
    
    assessments = existing.copy()
    all_links = set()

    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Step 1: Collect all assessment URLs
        print("\nStep 1: Collecting assessment URLs...\n")
        
        for type_param in [1, 2, None]:
            start = 0
            type_str = f"type={type_param}" if type_param else "no-type"
            print(f"Collecting from {type_str}...")
            
            while start < 600:  # Max 50 pages
                if type_param:
                    url = f"{BASE_URL}?start={start}&type={type_param}"
                else:
                    url = f"{BASE_URL}?start={start}"
                
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    time.sleep(2)
                    
                    soup = BeautifulSoup(page.content(), "html.parser")
                    
                    # Find assessment links
                    links = soup.find_all('a', href=True)
                    page_links = set()
                    
                    for link in links:
                        href = link.get('href', '')
                        if '/product-catalog/view/' in href:
                            if href.startswith('http'):
                                full_url = href
                            else:
                                full_url = 'https://www.shl.com' + href
                            
                            page_links.add(full_url.split('?')[0])
                    
                    if not page_links:
                        print(f"  {type_str}: No more results at page {start//12 + 1}")
                        break
                    
                    all_links.update(page_links)
                    print(f"  Page {start//12 + 1}: +{len(page_links)} (Total: {len(all_links)})")
                    
                    start += 12
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  Error at {url}: {e}")
                    break
            
            print(f"{type_str}: Collected {len(all_links)} total unique URLs\n")
            time.sleep(2)

        # Step 2: Scrape details for new URLs
        new_urls = [url for url in all_links if url not in seen_urls]
        print(f"\nStep 2: Scraping {len(new_urls)} new assessments...\n")
        
        for idx, url in enumerate(new_urls, 1):
            print(f"[{idx}/{len(new_urls)}] {url}")
            
            details = scrape_assessment_details(page, url)
            if details:
                assessments.append(details)
                print(f"  {details['assessment_name']} (Type: {details['test_type']})")
                
                # Save after each
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(assessments, f, indent=2, ensure_ascii=False)
                
                if idx % 10 == 0:
                    print(f"\n  Progress saved: {len(assessments)} total\n")
            
            time.sleep(1.5)

        browser.close()

    # Final report
    from collections import Counter
    types = Counter([a['test_type'] for a in assessments])
    
    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total assessments: {len(assessments)}")
    print(f"Type distribution: {dict(types)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    
    if len(assessments) >= 377:
        print("SUCCESS: Reached 377+ assessments!")
    else:
        print(f"Warning: {len(assessments)} assessments (need 377+)")

if __name__ == "__main__":
    scrape()
