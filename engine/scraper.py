# ============================================================
# ElectraAI — Web Scraper (Improved)
# Crawls books.toscrape.com with clean text extraction
# ============================================================

import requests
import json
import os
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Configuration ────────────────────────────────────────────
BASE_URL    = "http://books.toscrape.com"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__),
              "../data/raw/website_data.json")
MAX_PAGES   = 50

# ── Clean Text ───────────────────────────────────────────────
def clean_text(text):
    # Fix encoding issues (Â£ → £)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.replace('Â£', '£').replace('Ã', '').replace('\xa0', ' ')
    # Remove extra whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)

# ── Extract Meaningful Content Only ─────────────────────────
def extract_content(soup, url):
    # Remove navigation, header, footer noise
    for tag in soup(['nav', 'header', 'footer',
                     'script', 'style', 'noscript']):
        tag.decompose()

    content_parts = []

    # Extract page title
    title = soup.find('h1')
    if title:
        content_parts.append(f"Page: {title.get_text(strip=True)}")

    # Extract book details if on a product page
    if '/catalogue/' in url and 'category' not in url:
        # Book title
        book_title = soup.find('h1')
        if book_title:
            content_parts.append(f"Book Title: {book_title.get_text(strip=True)}")

        # Price
        price = soup.find('p', class_='price_color')
        if price:
            price_text = clean_text(price.get_text(strip=True))
            content_parts.append(f"Price: {price_text}")

        # Availability
        availability = soup.find('p', class_='availability')
        if availability:
            content_parts.append(
                f"Availability: {availability.get_text(strip=True)}")

        # Rating
        rating_tag = soup.find('p', class_='star-rating')
        if rating_tag:
            rating = rating_tag['class'][1] if rating_tag else 'Unknown'
            content_parts.append(f"Rating: {rating} stars")

        # Description
        description = soup.find('div', id='product_description')
        if description:
            desc_text = description.find_next_sibling('p')
            if desc_text:
                content_parts.append(
                    f"Description: {desc_text.get_text(strip=True)}")

        # Product info table
        table = soup.find('table', class_='table-striped')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                header = row.find('th')
                value  = row.find('td')
                if header and value:
                    content_parts.append(
                        f"{header.get_text(strip=True)}: "
                        f"{clean_text(value.get_text(strip=True))}")

    # For category pages — extract book listings
    elif 'category' in url or url == BASE_URL:
        articles = soup.find_all('article', class_='product_pod')
        for article in articles:
            book_name  = article.find('h3')
            book_price = article.find('p', class_='price_color')
            book_avail = article.find('p', class_='availability')
            book_rating = article.find('p', class_='star-rating')

            if book_name and book_price:
                name    = book_name.get_text(strip=True)
                price   = clean_text(book_price.get_text(strip=True))
                avail   = book_avail.get_text(strip=True) if book_avail else 'Unknown'
                rating  = book_rating['class'][1] if book_rating else 'Unknown'
                content_parts.append(
                    f"Book: {name} | Price: {price} | "
                    f"Rating: {rating} stars | {avail}")

    return '\n'.join(content_parts)

# ── Scraper ──────────────────────────────────────────────────
def scrape_website(base_url, max_pages=MAX_PAGES):
    visited  = set()
    all_data = []
    to_visit = [base_url]

    print(f"🌐 Starting scrape: {base_url}")
    print(f"📄 Max pages: {max_pages}\n")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, verify=False, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            content = extract_content(soup, url)

            if content.strip():
                all_data.append({"url": url, "content": content})
                print(f"✅ Scraped ({len(visited)}/{max_pages}): {url}")

            # Find links on same domain
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    if full_url not in visited:
                        to_visit.append(full_url)

        except Exception as e:
            print(f"❌ Failed: {url} → {e}")

    return all_data

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data = scrape_website(BASE_URL)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Scraping complete!")
    print(f"📦 Total pages scraped: {len(data)}")
    print(f"💾 Saved to: {OUTPUT_FILE}")