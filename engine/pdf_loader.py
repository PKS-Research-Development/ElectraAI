# ============================================================
# ElectraAI — PDF Loader
# Reads all PDFs from data/pdfs/ folder
# Extracts clean text page by page
# Saves to data/raw/pdf_data.json
# ============================================================

import os
import json
import re
from pypdf import PdfReader

# ── Configuration ────────────────────────────────────────────
PDF_FOLDER   = os.path.join(os.path.dirname(__file__),
               "../data/pdfs")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__),
               "../data/raw/pdf_data.json")

# ── Clean Extracted Text ─────────────────────────────────────
def clean_text(text):
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)
    # Fix common PDF extraction artifacts
    text = text.replace('\x00', '')
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words
    return text.strip()

# ── Extract Text from Single PDF ────────────────────────────
def extract_pdf(filepath):
    filename = os.path.basename(filepath)
    print(f"\n📄 Processing: {filename}")

    try:
        reader   = PdfReader(filepath)
        pages    = []
        total_pages = len(reader.pages)
        print(f"   Total pages: {total_pages}")

        for page_num, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text()
                text     = clean_text(raw_text)

                if len(text) < 50:  # Skip near-empty pages
                    print(f"   ⏭️  Page {page_num+1} skipped (too short)")
                    continue

                pages.append({
                    "source":      filename,
                    "page_number": page_num + 1,
                    "total_pages": total_pages,
                    "url":         f"pdf://{filename}#page={page_num+1}",
                    "content":     text,
                    "type":        "pdf"
                })
                print(f"   ✅ Page {page_num+1}/{total_pages} extracted "
                      f"({len(text)} chars)")

            except Exception as e:
                print(f"   ❌ Page {page_num+1} failed: {e}")
                continue

        return pages

    except Exception as e:
        print(f"   ❌ Failed to read PDF: {e}")
        return []

# ── Process All PDFs in Folder ───────────────────────────────
def process_all_pdfs(pdf_folder):
    # Check folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ PDF folder not found: {pdf_folder}")
        print(f"   Create it and add PDFs: mkdir -p {pdf_folder}")
        return []

    # Find all PDFs
    pdf_files = [
        f for f in os.listdir(pdf_folder)
        if f.lower().endswith('.pdf')
    ]

    if not pdf_files:
        print(f"⚠️  No PDFs found in: {pdf_folder}")
        print(f"   Drop your PDF files into: {pdf_folder}")
        return []

    print(f"📂 Found {len(pdf_files)} PDF(s) in {pdf_folder}")
    print(f"   Files: {', '.join(pdf_files)}\n")

    all_pages = []
    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        pages    = extract_pdf(filepath)
        all_pages.extend(pages)

    return all_pages

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("⚡ ElectraAI — PDF Loader")
    print("=" * 60)

    # Extract all PDFs
    all_pages = process_all_pdfs(PDF_FOLDER)

    if not all_pages:
        print("\n⚠️  No content extracted. Add PDFs and try again.")
        exit()

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    # Summary
    pdfs_processed = len(set(p['source'] for p in all_pages))
    print(f"\n{'=' * 60}")
    print(f"🎉 PDF Extraction Complete!")
    print(f"📚 PDFs processed:   {pdfs_processed}")
    print(f"📄 Pages extracted:  {len(all_pages)}")
    print(f"💾 Saved to:         {OUTPUT_FILE}")
    print(f"{'=' * 60}")