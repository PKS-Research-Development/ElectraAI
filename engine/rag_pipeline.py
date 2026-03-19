# ============================================================
# ElectraAI — RAG Pipeline (Updated)
# Loads BOTH website JSON and PDF JSON
# Merges into single ChromaDB vector store
# ============================================================

import json
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# ── Configuration ────────────────────────────────────────────
WEBSITE_DATA  = os.path.join(os.path.dirname(__file__),
                "../data/raw/website_data.json")
PDF_DATA      = os.path.join(os.path.dirname(__file__),
                "../data/raw/pdf_data.json")
CHROMA_DIR    = os.path.join(os.path.dirname(__file__),
                "../database/chroma_db")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
HF_TOKEN      = os.getenv("HF_TOKEN")

# ── Step 1: Load Data Sources ────────────────────────────────
def load_data():
    all_data = []

    # Load website data
    if os.path.exists(WEBSITE_DATA):
        with open(WEBSITE_DATA, "r", encoding="utf-8") as f:
            website_pages = json.load(f)
        # Normalise format
        for page in website_pages:
            page['type'] = page.get('type', 'website')
        all_data.extend(website_pages)
        print(f"🌐 Website pages loaded:  {len(website_pages)}")
    else:
        print(f"⚠️  No website data found at {WEBSITE_DATA}")

    # Load PDF data
    if os.path.exists(PDF_DATA):
        with open(PDF_DATA, "r", encoding="utf-8") as f:
            pdf_pages = json.load(f)
        all_data.extend(pdf_pages)
        print(f"📄 PDF pages loaded:      {len(pdf_pages)}")
    else:
        print(f"⚠️  No PDF data found at {PDF_DATA}")
        print(f"   Run pdf_loader.py first if you have PDFs")

    print(f"📦 Total pages to process: {len(all_data)}\n")
    return all_data

# ── Step 2: Convert to LangChain Documents ──────────────────
def prepare_documents(data):
    print("📝 Preparing documents...")
    documents = []
    for page in data:
        doc = Document(
            page_content=page["content"],
            metadata={
                "url":    page.get("url", ""),
                "type":   page.get("type", "website"),
                "source": page.get("source", page.get("url", "")),
                "page":   str(page.get("page_number", ""))
            }
        )
        documents.append(doc)
    print(f"✅ Prepared {len(documents)} documents\n")
    return documents

# ── Step 3: Split Into Chunks ────────────────────────────────
def chunk_documents(documents):
    print(f"✂️  Chunking documents...")
    print(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # Count by source type
    web_chunks = len([c for c in chunks
                      if c.metadata.get('type') == 'website'])
    pdf_chunks = len([c for c in chunks
                      if c.metadata.get('type') == 'pdf'])

    print(f"✅ Total chunks:    {len(chunks)}")
    print(f"   Website chunks: {web_chunks}")
    print(f"   PDF chunks:     {pdf_chunks}\n")
    return chunks

# ── Step 4: Embed & Store in ChromaDB ───────────────────────
def create_vector_store(chunks, source_type="all"):
    print(f"🧠 Creating embeddings using HuggingFace API...")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   This may take a few minutes...\n")

    embeddings = HuggingFaceEndpointEmbeddings(
        model=f"https://router.huggingface.co/hf-inference/models/"
              f"{EMBED_MODEL}/pipeline/feature-extraction",
        huggingfacehub_api_token=HF_TOKEN
    )

    # Check if ChromaDB already exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"📂 Existing ChromaDB found — appending new chunks...")

        # Load existing vectorstore
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        # Check what's already stored to avoid duplicates
        existing = vectorstore.get()
        existing_urls = set()
        if existing and existing['metadatas']:
            existing_urls = set(
                m.get('url', '') for m in existing['metadatas']
            )
            print(f"   Already stored: {len(existing_urls)} unique sources")

        # Filter out chunks that are already in the DB
        new_chunks = [
            c for c in chunks
            if c.metadata.get('url', '') not in existing_urls
        ]
        duplicate_chunks = len(chunks) - len(new_chunks)

        if duplicate_chunks > 0:
            print(f"   ⏭️  Skipping {duplicate_chunks} duplicate chunks")

        if not new_chunks:
            print(f"   ✅ Nothing new to add — DB is up to date!")
            return vectorstore

        print(f"   ➕ Adding {len(new_chunks)} new chunks...\n")
        vectorstore.add_documents(new_chunks)
        print(f"✅ Chunks appended to existing database!")

    else:
        print(f"📂 No existing ChromaDB — creating fresh database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print(f"✅ Fresh vector database created!")

    print(f"💾 Saved to: {CHROMA_DIR}\n")
    return vectorstore

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("⚡ ElectraAI — RAG Pipeline")
    print("=" * 60 + "\n")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    data        = load_data()
    if not data:
        print("❌ No data to process. Run scraper.py or pdf_loader.py first.")
        exit()

    documents   = prepare_documents(data)
    chunks      = chunk_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("=" * 60)
    print("🎉 RAG Pipeline Complete!")
    print(f"📦 Total chunks stored: {len(chunks)}")
    print(f"🗄️  Database: {CHROMA_DIR}")
    print("=" * 60)