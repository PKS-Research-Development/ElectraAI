# ============================================================
# ElectraAI — RAG Pipeline
# Loads scraped data, chunks it, creates embeddings,
# and stores everything in ChromaDB vector database
# ============================================================

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ── Configuration ────────────────────────────────────────────
DATA_FILE   = os.path.join(os.path.dirname(__file__),
              "../data/raw/website_data.json")
CHROMA_DIR  = os.path.join(os.path.dirname(__file__),
              "../database/chroma_db")
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

# ── Step 1: Load Scraped Data ────────────────────────────────
def load_data(filepath):
    print(f"📂 Loading scraped data from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} pages\n")
    return data

# ── Step 2: Convert to Documents ────────────────────────────
def prepare_documents(data):
    print("📝 Preparing documents...")
    documents = []
    for page in data:
        doc = Document(
            page_content=page["content"],
            metadata={"url": page["url"]}
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
    print(f"✅ Created {len(chunks)} chunks\n")
    return chunks

# ── Step 4: Create Embeddings & Store in ChromaDB ───────────
def create_vector_store(chunks):
    print(f"🧠 Creating embeddings using: {EMBED_MODEL}")
    print(f"   This may take a few minutes...\n")
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Vector database created!")
    print(f"💾 Saved to: {CHROMA_DIR}\n")
    return vectorstore

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("⚡ ElectraAI — RAG Pipeline Starting...")
    print("=" * 50 + "\n")

    # Ensure database directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Run pipeline
    data       = load_data(DATA_FILE)
    documents  = prepare_documents(data)
    chunks     = chunk_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("=" * 50)
    print("🎉 RAG Pipeline Complete!")
    print(f"📦 Total chunks stored: {len(chunks)}")
    print(f"🗄️  Database location: {CHROMA_DIR}")
    print("=" * 50)