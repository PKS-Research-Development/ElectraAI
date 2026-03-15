# ============================================================
# ElectraAI — Chatbot UI
# Streamlit-based chat interface powered by Mistral + ChromaDB
# ============================================================

import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ── Configuration ────────────────────────────────────────────
CHROMA_DIR  = os.path.join(os.path.dirname(__file__),
              "../database/chroma_db")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "mistral"
TOP_K       = 3  # Number of relevant chunks to retrieve

# ── Load Vector Database ─────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

# ── Load LLM ─────────────────────────────────────────────────
@st.cache_resource
def load_llm():
    return OllamaLLM(model=LLM_MODEL)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ElectraAI",
    page_icon="⚡",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("⚡ ElectraAI")
st.caption("Smart AI-powered search engine for your content")
st.divider()

# ── Load Resources ────────────────────────────────────────────
with st.spinner("🔌 Starting ElectraAI engine..."):
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(
                    search_kwargs={"k": TOP_K})
    llm         = load_llm()

# ── Chat History ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm ElectraAI. Ask me anything about the content I've learned!"
    })

# ── Display Chat History ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask ElectraAI anything..."):

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve relevant chunks from ChromaDB
    docs = retriever.invoke(prompt)
    context = "\n\n".join([
        f"Source {i+1}: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # Build prompt for Mistral
    full_prompt = f"""You are ElectraAI, a smart and helpful assistant.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say 'I don't have that information.'
Keep your answer clear and concise.

Context:
{context}

Question: {prompt}

Answer:"""

    # Get response from Mistral
    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            response = llm.invoke(full_prompt)
            st.write(response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("⚡ Powered by Mistral 7B + ChromaDB + LangChain")