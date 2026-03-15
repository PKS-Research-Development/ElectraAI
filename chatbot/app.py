# ============================================================
# ElectraAI — Chatbot UI
# Supports both Local (Ollama) and Cloud (Groq) modes
# ============================================================

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# ── Configuration ────────────────────────────────────────────
CHROMA_DIR   = os.path.join(os.path.dirname(__file__),
               "../database/chroma_db")
EMBED_MODEL  = "nomic-embed-text"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Detect Mode: Local or Cloud ──────────────────────────────
def get_llm():
    if GROQ_API_KEY:
        # Cloud mode — use Groq
        print("🌐 Using Groq Cloud LLM")
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"
        )
    else:
        # Local mode — use Ollama
        print("💻 Using Ollama Local LLM")
        return OllamaLLM(model="mistral")

# ── Load Vector Database ─────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

# ── Load LLM ─────────────────────────────────────────────────
@st.cache_resource
def load_llm():
    return get_llm()

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ElectraAI",
    page_icon="⚡",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("⚡ ElectraAI")
st.caption("Smart AI-powered search engine for your content")

# ── Mode Indicator ────────────────────────────────────────────
if GROQ_API_KEY:
    st.success("🌐 Running in Cloud Mode (Groq)")
else:
    st.info("💻 Running in Local Mode (Ollama)")

st.divider()

# ── Load Resources ────────────────────────────────────────────
with st.spinner("🔌 Starting ElectraAI engine..."):
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(
                    search_kwargs={"k": 3})
    llm         = load_llm()

# ── Chat History ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
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
    docs    = retriever.invoke(prompt)
    context = "\n\n".join([
        f"Source {i+1}: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # Build prompt for LLM
    full_prompt = f"""You are ElectraAI, a smart and helpful assistant.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say 'I don't have that information.'
Keep your answer clear and concise.

Context:
{context}

Question: {prompt}

Answer:"""

    # Get response from LLM
    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            if GROQ_API_KEY:
                # Groq returns message object
                response = llm.invoke(full_prompt)
                answer   = response.content
            else:
                # Ollama returns string directly
                answer = llm.invoke(full_prompt)
            st.write(answer)

    # Save response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ── Footer ────────────────────────────────────────────────────
st.divider()
mode = "Groq Cloud" if GROQ_API_KEY else "Ollama Local"
st.caption(f"⚡ Powered by {mode} + ChromaDB + LangChain")