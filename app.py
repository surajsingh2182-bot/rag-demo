import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from pathlib import Path
from pypdf import PdfReader

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Demo", page_icon="🔍", layout="centered")
st.title("🔍 RAG Demo")
st.caption("Ask questions about the docs. Powered by Groq + FAISS + sentence-transformers.")

# ── Load embedding model (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Helper: split text into overlapping chunks ────────────────────────────────
def _chunk_text(text: str, documents: list, chunk_size=500, overlap=50):
    words = text.split()
    for start in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            documents.append(chunk)

# ── Index all .txt and .pdf files in docs/ (cached) ───────────────────────────
@st.cache_resource(show_spinner="Indexing documents...")
def build_index(_embedder):
    docs_path = Path("docs")
    documents = []

    # Load .txt files
    for file in sorted(docs_path.glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        _chunk_text(text, documents)

    # Load .pdf files
    for file in sorted(docs_path.glob("*.pdf")):
        reader = PdfReader(str(file))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        _chunk_text(text, documents)

    embeddings = _embedder.encode(documents, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine after normalize
    index.add(embeddings)

    return index, documents

# ── Groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return Groq(api_key=api_key)

embedder = load_embedder()
index, documents = build_index(embedder)
groq_client = get_groq_client()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── RAG query function ────────────────────────────────────────────────────────
def ask_rag(question: str) -> str:
    # Embed and normalize query
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    # Retrieve top-4 chunks
    _, indices = index.search(q_vec, k=4)
    chunks = [documents[i] for i in indices[0] if i < len(documents)]

    if not chunks:
        return "I couldn't find relevant information in the docs."

    context = "\n\n---\n\n".join(chunks)

    # Call Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer using ONLY the provided context. "
                    "If the answer isn't in the context, say so clearly. Be concise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating..."):
            answer = ask_rag(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
