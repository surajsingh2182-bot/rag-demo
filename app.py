import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Demo", page_icon="🔍", layout="centered")
st.title("🔍 RAG Demo")
st.caption("Ask questions about the docs. Powered by Groq + ChromaDB + sentence-transformers.")

# ── Load models & index docs (cached — runs once) ────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Indexing documents...")
def build_index(_embedder):
    client = chromadb.Client()  # in-memory for cloud deployment
    collection = client.get_or_create_collection("docs")

    docs_path = Path("docs")
    documents, ids = [], []

    for i, file in enumerate(sorted(docs_path.glob("*.txt"))):
        text = file.read_text(encoding="utf-8").strip()
        # Simple fixed-size chunking with overlap
        chunk_size, overlap = 500, 50
        words = text.split()
        chunks = []
        for start in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[start:start + chunk_size])
            if chunk:
                chunks.append(chunk)
        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{file.stem}_chunk_{j}")

    if documents:
        embeddings = _embedder.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeddings, ids=ids)

    return collection

embedder = load_embedder()
collection = build_index(embedder)

# ── Groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return Groq(api_key=api_key)

groq_client = get_groq_client()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Query function ────────────────────────────────────────────────────────────
def ask_rag(question: str) -> str:
    # 1. Embed question
    q_embedding = embedder.encode(question).tolist()

    # 2. Retrieve top-k chunks
    results = collection.query(query_embeddings=[q_embedding], n_results=4)
    chunks = results["documents"][0]

    if not chunks:
        return "I couldn't find relevant information in the docs."

    context = "\n\n---\n\n".join(chunks)

    # 3. Augment prompt and call LLM
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the answer isn't in the context, say so clearly. "
        "Be concise and direct."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # free, fast
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
