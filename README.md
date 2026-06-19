# RAG Demo

A minimal Retrieval-Augmented Generation app built with:
- **Groq** (free LLM API — Llama 3.1 8B)
- **ChromaDB** (in-memory vector DB)
- **sentence-transformers** (local embeddings, no API key)
- **Streamlit** (UI + free hosting)

## Deploy to Streamlit Cloud (free public URL)

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick your repo → `app.py`
3. In **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Get your free Groq API key at [console.groq.com](https://console.groq.com)
5. Click Deploy → your app gets a public URL like `https://yourname-rag-demo.streamlit.app`

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key_here   # or add to .env
streamlit run app.py
```

## Add your own docs

Drop `.txt` files into the `docs/` folder. The app auto-indexes them on startup.
