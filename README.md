# HealthierYou Bariatric Clinic RAG Assistant

An AI-powered knowledge assistant that provides accurate, citation-backed answers about HealthierYou Bariatric Clinic services, pricing, onboarding, and policies using Retrieval-Augmented Generation (RAG).

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/xanderKariuki/healthieryou-rag)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Note: This application is hosted on the Hugging Face Spaces CPU Basic (Free) tier. If the app appears unavailable or displays a scheduling/runtime error, it is usually due to temporary resource allocation on the shared free infrastructure rather than an issue with the application itself. Please try again in a few minutes or refresh the page.

## project Overview
HealthierYou RAG is an AI system that allows users to ask natural language questions over a curated knowledge base (KB) related to a bariatric and weight-loss clinic.

Instead of relying purely on a language model’s memory, this system:
1. Retrieves relevant documents from a vector database.
2. Augments the prompt with those documents.
3. Generates grounded, explainable answers using an LLM.

This ensures:
1. Higher factual accuracy
2. Reduced hallucinations
3. Domain-specific intelligence

## ✨ Features

- 🔍 **Semantic Search** - Understands intent, not just keywords
- 📚 **Source Citations** - Every answer includes references [1], [2]
- 🎯 **Confidence Gating** - Refuses to answer when information isn't available
- 🔄 **LLM Reranking** - Intelligently selects most relevant content
- 🐛 **Debug Mode** - Transparent retrieval and ranking insights

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/AlexKarari/ask-my-docs.git
cd ask-my-docs

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables by creating a .env file
OPENAI_API_KEY=your_api_key_here

# Build vector database
python ingest.py

# Run the app
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## 🏗️ How It Works

```
User Question
      ↓
1. Retrieve top-k candidates (vector search)
2. Confidence gate (refuse if similarity too low)
3. Rerank with LLM (select best 3 chunks)
4. Generate grounded answer with citations
      ↓
Answer + Sources + Debug Info
```

## 📁 Project Structure

```
healthieryou-rag/
├── app.py                # Gradio UI
├── ingest.py             # Document ingestion
├── requirements.txt      # Dependencies
├── core/                 # RAG pipeline modules
│   ├── config.py         # Configuration settings
│   ├── chunking.py       # Markdown-aware chunking
│   ├── embeddings.py     # OpenAI embeddings API
│   ├── store.py          # ChromaDB interface
│   ├── retriever.py      # Vector similarity search
│   ├── reranker.py       # LLM-based reranking
│   ├── generator.py      # Grounded answer generation
│   └── rag_pipeline.py   # End-to-end orchestration
└── kb/                   # Knowledge base (Markdown)
    ├── 01_about.md
    ├── 02_pricing.md
    ├── 03_onboarding.md
    ├── 04_data_privacy.md
    └── 05_support_faq.md
```

## ⚙️ Configuration

Key settings in `core/config.py`:

```python
CONFIG = {
    "retrieve_k": 12,              # Initial candidates from vector search
    "max_best_distance": 0.5,      # Confidence threshold (lower = stricter)
    "keep_n_after_rerank": 3,      # Final chunks sent to LLM
    "temperature": 0.0             # Generation randomness (0 = deterministic)
}
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **UI** | Gradio 6.5 |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Database** | ChromaDB |
| **LLM** | OpenAI GPT-4o-mini |
| **Deployment** | Hugging Face Spaces |

## 📝 Adding Documents

1. Add Markdown files to `kb/` folder (use numbering: `06_new_topic.md`)
2. Run `python ingest.py` to rebuild vector database
3. Restart the app

## 🌐 Deployment on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select SDK: **Gradio**
3. Add `OPENAI_API_KEY` in **Settings → Repository secrets**
4. Push your code - the Space auto-builds and deploys

## 🐛 Troubleshooting

**"I don't know based on the current knowledge base"**
- The question isn't covered in your KB documents
- Add relevant content to `kb/` and re-run `python ingest.py`

**App not starting locally**
- Verify `OPENAI_API_KEY` is set in `.env`
- Run `python ingest.py` to create vector database first

## 📚 Resources

- [Live Demo](https://huggingface.co/spaces/xanderKariuki/healthieryou-rag)
- [RAG Explained](https://docs.anthropic.com/en/docs/build-with-claude/rag)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## 📄 License

MIT License - see LICENSE file for details

---

Built with ❤️ | 😪 | 😓 to demonstrate production-ready RAG with semantic chunking, confidence gating, and grounded generation.

**Questions / comments / suggestions?** Open an issue or contact: karari.alexander@gmail.com
