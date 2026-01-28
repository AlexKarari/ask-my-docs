"""
Ingestion script.

Run this to build/update your vector DB from kb/*.md.

Flow:
1) Read markdown files from kb/
2) Chunk using markdown-aware chunker
3) Embed chunks
4) Store in Chroma

Re-run whenever KB changes.
"""

from pathlib import Path
from dotenv import load_dotenv

from core.config import CONFIG
from core.chunking import chunk_markdown
from core.embeddings import embed_texts
from core.store import get_or_create_collection

load_dotenv()

KB_DIR = Path("kb")


def read_kb_files():
    """
    Read all markdown files in kb/ folder.
    Returns list of (filename, text).
    """
    docs = []
    for fp in sorted(KB_DIR.glob("*.md")):
        docs.append((fp.name, fp.read_text(encoding="utf-8")))
    return docs


def main():
    col = get_or_create_collection(CONFIG.db_dir, CONFIG.collection_name)

    try:
        col.delete(where={})
    except Exception:
        pass

    docs = read_kb_files()

    all_chunks = []
    metadatas = []
    ids = []

    idx = 0
    for fname, text in docs:
        chunks = chunk_markdown(
            text,
            max_chars=CONFIG.max_chunk_chars,
            overlap=CONFIG.chunk_overlap_chars,
        )
        for c in chunks:
            ids.append(f"{fname}-{idx}")
            all_chunks.append(c)
            metadatas.append({"source": fname})
            idx += 1

    vectors = embed_texts(all_chunks, model=CONFIG.embedding_model)

    col.add(ids=ids, documents=all_chunks, metadatas=metadatas, embeddings=vectors)
    print(f"Ingested {len(all_chunks)} chunks into {CONFIG.db_dir}/ (collection={CONFIG.collection_name})")


if __name__ == "__main__":
    main()
