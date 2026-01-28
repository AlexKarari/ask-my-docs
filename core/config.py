"""
Central configuration for the RAG application.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    # --- Storage ---
    db_dir: str = "vectordb"
    collection_name: str = "HealthierYou_kb"

    # --- Embeddings ---
    # This is comes from OpenAI's Embeddings API - Converts text to vectors
    embedding_model: str = "text-embedding-3-small"

    # --- LLM ---
    # gpt-4o-mini is one of OpenAI's cheap reasoning model optimized for Chat, RAG answering etc
    # It is a general-purpose language model that reads retrieved text and writes answers
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.2

    # --- Retrieval ---
    # Retrieve more than will be used, as reranking works with more candidates
    retrieve_k: int = 12

    # After reranking, keep only the top N chunks as context for the answer
    keep_n_after_rerank: int = 5

    # --- Confidence gating ---
    # Use this to evaluate the quality of retrieved information
    # Or the model's own answers to ensure they meet a certain threshold of reliability
    # Chroma returns distances (smaller distance = more similar).
    # If the best (smallest) distance is too large, retrieval is likely weak.
    # IMPORTANT: This value is dataset-dependent.
    # Start here, then adjust after testing.
    max_best_distance: float = 0.85

    # -- Chunking --
    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 200

CONFIG = RAGConfig()
