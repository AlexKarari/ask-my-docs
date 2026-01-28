"""
RAG pipeline module.

This file implements the "end-to-end" logic in one callable function. 

The UI calls this. It returns:
- answer
- sources
- debug details (retrieval + reranking) for one to inspect what is happening
"""

from typing import Any, Dict, List, Tuple
from .config import CONFIG
from .retriever import retrieve_candidates
from .reranker import rerank_with_llm
from .generator import build_context_blocks, generate_grounded_answer

def confidence_gate(candidates: List[Dict[str, Any]], max_best_distance: float) -> Tuple[bool, str]:
    """
    Determine if we trust retrieval enough to answer.

    In Chroma distances:
    - smaller distance means more similarity
    - if even the best match has a high distance, we likely don't have the answer in the KB

    Returns:
        (allowed, reason)
    """
    if not candidates:
        return False, "No candidates retrieved."

    best = min(c["distance"] for c in candidates)
    if best > max_best_distance:
        return (
            False,
            f"Low retrieval confidence (best distance={best:.3f} > {max_best_distance}).",
        )

    return True, f"Retrieval looks OK (best distance={best:.3f})."
