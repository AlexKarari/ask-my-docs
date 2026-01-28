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

"""
RAG pipeline module.

This file implements the "end-to-end" logic in one callable function.
The UI calls this. It returns:
- answer
- sources
- debug details (retrieval + reranking) so you can inspect what's happening.

Mechanisms included:
1) Retrieve candidates
2) Confidence gate (refuse if retrieval looks weak)
3) Rerank candidates
4) Generate grounded answer with citations
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


def run_rag(query: str) -> Dict[str, Any]:
    """
    Run the full RAG pipeline.

    Returns a dict containing:
        - answer: model output
        - sources: unique source filenames used
        - debug: retrieval/rerank diagnostics
    """
    # 1) Retrieve candidates
    candidates = retrieve_candidates(
        query=query,
        db_dir=CONFIG.db_dir,
        collection_name=CONFIG.collection_name,
        embedding_model=CONFIG.embedding_model,
        k=CONFIG.retrieve_k,
    )

    # 2) Confidence gate
    allowed, gate_reason = confidence_gate(candidates, CONFIG.max_best_distance)
    if not allowed:
        return {
            "answer": (
                "I don't know based on the current knowledge base.\n\n"
                f"Reason: {gate_reason}\n"
                "Tip: Add or improve a KB document that covers this topic."
            ),
            "sources": [],
            "debug": {
                "gate_reason": gate_reason,
                "retrieved": candidates,
                "reranked": [],
            },
        }

    # 3) Rerank
    reranked = rerank_with_llm(
        query=query,
        candidates=candidates,
        model=CONFIG.chat_model,
        keep_n=CONFIG.keep_n_after_rerank,
    )

    # 4) Generate grounded answer
    context = build_context_blocks(reranked)
    answer = generate_grounded_answer(
        query=query,
        context_blocks=context,
        model=CONFIG.chat_model,
        temperature=CONFIG.temperature,
    )

    # Unique sources used
    sources = []
    for h in reranked:
        if h["source"] not in sources:
            sources.append(h["source"])

    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "gate_reason": gate_reason,
            "retrieved": candidates,
            "reranked": reranked,
        },
    }

