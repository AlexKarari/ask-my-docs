"""
Reranker module.

Problem:
- Vector search is good, but not perfect.
- It may retrieve chunks that are "close" semantically, but not the best answer.

Solution:
- Retrieve top-k candidates
- Ask an LLM model to intelligently select relevant chunks for answering the query to significantly improve correctness
"""

from typing import Any, Dict, List
from openai import OpenAI

client = OpenAI()

def rerank_with_llm(
    query: str,
    candidates: List[Dict[str, Any]], # chunks retrieved from vector search
    model: str,
    keep_n: int,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using an LLM.

    Output is the selected top 'keep_n' candidates, in best-first order.

    Implementation approach:
    - Provide the query and numbered candidate summaries
    - Ask the model to return a comma-separated list of best indices
    """
    if not candidates: # if no candidates are retrieved
        return []

    # Create a compact view of candidates
    # Truncate long chunks to reduce token usage during reranking

    numbered = []
    for i, c in enumerate(candidates, start=1):
        preview = c["text"][:600].replace("\n", " ")
        numbered.append(f"{i}. ({c['source']}) {preview}")

    prompt = f"""
You are a retrieval reranker.
Given the user question and a list of candidate passages, select the passages
most useful for answering the question.

Rules:
- Pick the most relevant passages for answering.
- Prefer passages that directly contain facts needed to answer.
- Output ONLY a comma-separated list of numbers (e.g., "2,5,1").

QUESTION:
{query}

CANDIDATES:
{chr(10).join(numbered)} 
""".strip()

    # call the LLM
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,  # deterministic rerank
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract LLM response
    raw = resp.choices[0].message.content.strip()

    # Parse indices safely. LLMs may fail to follow instructions safely
    chosen = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(candidates):
                chosen.append(idx)

    # Remove duplicates while preserving order
    seen = set()
    final_indices = []
    for idx in chosen:
        if idx not in seen:
            seen.add(idx)
            final_indices.append(idx)

    # If the model returns nothing usable, fall back to original order
    if not final_indices:
        return candidates[:keep_n]

    reranked = [candidates[i - 1] for i in final_indices]
    return reranked[:keep_n]
