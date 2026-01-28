"""
Generator module.

This is the "answer" step that is enforced with strict grounding.

Key mechanism:
- The model must use ONLY the provided context.
- If context is insufficient, it must say "I don't know based on the KB."

This reduces hallucinations.
"""

from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()


def build_context_blocks(hits: List[Dict[str, Any]]) -> str:
    """
    Build a context string with numbered blocks so the model can cite them as [1], [2], etc.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] Source: {h['source']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_grounded_answer(
    query: str,
    context_blocks: str,
    model: str,
    temperature: float,
) -> str:
    """
    Generate an answer grounded ONLY in context.

    The prompt forces:
    - citations like [1]
    - refusal when answer isn't in context
    """
    system = (
        "You are a careful assistant. Answer ONLY using the provided context. "
        "If the answer is not explicitly supported by the context, say:\n"
        "\"I don't know based on the current knowledge base.\"\n"
        "You MUST cite sources using [1], [2], etc. "
        "Do not invent policies, numbers, dates, or rules."
    )

    user = f"""
QUESTION:
{query}

CONTEXT:
{context_blocks}

INSTRUCTIONS:
- Provide a concise answer.
- Every factual statement must include at least one citation like [1].
- If you cannot find support in context, refuse.
""".strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return resp.choices[0].message.content.strip()
