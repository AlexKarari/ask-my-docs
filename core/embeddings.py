"""
Embeddings module.

Embeddings converts text into a list of numbers (vectors) that represent the text's meaning.
"""

from typing import List
from openai import OpenAI

client = OpenAI()

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Embed a list of texts.

    Args:
        text (List[str]): list of strings to embed
        model (str): embedding model name

    Returns:
        List[List[float]]: list of embedding vectors (each a list of floats)
    """
    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in resp.data]

def embed_query(query: str, model: str) -> List[float]:
    """Embed a single query string

    Returns:
        embedding vector for the query
    """
    return embed_texts([query], model=model)
