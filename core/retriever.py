"""
Retriever module.

- This retrieves candidate chunks from the vector DB.
- It retrieves more candidates (retrieve_k) because reranking works best when it has options to choose from.
"""

from typing import Any, Dict, List
from .embeddings import embed_query
from .store import get_collection

def retrieve_candidates(
    query: str,
    db_dir: str,
    collection_name: str,
    embedding_model: str,
    k: int,
) -> List[Dict[str, Any]]:
    """Retrieve top-k candidate chunks from the vector DB.

    Returns:
        List[Dict[str, Any]]: A list of dicts with keys: text, source, distance
    """
    col = get_collection(db_dir, collection_name)
    qvec = embed_query(query, model=embedding_model)

    # Query the Vector Database to provide vector similarity search functionality
    # .query() is a method of the ChromaDB collection class.
    # 
    res = col.query(
        query_embeddings=[qvec], # Vector to search for - wrapped in a list because ChromaDB supports batch queries. We search for one query at a time.
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "unkown"),
                "distance": dist
            }
        )
    return hits


