"""
Vector store module (Chroma).

PersistentClient is used so that the DB. persists locally under vectordb/.
"""

import chromadb

def get_or_create_collection(db_dir: str, collection_name: str):
    chroma = chromadb.PersistentClient(path=db_dir)
    return chroma.get_or_create_collection(name=collection_name)

def get_collection(db_dir: str, collection_name: str):
    chroma = chromadb.PersistentClient(path=db_dir)
    return chroma.get_collection(name=collection_name)
