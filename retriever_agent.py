"""
Retriever Agent — NON-LLM
Performs FAISS vector search over locally indexed document chunks.
Returns top-k most relevant chunks for a given query.
"""

import numpy as np
from typing import List, Dict

from rag.document_processor import load_index, index_exists


def retrieve(query: str, top_k: int = 6) -> List[Dict]:
    """
    Search the FAISS index for chunks most relevant to `query`.
    Returns list of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
    """
    if not index_exists():
        raise RuntimeError("No index found. Please upload and index documents first.")

    index, meta, embedder = load_index()

    query_vec = embedder.transform([query])   # shape (1, dim)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text":   meta[idx]["text"],
            "source": meta[idx]["source"],
            "score":  float(score),   # higher = more similar (cosine)
        })

    # Sort descending — highest cosine similarity first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def format_context(retrieved_chunks: List[Dict]) -> str:
    """Format retrieved chunks into a single context string for LLM agents."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Chunk {i} | Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
