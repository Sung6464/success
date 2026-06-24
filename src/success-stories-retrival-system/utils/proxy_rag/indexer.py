"""Index builder using PostgreSQL and pgvector."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import config, openai_client as llm
from .tree_builder import iter_nodes, load_tree
from .pg_vector_store import PgVectorStore


def _node_embedding_text(node: dict) -> str:
    body = node.get("text", "")[:4000]
    return f"{node['breadcrumb']}\n\n{body}".strip()


# Global PgVectorStore instance
_db_store = None

def get_db_store() -> PgVectorStore:
    global _db_store
    if _db_store is None:
        _db_store = PgVectorStore()
        _db_store.create_tables(config.EMBED_DIM)
    return _db_store


def index_document(doc_id: str, doc_metadata: dict | None = None) -> int:
    """(Re)index one document. Inserts structural nodes and embeddings into PostgreSQL."""
    tree = load_tree(doc_id)
    nodes = [n for n in iter_nodes(tree) if n.get("text") or n.get("images")]
    if not nodes:
        return 0

    texts = [_node_embedding_text(n) for n in nodes]
    vecs = np.array(llm.embed_texts(texts), dtype="float32")
    
    # L2 normalize embeddings for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1, norms)

    db = get_db_store()
    
    # Delete any existing entries for this doc
    db.delete_document(doc_id)

    # Insert new entries
    new_meta = []
    doc_meta = doc_metadata or {}
    for n in nodes:
        new_meta.append(
            {
                "doc_id": doc_id,
                "node_id": n["node_id"],
                "breadcrumb": n["breadcrumb"],
                "title": n["title"],
                "snippet": n["snippet"],
                "text": n["text"],
                "images": n["images"],
                "url": doc_meta.get("url") or doc_meta.get("source") or "",
                "download_url": doc_meta.get("download_url") or doc_meta.get("graph_url") or "",
                "category": doc_meta.get("category") or "general",
            }
        )

    db.insert_nodes(new_meta, vecs.tolist())
    return len(nodes)


class RetrievalStore:
    """Loaded PostgreSQL database store, ready for search."""

    def __init__(self):
        self.db = get_db_store()

    @property
    def ready(self) -> bool:
        # Database pool is initialized
        return self.db.pool is not None

    def search(self, qvec: np.ndarray, k: int):
        if not self.ready:
            return []
        
        # L2 normalize query vector
        qnorm = np.linalg.norm(qvec)
        qvec_norm = qvec / (qnorm if qnorm > 0 else 1.0)
        
        qvec_list = qvec_norm[0].tolist()
        return self.db.search(qvec_list, k)
