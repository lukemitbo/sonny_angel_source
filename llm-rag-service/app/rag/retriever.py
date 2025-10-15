from typing import List, Tuple, Dict, Any, Protocol

import numpy as np

from .embedding import STEmbedder
from .vector_store import LocalFaissVectorStoreManager


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        ...


class SimpleRetriever:
    """Retriever that embeds the query and looks up nearest texts."""

    def __init__(self, vector_store: LocalFaissVectorStoreManager, embedder: STEmbedder | None = None):
        self.vector_store = vector_store
        self.embedder = embedder or vector_store.embedder

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        q = self.embedder.embed([query])
        distances, indices = self.vector_store.search(q, k)
        idxs = indices[0]
        out: List[Tuple[str, Dict[str, Any]]] = []
        for ix in idxs:
            if ix == -1:
                continue
            rec = self.vector_store.get_record(ix)
            out.append((rec.text, rec.meta))
        return out


