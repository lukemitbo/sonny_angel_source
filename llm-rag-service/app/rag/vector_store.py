from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Protocol

import faiss
import numpy as np
from tqdm import tqdm

from .embedding import STEmbedder
from .metastore import MetaStore, Record


class FaissStore:
    """
    Cosine similarity via IndexFlatIP on normalized vectors.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine when normalized

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32 and vectors.shape[1] == self.dim
        self.index.add(vectors)  # vectors must already be L2-normalized

    def search(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert vectors.dtype == np.float32 and vectors.shape[1] == self.dim
        return self.index.search(vectors, k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, dim_hint: Optional[int] = None) -> "FaissStore":
        if not path.exists():
            if dim_hint is None:
                raise ValueError(
                    "Index file not found and dim_hint not provided to create a new one."
                )
            store = cls(dim_hint)
            return store
        idx = faiss.read_index(str(path))
        store = cls(idx.d)
        store.index = idx
        return store


class VectorStoreManager(Protocol):
    def add_texts(self,
                  texts: List[str],
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  batch: int = 64) -> None:
        ...

    def search(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_record(self, idx: int) -> Record:
        ...

    def status(self) -> Dict[str, Any]:
        ...


class LocalFaissVectorStoreManager:
    """Simple manager that couples FAISS index and JSONL metastore.

    Responsible for persistence, adding texts, and vector search.
    """

    def __init__(self, index_dir: str, embedder: Optional[STEmbedder] = None):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.jsonl"
        self.embedder = embedder or STEmbedder()

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.metastore = MetaStore(self.meta_path)
        self.metastore.load()
        self.store = FaissStore.load(self.index_path, dim_hint=self.embedder.dim)

        if self.store.ntotal != len(self.metastore):
            raise RuntimeError(
                f"Index/metadata mismatch: faiss has {self.store.ntotal} vectors, "
                f"metadata has {len(self.metastore)} records. "
                f"Make sure you didn't manually edit files.")

    def add_texts(self,
                  texts: List[str],
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  batch: int = 64) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        assert len(texts) == len(metadatas)

        vectors_to_add: List[np.ndarray] = []
        records_to_add: List[Record] = []

        for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
            chunk = texts[i:i + batch]
            embs = self.embedder.embed(chunk)  # already normalized
            vectors_to_add.append(embs)
            for j, t in enumerate(chunk):
                records_to_add.append(Record(text=t, meta=metadatas[i + j]))

        if vectors_to_add:
            all_vecs = np.vstack(vectors_to_add).astype(np.float32, copy=False)
            self.store.add(all_vecs)
            self.metastore.append_many(records_to_add)
            self._persist()

    def search(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.store.search(vectors, k)

    def get_record(self, idx: int) -> Record:
        return self.metastore.get(idx)

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(self.index_path)

    def status(self) -> Dict[str, Any]:
        p = self.index_path if self.index_path else None
        return {
            "loaded": self.store.ntotal > 0,
            "ntotal": int(self.store.ntotal) if self.store.ntotal else 0,
            "dim": self.embedder.dim,
            "embedding_model": str(self.embedder.model),
            "index_file": str(self.index_path),
            "index_size_bytes": p.stat().st_size if p and p.exists() else None,
            "index_sha256_12": "",
            "meta_file": str(self.meta_path),
            "meta_count": len(self.metastore) if self.metastore else 0,
        }


