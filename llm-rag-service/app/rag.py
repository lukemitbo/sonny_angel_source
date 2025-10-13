import json
import hashlib
from dataclasses import dataclass
import time
import pathlib
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import faiss
import numpy as np
from tqdm import tqdm

# -------- Embedding backend (Sentence-Transformers) --------
class STEmbedder:
    """Simple wrapper over sentence-transformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import so users without ST for other backends don't break
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # MiniLM-L6-v2 outputs 384-d vectors
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        # We'll normalize ourselves to guarantee cosine via IP
        faiss.normalize_L2(embs)
        return embs


# -------- Metadata sidecar (jsonl) --------
@dataclass
class Record:
    text: str
    meta: Dict[str, Any]

class MetaStore:
    """
    Stores/loads records in a jsonl file with the SAME order as vectors in FAISS.
    The nth line corresponds to the nth vector.
    """
    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self._records: List[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    def load(self) -> None:
        self._records.clear()
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    self._records.append(Record(text=obj["text"], meta=obj.get("meta", {})))

    def append_many(self, records: Iterable[Record]) -> None:
        self._records.extend(records)
        # append to disk to avoid rewriting the whole file
        with self.path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"text": r.text, "meta": r.meta}, ensure_ascii=False) + "\n")

    def get(self, idx: int) -> Record:
        return self._records[idx]


# -------- FAISS store --------
class FaissStore:
    """
    Cosine similarity via IndexFlatIP on normalized vectors.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(
            dim)  # inner product == cosine when normalized

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32 and vectors.shape[1] == self.dim
        # vectors must already be L2-normalized
        self.index.add(vectors)

    def search(self, vectors: np.ndarray,
               k: int) -> Tuple[np.ndarray, np.ndarray]:
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

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()[:12]


# -------- High-level RAG wrapper --------
class RAG:
    """
    Folder layout:
      <base_dir>/
        YYYYMMDD_HHMMSS/    # Timestamped directory for each index
          index.faiss       # FAISS vectors
          meta.jsonl        # parallel metadata/text, ith line -> ith vector
    """

    def __init__(self,
                 index_dir: str,
                 embedder: Optional[STEmbedder] = None,
                 create_timestamp: bool = False):
        """Initialize RAG with a directory for indices.
        
        Args:
            index_dir: Base directory for indices
            embedder: Optional custom embedder
            create_timestamp: If True, creates a timestamped subdirectory (for building new indices).
                            If False, uses index_dir directly (for loading existing indices).
        
        Directory structure when create_timestamp=True:
            <index_dir>/YYYYMMDD_HHMMSS/index.faiss
            <index_dir>/YYYYMMDD_HHMMSS/meta.jsonl
            
        Directory structure when create_timestamp=False:
            <index_dir>/index.faiss
            <index_dir>/meta.jsonl
        """
        if create_timestamp:
            self.base_dir = Path(index_dir)
            self.index_dir = self.base_dir
        else:
            # Use the provided directory directly
            self.index_dir = Path(index_dir)

        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.jsonl"

        self.embedder = embedder or STEmbedder()

        # Create directory structure
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.metastore = MetaStore(self.meta_path)
        self.metastore.load()
        self.store = FaissStore.load(self.index_path,
                                     dim_hint=self.embedder.dim)

        if self.store.ntotal != len(self.metastore):
            raise RuntimeError(
                f"Index/metadata mismatch: faiss has {self.store.ntotal} vectors, "
                f"metadata has {len(self.metastore)} records. "
                f"Make sure you didn't manually edit files.")

    # ---------- Building ----------
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

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(self.index_path)
        # meta.jsonl is appended on the fly

    # ---------- Retrieval ----------
    def retrieve(self,
                 query: str,
                 k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        q = self.embedder.embed([query])  # normalized
        distances, indices = self.store.search(q, k)
        idxs = indices[0]
        out: List[Tuple[str, Dict[str, Any]]] = []
        for ix in idxs:
            if ix == -1:
                continue
            rec = self.metastore.get(ix)
            out.append((rec.text, rec.meta))
        return out

    def status(self):
        p = pathlib.Path(self.index_path) if self.index_path else None
        m = pathlib.Path(self.meta_path) if self.meta_path else None
        return {
            "loaded": self.store.ntotal > 0,
            "ntotal": int(self.store.ntotal) if self.store.ntotal else 0,
            "dim": self.embedder.dim,
            "embedding_model": str(self.embedder.model),
            "index_file": str(self.index_path),
            "index_size_bytes": p.stat().st_size if p and p.exists() else None,
            "index_sha256_12": sha256(self.index_path) if p and p.exists() else None,
            "meta_file": str(self.meta_path),
            "meta_count": len(self.metastore) if self.metastore else 0,
        }




# -------- Utility functions --------
def read_texts_from_folder(folder: Path, exts=(".txt", ".md")) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Read text files from a folder and return their contents and metadata.
    
    Args:
        folder: Path to the folder containing text files
        exts: Tuple of file extensions to include
        
    Returns:
        Tuple of (list of text contents, list of metadata dicts)
    """
    texts, metas = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            texts.append(txt)
            metas.append({"source_path": str(p)})
    return texts, metas


def get_index_dir() -> Optional[Path]:
    """Find the FAISS index directory."""
    # Try Docker path first, then fall back to local path
    docker_path = Path("/app/artifacts/rag")
    if docker_path.exists():
        return docker_path
        
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "artifacts" / "rag"
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")
    return base_dir

rag = RAG(get_index_dir())
