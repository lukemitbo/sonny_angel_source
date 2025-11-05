import faiss
import numpy as np


class STEmbedder:
    """Simple wrapper over sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import so users without ST for other backends don't break
        from sentence_transformers import SentenceTransformer
        print(f"[STEMBEDDER] Initializing sentence-transformers model {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        # MiniLM-L6-v2 outputs 384-d vectors
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        # We'll normalize ourselves to guarantee cosine via IP
        faiss.normalize_L2(embs)
        return embs


