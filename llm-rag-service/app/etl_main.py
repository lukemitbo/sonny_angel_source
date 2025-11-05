import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import requests

from app.rag.embedding import STEmbedder
from app.rag.vector_store import LocalFaissVectorStoreManager
from app.rag.utils import read_texts_from_folder


# -----------------------------
# Chunking / Cleaning utilities
# -----------------------------

def clean_text(text: str) -> str:
    """Minimal cleaning: collapse whitespace and trim."""
    # Avoid heavy changes to preserve original content for traceability
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 120) -> List[str]:
    """Simple heuristic chunker based on characters as proxy for tokens.

    This avoids a tokenizer dependency and is usually sufficient for MiniLM.
    """
    if not text:
        return []
    # Treat ~4 chars ~= 1 token heuristic
    max_chars = max(200, max_tokens * 4)
    overlap_chars = max(0, min(overlap * 4, max_chars // 2))

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_chars
        if start < 0:
            start = 0
    return chunks


# -----------------------------
# Manifest
# -----------------------------

@dataclass
class Manifest:
    run_id: str
    created_at_epoch: float
    docs_count: int
    chunks_count: int
    embedding_model: str
    index_s3_uri: str
    meta_s3_uri: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# -----------------------------
# Core ETL
# -----------------------------

def load_env() -> Dict[str, str]:
    env = {
        "ARTIFACTS_BUCKET": os.getenv("ARTIFACTS_BUCKET", "").strip(),
        "RAG_BASE_PREFIX": os.getenv("RAG_BASE_PREFIX", "rag/").strip() or "rag/",
        "RUN_ID": os.getenv("RUN_ID", "").strip(),
        "EMBEDDER_MODEL": os.getenv(
            "EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        "DOCS_DIR": os.getenv("DOCS_DIR", "/app/docs").strip(),
        # Optional hot reload
        "RELOAD_URL": os.getenv("RELOAD_URL", "").strip(),
        "RELOAD_TOKEN": os.getenv("RELOAD_TOKEN", "").strip(),
        # Dry run mode
        "DRY_RUN": int(os.getenv("DRY_RUN", "0").strip()),
    }
    if env["DRY_RUN"] != 1 and not env["ARTIFACTS_BUCKET"]:
        raise RuntimeError("ARTIFACTS_BUCKET env var is required unless DRY_RUN=1")
    if not env["RUN_ID"]:
        env["RUN_ID"] = uuid.uuid4().hex[:12]
    return env


def fetch_documents(docs_dir: str) -> Tuple[List[str], List[Dict]]:
    folder = Path(docs_dir)
    if not folder.exists():
        raise RuntimeError(f"Docs directory not found: {folder}")
    texts, metas = read_texts_from_folder(folder)
    return texts, metas


def preprocess_and_chunk(texts: List[str], metas: List[Dict]) -> Tuple[List[str], List[Dict]]:
    cleaned_chunks: List[str] = []
    chunk_metas: List[Dict] = []
    for text, meta in zip(texts, metas):
        cleaned = clean_text(text)
        parts = chunk_text(cleaned)
        for i, part in enumerate(parts):
            m = dict(meta)
            m["chunk_index"] = i
            m["num_chunks_in_doc"] = len(parts)
            cleaned_chunks.append(part)
            chunk_metas.append(m)
    return cleaned_chunks, chunk_metas


def build_index(local_out_dir: Path, chunks: List[str], metas: List[Dict], embedder_model: str) -> LocalFaissVectorStoreManager:
    embedder = STEmbedder(model_name=embedder_model)
    store = LocalFaissVectorStoreManager(str(local_out_dir), embedder=embedder)
    if chunks:
        store.add_texts(chunks, metas)
    return store


def upload_to_s3(bucket: str, base_prefix: str, run_id: str, local_out_dir: Path) -> Tuple[str, str]:
    s3 = boto3.client("s3")
    key_prefix = f"{base_prefix.rstrip('/')}/{run_id}/"
    index_path = local_out_dir / "index.faiss"
    meta_path = local_out_dir / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError("Expected index.faiss and meta.jsonl to exist after build")
    index_key = f"{key_prefix}index.faiss"
    meta_key = f"{key_prefix}meta.jsonl"
    s3.upload_file(str(index_path), bucket, index_key)
    s3.upload_file(str(meta_path), bucket, meta_key)
    return f"s3://{bucket}/{index_key}", f"s3://{bucket}/{meta_key}"


def write_manifest_and_upload(
    bucket: str,
    base_prefix: str,
    run_id: str,
    manifest: Manifest,
) -> str:
    s3 = boto3.client("s3")
    key_prefix = f"{base_prefix.rstrip('/')}/{run_id}/"
    manifest_key = f"{key_prefix}manifest.json"
    tmp_path = Path("./manifest_tmp.json")
    tmp_path.write_text(manifest.to_json(), encoding="utf-8")
    try:
        s3.upload_file(str(tmp_path), bucket, manifest_key)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return f"s3://{bucket}/{manifest_key}"


def write_manifest_local(local_out_dir: Path, manifest: Manifest) -> str:
    local_path = local_out_dir / "manifest.json"
    local_path.write_text(manifest.to_json(), encoding="utf-8")
    return str(local_path)


def maybe_reload(url: str, token: str) -> None:
    if not url:
        return
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.post(url, headers=headers, timeout=10)
        if resp.status_code >= 400:
            print(f"[warn] reload request failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[warn] reload request error: {e}")


def main() -> int:
    try:
        env = load_env()
        bucket = env.get("ARTIFACTS_BUCKET", "")
        base_prefix = env["RAG_BASE_PREFIX"]
        run_id = env["RUN_ID"]
        model_name = env["EMBEDDER_MODEL"]
        docs_dir = env["DOCS_DIR"]
        dry_run = env["DRY_RUN"] == 1

        # Local output dir for this run (docker volume mount compatible when DRY_RUN)
        local_base = Path("/artifacts/rag") if dry_run else Path(
            "./artifacts/rag")
        local_out_dir = local_base / run_id
        local_out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Fetch docs
        print(f"[RAG] Fetching documents from {docs_dir}")
        texts, metas = fetch_documents(docs_dir)
        docs_count = len(texts)
        if docs_count == 0:
            raise RuntimeError("No documents found to index")

        # 2) Clean + chunk
        chunks, chunk_metas = preprocess_and_chunk(texts, metas)

        # 3) Embed + FAISS
        print(f"[RAG] Building index with model {model_name}")
        store = build_index(local_out_dir, chunks, chunk_metas, model_name)

        # 4) Upload artifacts (or keep local in DRY_RUN)
        if dry_run:
            index_uri = str(local_out_dir / "index.faiss")
            meta_uri = str(local_out_dir / "meta.jsonl")
        else:
            index_uri, meta_uri = upload_to_s3(bucket, base_prefix, run_id,
                                               local_out_dir)

        # 5) Manifest
        manifest = Manifest(
            run_id=run_id,
            created_at_epoch=time.time(),
            docs_count=docs_count,
            chunks_count=len(chunks),
            embedding_model=model_name,
            index_s3_uri=index_uri,
            meta_s3_uri=meta_uri,
        )
        if dry_run:
            manifest_uri = write_manifest_local(local_out_dir, manifest)
        else:
            manifest_uri = write_manifest_and_upload(bucket, base_prefix,
                                                     run_id, manifest)

        # 6) Optional reload
        maybe_reload(env.get("RELOAD_URL", ""), env.get("RELOAD_TOKEN", ""))

        print(
            json.dumps({
                "status": "ok",
                "run_id": run_id,
                "index": index_uri,
                "meta": meta_uri,
                "manifest": manifest_uri,
                "ntotal": store.store.ntotal,
            }))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
