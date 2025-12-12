import hashlib
import os
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()[:12]


def ensure_local_index_dir() -> Path:
    """Ensure FAISS index is available locally depending on environment.

    Behavior:
    - In Docker/ECS: download from S3 into /artifacts/rag unconditionally.
    - On local dev machine: if project artifacts exist, use them; otherwise,
      optionally download from S3 when RAG_BOOTSTRAP_S3 is truthy.

    Returns:
        Path to local directory containing index files.
    """
    # Detect container/ECS environment
    is_docker = Path("/.dockerenv").exists()
    is_ecs = bool(os.getenv("ECS_CONTAINER_METADATA_URI") or os.getenv("ECS_CONTAINER_METADATA_URI_V4") or os.getenv("AWS_EXECUTION_ENV", "").startswith("AWS_ECS"))

    bucket = os.getenv("S3_ARTIFACTS_BUCKET")
    prefix = os.getenv("RAG_PREFIX", "rag/")
    force_s3 = os.getenv("RAG_BOOTSTRAP_S3", "").lower() in {"1", "true", "yes"}

    files = ["index.faiss", "meta.jsonl"]

    if is_docker or is_ecs:
        # Running in container (e.g., AWS ECS). Download to container path.
        local_dir = Path("/artifacts/rag")
        local_dir.mkdir(parents=True, exist_ok=True)
        if not bucket:
            raise RuntimeError("S3_ARTIFACTS_BUCKET must be set in container environment")
        if all((local_dir / f).exists() for f in files):
            print("[bootstrap] artifacts already present; skipping download")
            return local_dir
        s3 = boto3.client(
            "s3",
            config=Config(
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        for fname in files:
            local_path = local_dir / fname
            s3_key = f"{prefix}{fname}"

            t0 = time.time()
            try:
                print("[bootstrap] verifying S3 access with head_object...")
                s3.head_object(Bucket=bucket, Key=s3_key)
                print("[bootstrap] head_object ok")
                print(f"[bootstrap] downloading s3://{bucket}/{s3_key} → {local_path}")
                s3.download_file(bucket, s3_key, str(local_path))
            except (ClientError, BotoCoreError) as e:
                print(f"[bootstrap][ERROR] download failed for {s3_key}: {e}")
                raise
            finally:
                print(f"[bootstrap] download step finished in {time.time() - t0:.2f}s for {s3_key}")
        return local_dir

    # Local development: prefer repo artifacts if present
    project_root = Path(__file__).resolve().parent.parent.parent
    local_dir = project_root / "artifacts" / "rag"
    local_dir.mkdir(parents=True, exist_ok=True)

    have_local_files = all((local_dir / f).exists() for f in files)
    if have_local_files and not force_s3:
        # Use existing local index
        return local_dir

    # Optionally pull from S3 when configured
    if bucket and (force_s3 or not have_local_files):
        s3 = boto3.client(
            "s3",
            config=Config(
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

        for fname in files:
            local_path = local_dir / fname
            s3_key = f"{prefix}{fname}"
            print("[bootstrap] verifying S3 access with head_object...")
            s3.head_object(Bucket=bucket, Key=s3_key)
            print("[bootstrap] head_object ok")
            print(f"[bootstrap] downloading s3://{bucket}/{s3_key} → {local_path}")
            s3.download_file(bucket, s3_key, str(local_path))
        return local_dir

    # No S3 download; return (possibly empty) local dir
    return local_dir


def read_texts_from_folder(folder: Path, exts=(".txt", ".md")) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts, metas = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            texts.append(txt)
            metas.append({"source_path": str(p)})
    return texts, metas


def get_index_dir() -> Optional[Path]:
    """Find the FAISS index directory."""
    docker_path = Path("/artifacts/rag")
    if docker_path.exists():
        return docker_path

    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir = project_root / "artifacts" / "rag"
    if base_dir.exists():
        return base_dir
    return None


