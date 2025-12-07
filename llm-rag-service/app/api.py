import traceback
from threading import Lock
from typing import Optional, Tuple

import fastapi
from fastapi import HTTPException, Request
from pydantic import BaseModel

from .llm import LLMService
from .rag import (
    ensure_local_index_dir,
    get_index_dir,
    LocalFaissVectorStoreManager,
    SimpleRetriever,
)

app = fastapi.FastAPI(title="EXAONE RAG Service")

# ---------------------------------------------------------------------------
# Global init state (lazy initialization)
# ---------------------------------------------------------------------------

_init_lock = Lock()
_initialized = False


def _init_llm_if_needed() -> None:
    """
    Lazily initialize vector store + retriever + LLMService.

    Runs only once per process, guarded by a lock.
    """
    global _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        print("[INIT] Starting RAG + LLM initialization")

        # Ensure local index exists (download from S3 if needed)
        local_dir = ensure_local_index_dir()
        index_dir = get_index_dir() or local_dir
        print(f"[INIT] Loading FAISS index from: {index_dir}")

        vector_store = LocalFaissVectorStoreManager(str(index_dir))
        retriever = SimpleRetriever(vector_store)

        app.state.vector_store = vector_store
        app.state.retriever = retriever

        app.state.llm_service = LLMService(retriever=retriever)
        app.state.llm_service.initialize()

        st = vector_store.status()
        print(
            f"[INIT] Index loaded: ntotal={st.get('ntotal')} "
            f"dim={st.get('dim')} model={st.get('embedding_model')}"
        )

        _initialized = True
        print("[INIT] RAG + LLM initialization complete")


# ---------------------------------------------------------------------------
# FastAPI lifecycle & middleware
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _startup() -> None:
    """
    Keep startup light so the container can become 'healthy' quickly.

    Heavy model initialization is done lazily on first /query call.
    """
    print("[STARTUP] FastAPI app started (LLM init is lazy).")


@app.middleware("http")
async def log_query_body(request: Request, call_next):
    """
    Simple logging middleware for /query requests.
    """
    if request.url.path == "/query":
        try:
            body_bytes = await request.body()
            print(f"[REQUEST] /query body: {body_bytes.decode('utf-8', 'ignore')}")
        except Exception as e:
            print(f"[REQUEST] Failed to read /query body: {e}")

    return await call_next(request)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    """
    Lightweight health endpoint for ALB.

    - Always returns 200 as long as the process is running.
    - Includes 'ready' flag if the FAISS index / LLM are initialized.
    """
    ready = hasattr(app.state, "vector_store") and hasattr(app.state, "llm_service")

    status = {
        "status": "ok",
        "ready": ready,
    }

    # If initialized, include some extra info, but don't break healthz if it fails.
    if ready:
        try:
            st = app.state.vector_store.status()
            status.update(
                {
                    "loaded": st.get("loaded"),
                    "ntotal": st.get("ntotal"),
                    "dim": st.get("dim"),
                    "embedding_model": st.get("embedding_model"),
                }
            )
        except Exception as e:
            status["ready"] = False
            status["error"] = f"status() failed: {e}"

    return status


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class QueryResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------------


def _run_rag_query(
    req: QueryRequest,
) -> Tuple[Optional[str], str]:
    """
    Wraps LLMService call so we can more easily handle / log errors.
    """
    llm_service: LLMService = app.state.llm_service  # type: ignore[attr-defined]
    context, response = llm_service.query_with_rag(
        query=req.query,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return context, response


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    RAG-powered query endpoint.

    - Lazily initializes FAISS index + LLM on first request.
    - Wraps exceptions into a 500 HTTP response with a simple message.
    """
    try:
        _init_llm_if_needed()

        context, response = _run_rag_query(request)

        print(f"[RAG] Context: {context}")
        print(f"[RAG] Response: {response}")

        if context:
            final_response = f"<context>{context}</context>\n\n{response}"
        else:
            final_response = response

        return QueryResponse(response=final_response)

    except HTTPException:
        # Let explicit HTTPExceptions pass through unchanged
        raise
    except Exception as e:
        print("[ERROR] Exception in /query:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Uvicorn entrypoint (for docker CMD)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("Starting API server (uvicorn, 0.0.0.0:8080)...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080)  # adjust to module path
