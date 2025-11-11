import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

from .llm import LLMService
from .rag import ensure_local_index_dir, get_index_dir, LocalFaissVectorStoreManager, SimpleRetriever


app = fastapi.FastAPI(title="Mistral LLM Service")

@app.middleware("http")
async def log_start(request, call_next):
    if request.url.path == "/query":
        print(f"Query received: {request.body}")
    return await call_next(request)

@app.on_event("startup")
def _startup():
    # Ensure local index is available (no-op if already present)
    local_dir = ensure_local_index_dir()
    index_dir = get_index_dir() or local_dir
    print(f"[RAG] Loading index from {index_dir}")
    vector_store = LocalFaissVectorStoreManager(str(index_dir))
    retriever = SimpleRetriever(vector_store)
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.llm_service = LLMService(retriever=retriever)
    app.state.llm_service.initialize()
    st = vector_store.status()
    print(
        f"[RAG] Loaded index: ntotal={st['ntotal']} dim={st['dim']} model={st['embedding_model']}"
    )


@app.get("/healthz")
def healthz():
    st = app.state.vector_store.status()
    ok = st["loaded"] and st["ntotal"] > 0
    return {"hello???": ":)", "ok": ok, **st}


class QueryRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class QueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        context, response = app.state.llm_service.query_with_rag(
            query=request.query,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        print(f"Context: {context}")
        print(f"Response: {response}")
        # Include context in response if available
        final_response = response if not context else f"<context>{context}</context>\n\n{response}"
        return QueryResponse(response=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
