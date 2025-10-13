import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional

from .llm import llm_service
from .rag import rag


app = fastapi.FastAPI(title="Mistral LLM Service")

@app.on_event("startup")
def _startup():
    # container-local paths after you download from S3
    st = rag.status()
    print(f"[RAG] Loaded index: ntotal={st['ntotal']} dim={st['dim']} model={st['embedding_model']}")


@app.get("/healthz")
def healthz():
    st = rag.status()
    ok = st["loaded"] and st["ntotal"] > 0
    return {"ok": ok, **st}


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
        context, response = llm_service.query_with_rag(
            query=request.query,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        # Include context in response if available
        final_response = response if not context else f"<context>{context}</context>\n\n{response}"
        return QueryResponse(response=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
