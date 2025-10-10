import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional

from .llm import llm_service


app = fastapi.FastAPI(title="Mistral LLM Service")


@app.get("/healthz")
def healthz():
    return {"message": "OK"}


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
        response = llm_service.generate(
            prompt=request.query,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)