import fastapi
from fastapi import HTTPException
from pathlib import Path
from pydantic import BaseModel
import uvicorn
from typing import Optional

from .llm import llm_service
from .rag import RAG


app = fastapi.FastAPI(title="Mistral LLM Service")


def get_latest_index_dir() -> Optional[Path]:
    """Find the most recent FAISS index directory."""
    base_dir = Path(__file__).parent.parent / "artifacts" / "rag"
    if not base_dir.exists():
        return None

    # List all timestamped directories and sort by name (timestamps) in descending order
    index_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    return index_dirs[0] if index_dirs else None


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
        # Get latest index directory
        index_dir = get_latest_index_dir()
        if not index_dir:
            # No index available, fall back to raw LLM
            response = llm_service.generate(
                prompt=request.query,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p)
            return QueryResponse(response=response)

        # Initialize RAG with latest index (don't create new timestamp folder)
        rag = RAG(str(index_dir), create_timestamp=False)

        # Retrieve relevant context
        context_results = rag.retrieve(request.query, k=3)
        context = "\n\n".join(text for text, _ in context_results)

        # Build prompt with context
        prompt = f"""Use only the following context to answer the question. If the context doesn't contain relevant information, say so and refuse to answer.

Context:
{context}

Question: {request.query}

Answer:"""
        print(prompt)
        # Generate response with context
        response = llm_service.generate(prompt=prompt,
                                        max_new_tokens=request.max_new_tokens,
                                        temperature=request.temperature,
                                        top_p=request.top_p)
        return QueryResponse(response=f"<context>{context}</context>\n\n{response}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
