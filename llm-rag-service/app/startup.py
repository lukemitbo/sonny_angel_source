import uvicorn
from .llm import LLMService
from .rag import ensure_local_index_dir, get_index_dir, LocalFaissVectorStoreManager, SimpleRetriever

def init():
    # Build dependencies to allow warm initialization when running directly
    local_dir = ensure_local_index_dir()
    index_dir = get_index_dir() or local_dir
    vector_store = LocalFaissVectorStoreManager(str(index_dir))
    retriever = SimpleRetriever(vector_store)
    service = LLMService(retriever=retriever)
    service._initialize()
    
if __name__ == "__main__":
    print("Initializing LLM service...")
    init()
    print("Starting API server...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080, reload=True)
