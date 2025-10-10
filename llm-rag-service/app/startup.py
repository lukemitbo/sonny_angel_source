import uvicorn
from .llm import llm_service

def init():
    # Force initialization of the LLM
    llm_service._initialize()
    
if __name__ == "__main__":
    print("Initializing LLM service...")
    init()
    print("Starting API server...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080)
