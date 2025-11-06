import uvicorn
    
if __name__ == "__main__":
    print("Starting API server...")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080)
