from fastapi import FastAPI, HTTPException, Request
from app.rag_model import get_rag_response
from app.cache import get_cached_response, set_cached_response

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/query/")
async def query_rag(request: Request):
    """
    Endpoint to handle RAG queries.
    Checks the cache first before generating a response.
    """
    data = await request.json()
    query = data.get("query")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Check cache for the query response
    cached_response = get_cached_response(query)
    if cached_response:
        return {"response": cached_response, "source": "cache"}

    # Generate RAG response if not cached
    response = get_rag_response(query)
    set_cached_response(query, response)  # Cache the new response
    
    return {"response": response, "source": "RAG"}
