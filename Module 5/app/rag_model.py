import openai
import pinecone
from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

# Initialize OpenAI and Pinecone clients
client = openai.OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Define your Pinecone index
index_name = "rag-index"
index = pinecone.Index(index_name)

def get_rag_response(query):
    """
    Generates a response using RAG by querying Pinecone and
    using OpenAI for context-augmented generation.
    """
    # Step 1: Create embeddings for the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    # Step 2: Query Pinecone for relevant documents
    results = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    documents = [match["metadata"]["text"] for match in results["matches"]]
    
    # Step 3: Create a prompt with context for OpenAI
    prompt = f"Query: {query}\n\nContext:\n" + "\n".join(documents) + "\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    ).choices[0].message.content
    
    return response
