import boto3
import json
import os
import pinecone
from pinecone import Pinecone

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))

index_name = "ecommerce-index"
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_db = pc.Index(index_name)


def get_system_prompt(context):
    return f"""
    You are a helpful E-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
    Our store sells e-books and courses for IT professionals.
    
    Only answer based on the context!

    Context: {context}"""

def retrieve_faq(query_embedding, top_k=1):
    response = index_db.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="ns1"
    )
    return response.matches[0].metadata['answer']

def get_embedding_model(prompt, model="amazon.titan-embed-text-v1"):
    body = json.dumps({
        "inputText": prompt
        })

    response = bedrock_runtime.invoke_model(
        modelId=model,
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def combine_documents(retrieved_docs):
    return "\n\n".join(retrieved_docs)

def candidates_generation(query, n_candidates=5):

    system_prompt = f"""You are an AI based algorithm that has a task to generate ({n_candidates}) different versions of the user-generated question.

    These questions will serve as candidates to retrieve relevant documents from vector database.
    Questions should be short to the point.
    The output should be in the JSON format:
    {{
        "candidate_1": "first candidate question",
        "candidate_2": "second candidate question",
        ...
    }}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1500
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    #print(f"DEBUG - Full response body: {response_body}")
    return response_body['choices'][0]['message']['content']

def retrieve_faq_top_n(query_embedding, top_k=5):
    response = index_db.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="ns1"
    )

    results = []
    for res in response['matches']:
        results.append(res['metadata']['answer'])
    return results

def reciprocal_rank_fusion(results, k=60, top_n=5):
    ranked_docs = {}

    for docs in results:
        for i, doc in enumerate(docs):
            if doc not in ranked_docs:
                ranked_docs[doc] = 0

            ranked_docs[doc] += 1 / (i + k)

    # Sort documents by their RRF score in descending order
    top_n_docs = [doc for doc, score in sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    return top_n_docs

def clean_response(response_text):
    # Clean the response - remove reasoning tags and markdown code blocks
    if "<reasoning>" in response_text:
        response_text = response_text.split("</reasoning>")[1].strip()
    
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    return response_text

def fusion_rag_chatbot(query):

    # Step 1: Get multi-representation
    candidates_json = candidates_generation(query, n_candidates=5)
    candidates_json = clean_response(candidates_json)
    candidates = json.loads(candidates_json)

    # Step 2: Retrieve the most relevant FAQ from Pinecone (for each candidate)
    relevant_docs = []
    for key, candidate in candidates.items():
      candidate_embedding = get_embedding_model(candidate)
      best_matches = retrieve_faq_top_n(candidate_embedding, top_k=5)
      relevant_docs.append(best_matches)

    # Step 3: Ranking - Run the reciprotial fusion ranking algorithm to re-rank all results
    ranked_docs = reciprocal_rank_fusion(relevant_docs, k=60, top_n=4)

    # Step 4: Combine docs
    context = combine_documents(ranked_docs)

    # Step 5: Augment the query with context
    augmented_prompt = get_system_prompt(context)

    messages = [{"role": "system","content": augmented_prompt},
                {"role": "user","content": query}]

    # Step 6: Use OpenAI to generate a response
    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": messages,
            "temperature": 0,
            "max_tokens": 250
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    answer = response_body['choices'][0]['message']['content']
    
    # Remove reasoning tags if present
    if "<reasoning>" in answer:
        answer = answer.split("</reasoning>")[1].strip()
    
    return answer

""" 
Fusion RAG (Reciprocal Rank Fusion)

Combines results from multiple retrieval strategies (e.g., semantic search + keyword search + metadata filtering) and ranks them using reciprocal rank fusion scoring to find the best matches.

Use when:

- You have multiple search methods available (dense + sparse vectors, metadata filters)
- You need to balance different retrieval approaches
- Results from single methods are inconsistent
- You want to leverage both semantic similarity and exact keyword matching

Example: Searches both by embeddings similarity AND exact keyword match, then fuses rankings to get the most relevant documents.
"""
def main():
    print(f"### Fusion RAG ###\n")

    try:
        # Test RAG with fusion
        query = "what if i don't like the product, how many days do i have to return it?"
        response = fusion_rag_chatbot(query)
        print(f"User: {query}")
        print("------------------------------------------------------------------------")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()