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
        "candidate_2": "second candidate question"
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

def clean_response(response_text):
    # Clean the response - remove reasoning tags and markdown code blocks
    if "<reasoning>" in response_text:
        response_text = response_text.split("</reasoning>")[1].strip()
    
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    return response_text

def multi_query_rag_chatbot(query):

    # Step 1: Get multi-representation
    candidates_json = candidates_generation(query, n_candidates=5)
    candidates_json = clean_response(candidates_json)
    candidates = json.loads(candidates_json)

    # Step 2: Retrieve the most relevant FAQ from Pinecone (for each candidate)
    relevant_docs = []
    for key, candidate in candidates.items():
      candidate_embedding = get_embedding_model(candidate)
      best_match = retrieve_faq(candidate_embedding)
      relevant_docs.append(best_match)

    # Step 3: Combine docs
    context = combine_documents(relevant_docs)

    # Step 4: Augment the query with context
    augmented_prompt = get_system_prompt(context)

    messages = [{"role": "user", "content": [
        { "type": "text", "text": f"{query}" }
        ]}]

    # Step 4: Use Sonnet to generate a response
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 250,
            "system": augmented_prompt,
            "messages": messages,
            "temperature": 0.3
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']


""" 
Multi-Query RAG

Generates multiple variations of the user's question to retrieve different perspectives from the vector database. 
Each variation is embedded and searched independently, then results are combined.

Use when:
- User questions are vague or ambiguous
- You want to capture different interpretations of the same query
- The knowledge base has semantically similar content in different contexts
- You need broader context coverage

Example: "student discount?" → generates "Do you offer discounts for students?", "What are the student pricing options?", "Are there educational discounts available?
"""
def main():

    print(f"### Multi-Query RAG ###\n")

    try:
        # Test RAG with multi-query
        query = "do you have any student discount?"
        response = multi_query_rag_chatbot(query)
        print(f"User: {query}")
        print("------------------------------------------------------------------------")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()