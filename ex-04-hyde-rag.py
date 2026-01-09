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

def generate_hypothetical_doc(query):

    system_prompt = f"Create a hypothetical document based on the following query: {query}"
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": messages,
            #"temperature": 0,
            "max_tokens": 300
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    return response_body['choices'][0]['message']['content']

def hypo_chatbot(query):

    # Step 1: Get multi-representation
    hypo_candidate = generate_hypothetical_doc(query)

    # Step 2: Retrieve the most relevant FAQ from Pinecone (for each candidate)
    candidate_embedding = get_embedding_model(hypo_candidate)

    # Step 3: Get real answer
    best_match = retrieve_faq(candidate_embedding)

    # Step 4: Augment the query with context
    augmented_prompt = get_system_prompt(best_match)

    messages = [{"role": "system","content": augmented_prompt},
                {"role": "user","content": query}]

    # Step 6: Use OpenAI to generate a response
    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": messages,
            #"temperature": 0,
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
Hypothetical Document Embeddings (HyDE) RAG

Generates a hypothetical answer to the user's question first, then embeds and searches for that hypothetical answer in the vector database. This approach finds documents similar to what the answer should look like, rather than searching by the question itself.

Use when:
- The question and answer have different semantic structures
- Your knowledge base contains answers/solutions rather than questions
- You want to bridge the semantic gap between questions and answers
- Users ask questions in ways that don't match how answers are stored

Example: Query: "what if i don't like the product?" → HyDE generates: "Our return policy allows customers to return products within 30 days if unsatisfied..." → searches for documents similar to this hypothetical answer.
"""
def main():
    print(f"### Hypothetical Document Embeddings (HyDE)  ###\n")

    try:
        # Test RAG with fusion
        query = "do you wrap some presents?"
        response = hypo_chatbot(query)
        print(f"User: {query}")
        print("------------------------------------------------------------------------")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()