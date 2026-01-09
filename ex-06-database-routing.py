import boto3
import json
import os
from pinecone import Pinecone

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

tech_name = "tech-index"
finance_name = "finance-index"
product_name = "product-index"

product_db = pc.Index(product_name)
tech_db = pc.Index(tech_name)
finance_db = pc.Index(finance_name)

system_prompt = {
                    "role": "system",
                    "content": f"""
                    You are a helpfull E-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
                    Our store sells e-books and courses for IT professionals.

                    Answer only based on the context!
                    Context: {{}}
                    """,
                }

def clean_response(response_text):
    """Remove reasoning tags from model responses"""
    # Remove reasoning tags (both complete and incomplete)
    if "<reasoning>" in response_text:
        if "</reasoning>" in response_text:
            # Complete reasoning tags - get content after
            parts = response_text.split("</reasoning>")
            if len(parts) > 1 and parts[1].strip():
                response_text = parts[1].strip()
            else:
                # If nothing after, get content before
                response_text = response_text.split("<reasoning>")[0].strip()
        else:
            # Incomplete reasoning tag - remove it and everything after
            response_text = response_text.split("<reasoning>")[0].strip()
    
    return response_text.strip() if response_text.strip() else "other"

# Step 1: Basic Intent Classifier
def classify_intent_db_route(query):
    system_msg = """You are a classification assistant. Classify the user's question into exactly ONE of these categories:
- product (questions about what products we sell, product features, courses, e-books)
- finance (questions about payment, refunds, pricing, discounts, billing)
- tech (questions about technical issues, login, platform access, downloads)
- other (anything else)

Respond with ONLY the category name, nothing else."""
    
    messages = [
        {"role": "user", "content": query}
    ]

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 250,
            "system": system_msg,
            "messages": messages,
            "temperature": 0.3
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    print
    return response_body['content'][0]['text']

# Step 2: Prompt Selection Based on Intent
def get_index(intent):
    if intent == 'product':
        return product_db
    elif intent == 'finance':
        return finance_db
    elif intent == 'tech':
        return tech_db
    return None
    
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

# Step 3: Enhanced database routing RAG function
def advanced_database_routing_rag(query):
    # Route to the correct index
    intent = classify_intent_db_route(query)
    index = get_index(intent)
    print(f"DEBUG - Routing to index for intent: {intent}")
    if index:
        # Generate embedding
        query_embedding = get_embedding_model(query)

        # Retrieve documents from the correct index
        response = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace="ns1")

        return "\n\n".join([match['metadata']['answer'] for match in response['matches']])
    else:
        return None

def prompt_builder(system_message, context):
  return system_message['content'].format(context)

# Step 3: Answer question
def routing_rag(query):

    context = advanced_database_routing_rag(query)

    if context is None:
      return "Can't help you with that."

    augmented_prompt = system_prompt['content'].format(context)

    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

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
Database Routing

"""
def main():
    print(f"### Database Routing RAG ###\n")

    try:
        # Test RAG with routing database
        #query = "what products do we sell?" # Expected to route to product index
        query = "can I take my course using Chrome?" # Expected to route to tech index
        response = routing_rag(query)
        print(f"User: {query}")
        print("------------------------------------------------------------------------")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()