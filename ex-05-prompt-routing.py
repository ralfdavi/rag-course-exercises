import boto3
import json
import os

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))

def clean_response(response_text):
    """Remove reasoning tags from model responses"""
    original = response_text
    if "<reasoning>" in response_text and "</reasoning>" in response_text:
        # Extract content after reasoning tags
        parts = response_text.split("</reasoning>")
        if len(parts) > 1 and parts[1].strip():
            response_text = parts[1].strip()
        else:
            # If nothing after reasoning, extract content before reasoning tag
            response_text = response_text.split("<reasoning>")[0].strip()
    
    # If cleaning resulted in empty string, return original
    if not response_text.strip():
        return original.strip()
    
    return response_text

# Step 1: Basic Intent Classifier
def classify_intent(query):
    """
    Classifies the intent of a query into 'factual', 'explanation', or 'guidance'.
    """
    system_prompt = """Classify the user query into exactly one category: 'factual', 'explanation', or 'guidance'.
    
    - factual: Questions asking for specific facts, definitions, or yes/no answers
    - explanation: Questions asking how or why something works
    - guidance: Questions asking for advice, recommendations, or what to do
    
    Do not use reasoning tags. Respond with ONLY one word: factual, explanation, or guidance."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    raw_intent = response_body['choices'][0]['message']['content']
    print(f"DEBUG - Raw intent response: '{raw_intent}'")
    intent = raw_intent.strip().lower()
    # Clean any reasoning tags from intent
    intent = clean_response(intent)
    print(f"DEBUG - Cleaned intent: '{intent}'")
    return intent

# Step 2: Prompt Selection Based on Intent
def generate_prompt(query, intent):
    if intent == 'factual':
        return f"Provide a factual answer to the question: {query}"
    elif intent == 'explanation':
        return f"Explain in detail: {query}"
    elif intent == 'guidance':
        return f"Provide guidance for the following situation: {query}"
    else:
        return f"Answer the question: {query}"
    
# Step 3: RAG response using Prompt Routing
def prompt_routing_rag(query):
    # Classify the intent
    intent = classify_intent(query)
    print(f"Detected Intent: {intent}")

    # Generate appropriate prompt
    prompt = generate_prompt(query, intent)
    print(f"Generated Prompt: {prompt}")

    # Generate response based on prompt
    response = bedrock_runtime.invoke_model(
        modelId='openai.gpt-oss-20b-1:0',
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250
        }),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    answer = response_body['choices'][0]['message']['content']
    return clean_response(answer)

"""
Flow Routing

Helps direct queries to specific indexes or prompts based on the nature of the query. The primary types of routing are:

- Prompt Routing**: Directs a query to different prompts based on the intent or topic.
- Database Routing**: Directs a query to different subsets or indexes in a database based on the category or type of information required.

Using flow routing optimizes retrieval pathways, especially useful for large, diverse databases or complex user queries.
"""
def main():
    print(f"### Prompt Routing RAG ###\n")

    try:
        # Test RAG with fusion
        #query = "what is quantum entanglement?"
        query = "how to change a bike tire?"
        response = prompt_routing_rag(query)
        print(f"User: {query}")
        print("------------------------------------------------------------------------")
        print(f"Bot: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()