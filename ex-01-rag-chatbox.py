import boto3
import json
import os
import numpy as np
import pinecone
from pinecone import Pinecone

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))

index_name = "ecommerce-index"
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_db = pc.Index(index_name)

faq_database = {
    "What is your return policy?": "Our return policy allows customers to return products within 30 days of purchase. Items must be in their original condition and packaging. To initiate a return, visit our return portal and provide your order number and email address.",

    "How do I track my order?": "You can track your order by using the tracking number provided in the shipment confirmation email. Alternatively, you can log in to your account and go to the 'Order History' section to find the tracking link.",

    "What payment methods do you accept?": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay. For corporate accounts, we also offer invoicing options. Please contact support for more information on setting up a corporate account.",

    "Can I change or cancel my order after it’s been placed?": "Once an order has been placed, we are unable to modify it directly. However, you can cancel your order within the first hour of placing it through the 'My Orders' section of your account. After that, you’ll need to wait for the order to be delivered and then initiate a return.",

    "What are your shipping options?": "We offer standard, expedited, and overnight shipping. Standard shipping takes 5-7 business days, while expedited shipping takes 2-3 business days. Overnight shipping ensures delivery by the next business day. International shipping options are also available, with delivery times varying by destination.",

    "How do I reset my account password?": "To reset your password, go to the login page and click on 'Forgot Password'. You will receive an email with instructions to reset your password. If you don't see the email, check your spam folder or contact customer support for help.",

    "Do you ship internationally?": "Yes, we ship to select international destinations. International shipping costs and delivery times vary depending on the destination. You can calculate the shipping costs at checkout after providing your address.",

    "What do I do if I receive a damaged or defective product?": "If you receive a damaged or defective product, please contact our customer support within 48 hours of receiving the item. We will provide instructions on how to return the product or arrange for a replacement. Make sure to include photos of the damaged item and packaging for faster processing.",

    "How do I contact customer support?": "You can contact our customer support via email at support@ourcompany.com, or by calling our support line at 1-800-123-4567 during business hours (9 AM to 5 PM, Monday to Friday). We also offer live chat support on our website.",

    "Can I use multiple discount codes on a single order?": "No, our system only allows one discount code per order. However, you can apply store credit or a gift card in addition to a discount code at checkout.",

    "How do I update my shipping address after placing an order?": "If your order has not yet been processed, you can update your shipping address by logging into your account and navigating to the 'My Orders' section. If the order has already been processed or shipped, you will need to contact customer support to discuss possible options.",

    "What should I do if I never received my order?": "If your order has not arrived by the estimated delivery date, first check the tracking information. If the tracking shows the item was delivered but you didn't receive it, contact customer support so we can investigate and resolve the issue.",

    "Do you offer gift wrapping?": "Yes, we offer gift wrapping for an additional fee. You can select the gift wrapping option at checkout, and you can also include a personalized message with the gift.",

    "Can I return a product after 30 days?": "Unfortunately, returns are only accepted within 30 days of the purchase date. If you have extenuating circumstances, please contact customer support to discuss possible exceptions on a case-by-case basis.",

    "What are your business hours?": "Our customer support team is available from 9 AM to 5 PM, Monday through Friday, excluding holidays. Our website is available for orders 24/7.",

    "How do I subscribe to your newsletter?": "To subscribe to our newsletter, scroll to the bottom of our homepage and enter your email in the subscription box. You’ll receive exclusive offers, product updates, and company news directly to your inbox.",

    "What is your warranty policy?": "We offer a one-year warranty on all our products. The warranty covers manufacturing defects but does not cover damage caused by misuse, accidents, or normal wear and tear. To file a warranty claim, contact our customer support team with your order details and a description of the issue.",

    "How can I become a reseller of your products?": "We welcome reseller partnerships! If you're interested in becoming a reseller, please contact our sales team at sales@ourcompany.com with details about your business, and we’ll get back to you with more information.",

    "Do you offer student discounts?": "Yes, we offer a 10% discount for students. To get the discount, sign up with your valid student email, and we will verify your status. After verification, you will receive a unique discount code to use at checkout.",

    "Can I expedite the shipping of my order?": "Yes, you can select expedited or overnight shipping at checkout. Expedited shipping typically takes 2-3 business days, while overnight shipping ensures delivery by the next business day. Please note that expedited shipping costs more than standard shipping."
}

def get_answer(question):
    return faq_database.get(question, "I'm sorry, I don't have an answer for that question.")

def create_faq_vector():
    faq_vector_db = {}
    for question in faq_database.keys():
        faq_vector_db[question] = get_embedding_model(question)
    return faq_vector_db

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar(query_embedding, faq_vector_db):

    # STEP 1: initialize variables to track the best match
    best_batch = None
    highest_similarity = -1  # Initialize with a very low value

    # STEP 2: compare query embedding with each FAQ embedding using cosine similarity
    for faq_question, faq_embedding in faq_vector_db.items():
        similarity = cosine_similarity(query_embedding, faq_embedding)
        if similarity > highest_similarity:
            best_batch = faq_question
            highest_similarity = similarity
    
    return best_batch

def simple_chatbot(query):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": "You are a helpful E-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store. Our store sells e-books and courses for IT professionals.",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    })

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def rag_chatbot(query, vector_database, faq_database):
    # Step 1: Encode the query
    query_embedding = get_embedding_model(query)

    # Step 2: Find the most similar FAQ using cosine similarity
    best_match = find_most_similar(query_embedding, vector_database)
    best_answer = get_answer(best_match)
    
    system_prompt = f"""You are a helpful E-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
        Our store sells e-books and courses for IT professionals.
        Only answer based on the context!
        Context: {best_answer}"""

    # Step 3: Use Bedrock API to generate a response with context
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "system": system_prompt,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.5
        }),
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

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

def rag_chatbot_with_pinecone(query):
    # Step 1: Encode the query
    query_embedding = get_embedding_model(query)

    # Step 2: Find the most similar FAQ from Pinecone
    best_match = retrieve_faq(query_embedding)
    
    # Step 3: Augment the query with context
    augmented_prompt = get_system_prompt(best_match)

    messages = [{"role": "user","content": query}]

    # Step 4: Use Bedrock API to generate a response with context
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

def main():
    print(f"Chatbot with RAG")

    try:
        ## Populate Pinecone vector database (RUN ONCE)
        #populate_vector_database()

        # Example usage of embedding model
        #prompt = "Can I return a product if I am not satisfied?"
        #most_similar_question = find_most_similar(prompt, faq_vector_db)
        #print(f"The most similar question is: {most_similar_question}")
        #print(f"Answer: " + faq_database[most_similar_question])

        # Test rag chatbot with Pinecone
        #faq_vector_db = create_faq_vector()
        response = rag_chatbot_with_pinecone("do you have discount for students?")
        print(f"RAG: {response}")

        # Test Pinecone retrieval
        #embedding_vector = embedding_model("do you have any promocode?")
        #responseFromVectorDB = retrieve_faq(embedding_vector, index_db, top_k=1)
        #print(f"FAQ: {responseFromVectorDB.matches[0].metadata['answer']}")


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()