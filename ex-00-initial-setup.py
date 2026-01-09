import boto3
import os
import json
from pinecone import Pinecone

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

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

product_faq = {
    "What types of products do you offer?": "We offer a variety of e-books and online courses tailored for IT professionals.",
    "Are your e-books downloadable?": "Yes, all our e-books are available for download immediately after purchase.",
    "Do your courses have certifications?": "Yes, upon completion of our courses, you will receive a certificate of completion.",
    "Are the courses live or pre-recorded?": "Our courses are pre-recorded, allowing you to learn at your own pace.",
    "Can I preview a course before purchasing?": "Yes, each course page includes a preview section with sample videos and course material.",
    "Are there any prerequisites for your courses?": "Each course description includes any prerequisites needed, if applicable.",
    "How long do I have access to a course after purchasing?": "Once purchased, you have lifetime access to the course material.",
    "Can I access my e-books and courses on multiple devices?": "Yes, you can access your purchases on any device with internet access.",
    "Are the courses beginner-friendly?": "Yes, we offer courses for all levels, from beginners to advanced professionals.",
    "What if a course I want is out of stock?": "Digital courses do not run out of stock, so you can purchase anytime.",
    "Do you have any group discounts for teams?": "Yes, we offer group rates for teams. Please contact support for more details.",
    "Can I gift a course or e-book to someone else?": "Currently, our system doesn’t support gifting, but you may share your purchased, if allowed.",
    "Are there any free courses or e-books?": "Yes, we offer a selection of free resources on our website.",
    "Can I download course videos for offline use?": "Currently, courses are available only for online streaming.",
    "Is there a limit to the number of e-books I can download?": "No, once purchased, you can download your e-books as many times as needed.",
    "Do courses have subtitles or closed captions?": "Yes, most of our courses include subtitles in multiple languages.",
    "Is there a way to contact the course instructor?": "Some courses offer a discussion forum or Q&A section for this purpose.",
    "How often do you update your course material?": "We regularly update course materials to ensure they reflect the latest industry standards.",
    "Can I suggest a new course topic?": "Yes, we welcome suggestions. Feel free to reach out through our contact form.",
}

finance_faq = {
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay.",
    "Is there a money-back guarantee on courses?": "Yes, we offer a 30-day money-back guarantee on all courses.",
    "Do I get a receipt for my purchase?": "Yes, you will receive a digital receipt by email immediately after purchase.",
    "Are there any hidden fees?": "No, there are no hidden fees. All costs are outlined at checkout.",
    "Can I pay in installments?": "For some courses, we offer installment plans. Details are available on the course page.",
    "Do you offer student discounts?": "Yes, students can apply for a discount. Contact support for more information.",
    "Is there a refund policy for e-books?": "E-books are non-refundable once downloaded, as they are digital products.",
    "Will I be charged any taxes on my purchase?": "Taxes are applied based on local regulations, and will be shown at checkout.",
    "Can I use multiple discounts on one purchase?": "Only one discount code can be applied per transaction.",
    "Is my payment information secure?": "Yes, we use secure, encrypted payment processing to protect your data.",
    "How do I apply a discount code?": "Enter the discount code at checkout in the designated field.",
    "Are your courses tax-deductible as a business expense?": "Depending on your location and business, courses may be deductible. Consult a tax advisor.",
    "Can I pay by bank transfer?": "Currently, we only accept online payments via credit cards, PayPal, and Apple Pay.",
    "Why was my payment declined?": "Please ensure your card details are correct or contact your bank for assistance.",
    "Do you provide invoices for purchases?": "Yes, you can download an invoice from your account dashboard after purchase.",
    "Is there a charge for currency conversion?": "Currency conversion fees may apply depending on your bank’s policies.",
    "What should I do if I’m double-charged?": "Please contact our support team with proof, and we will assist you.",
    "Are there any exchange rates applied on payments?": "Payments are processed in USD, and your bank may apply exchange rates if using another currency.",
    "How can I cancel my order?": "For downloadable items like e-books and courses, cancellations are only possible before downloading.",
}

tech_faq = {
    "I’m having trouble accessing my course. What should I do?": "Ensure you’re logged in and have a stable internet connection. Contact support if issues persist.",
    "What devices are compatible with your platform?": "Our platform works on most devices, including desktops, laptops, tablets, and smartphones.",
    "Why is my video not playing?": "Try refreshing your browser or clearing your cache. If the problem continues, contact tech support.",
    "Is there a mobile app for accessing courses?": "Currently, our platform is accessible via web browser but does not have a dedicated app.",
    "What browsers are recommended for the best experience?": "We recommend using the latest versions of Chrome, Firefox, or Safari.",
    "How can I reset my password?": "Use the 'Forgot Password' link on the login page to reset your password.",
    "Why is my download not starting?": "Check your internet connection and try again. If the issue persists, contact support.",
    "Can I access courses on multiple devices?": "Yes, you can access your account and courses on multiple devices.",
    "Why is my e-book not opening on my device?": "Ensure you have a compatible e-reader or app. Contact support if the issue continues.",
    "How do I change my account email?": "Go to 'Account Settings' to update your email.",
    "What should I do if I encounter a bug?": "Report any issues through our support portal, and our team will investigate.",
    "Are course videos available in HD?": "Yes, all videos are available in HD quality. You can adjust video quality as needed.",
    "Why am I logged out unexpectedly?": "This could be due to security protocols. Ensure your session is active and re-login if needed.",
    "How can I download my e-books for offline use?": "Once purchased, e-books are available for download through your account.",
    "Can I speed up or slow down course videos?": "Yes, you can control playback speed through the video player settings.",
    "Why isn’t my certificate downloading?": "Check your browser’s settings to allow downloads. Contact support if issues persist.",
    "How can I enable subtitles in a video?": "Use the 'CC' button on the video player to enable subtitles if available.",
    "Do I need a specific software to view the e-books?": "Our e-books are in PDF format, so any PDF viewer should work.",
    "I’m experiencing audio issues. What should I do?": "Check your device’s audio settings and ensure the video player volume is not muted.",
}

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
    
def populate_vector_database(database, index_name):
    index_db = pc.Index(index_name)

    data_to_upsert = []
    for i, (q, a) in enumerate(database.items()):
        data_to_upsert.append(
            {
                "id": str(i),
                "values": get_embedding_model(q),
                "metadata": {"question": q, "answer": a}
            }
        ) 
    index_db.upsert(vectors=data_to_upsert, namespace="ns1")

    print(f"Uploaded {len(data_to_upsert)} FAQ entries to the vector database.")

def main():
    print(f"Initial setup started.")

    try:

        print(f"Populating Ecommerce FAQ database...")
        populate_vector_database(faq_database, "ecommerce-index")

        print(f"Populating Product FAQ database...")
        populate_vector_database(product_faq, "product-index")

        print(f"Populating Finance FAQ database...")
        populate_vector_database(finance_faq, "finance-index")   

        print(f"Populating Tech FAQ database...")
        populate_vector_database(tech_faq, "tech-index")

        print(f"Initial setup finished.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()