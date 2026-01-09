# RAG Course Exercises

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations demonstrating various advanced RAG techniques and patterns using AWS Bedrock, Pinecone, and FastAPI.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [RAG Techniques Implemented](#rag-techniques-implemented)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Module 5: Production RAG API](#module-5-production-rag-api)
- [Technologies Used](#technologies-used)
- [License](#license)

## 🎯 Overview

This repository contains hands-on exercises for learning and implementing various RAG (Retrieval-Augmented Generation) patterns. Each exercise demonstrates a different approach to improving the quality and relevance of AI-generated responses by leveraging vector databases and advanced prompting techniques.

The project simulates an e-commerce customer support system that sells e-books and IT courses, with multiple knowledge bases (product information, finance policies, technical support) stored in Pinecone vector databases.

## ✨ Features

- **Multiple RAG Patterns**: Six different RAG implementations from basic to advanced
- **Vector Database Integration**: Uses Pinecone for semantic search and document retrieval
- **AWS Bedrock Integration**: Leverages Amazon Titan embeddings and Claude/GPT models
- **Production-Ready API**: FastAPI-based REST API with caching (Module 5)
- **Docker Support**: Containerized deployment for easy scaling
- **Multi-Domain Knowledge**: Separate vector indices for products, finance, and technical support

## 📁 Project Structure

```
rag-course-exercises/
├── ex-00-initial-setup.py          # Database initialization and population
├── ex-01-rag-chatbox.py            # Basic RAG implementation
├── ex-02-multi-query-rag.py        # Multi-Query RAG pattern
├── ex-03-fusion-rag.py             # Fusion RAG with ranking
├── ex-04-hyde-rag.py               # HyDE (Hypothetical Document Embeddings)
├── ex-05-prompt-routing.py         # Intent-based prompt routing
├── ex-06-database-routing.py       # Database routing by intent classification
├── Module 5/                       # Production FastAPI application
│   ├── app/
│   │   ├── main.py                # FastAPI endpoints
│   │   ├── rag_model.py           # RAG logic
│   │   ├── cache.py               # Response caching
│   │   └── config.py              # Configuration
│   ├── Dockerfile                 # Container definition
│   └── requirements.txt           # Python dependencies
└── README.md                       # This file
```

## 🔬 RAG Techniques Implemented

### 1. **Initial Setup** (`ex-00-initial-setup.py`)
- Populates Pinecone vector databases with FAQ data
- Creates separate indices for:
  - E-commerce policies (returns, shipping, orders)
  - Product information (courses, e-books, features)
  - Finance (payments, refunds, discounts)
  - Technical support (login, downloads, platform issues)

### 2. **Basic RAG Chatbot** (`ex-01-rag-chatbox.py`)
- Simple RAG implementation
- Embeds user query using Amazon Titan
- Retrieves most relevant FAQ from Pinecone
- Generates response using Claude Sonnet

**Use Case**: Direct, straightforward questions with clear intent

### 3. **Multi-Query RAG** (`ex-02-multi-query-rag.py`)
- Generates multiple variations of the user's question
- Retrieves documents for each variation
- Combines results for broader context coverage

**Use Case**: Vague or ambiguous questions that could have multiple interpretations

**Example**: "student discount?" → generates:
- "Do you offer discounts for students?"
- "What are the student pricing options?"
- "Are there educational discounts available?"

### 4. **Fusion RAG** (`ex-03-fusion-rag.py`)
- Generates multiple query variations
- Retrieves and ranks results using Reciprocal Rank Fusion (RRF)
- Combines top-ranked documents for response generation

**Use Case**: Complex queries requiring comprehensive information from multiple perspectives

**Benefits**: Better ranking of relevant documents, reduced noise from single queries

### 5. **HyDE RAG** (`ex-04-hyde-rag.py`)
- Hypothetical Document Embeddings approach
- Generates a hypothetical answer to the question
- Embeds and searches with the hypothetical answer
- Retrieves actual relevant documents

**Use Case**: Technical or specific queries where the question and answer vocabulary differ

**Example**: User asks "How do I get my certificate?" → generates hypothetical answer describing the certificate process → finds actual FAQ about certificates

### 6. **Prompt Routing** (`ex-05-prompt-routing.py`)
- Classifies query intent (factual, explanation, guidance)
- Routes to specialized prompts based on intent
- Optimizes response style for query type

**Use Case**: When different types of questions require different response formats

**Intent Types**:
- **Factual**: Direct facts, yes/no answers
- **Explanation**: How/why questions requiring detail
- **Guidance**: Advice or recommendations

### 7. **Database Routing** (`ex-06-database-routing.py`)
- Classifies query into domain categories (product, finance, tech)
- Routes to the appropriate Pinecone index
- Searches only relevant knowledge base

**Use Case**: Multi-domain systems where different knowledge bases exist

**Benefits**: 
- Improved accuracy by searching relevant domain only
- Faster retrieval with smaller search space
- Better separation of concerns

## 🔧 Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- Pinecone account and API key
- Docker (for Module 5 deployment)

### Required AWS Bedrock Models
- `amazon.titan-embed-text-v1` - Text embeddings (1536 dimensions)
- `anthropic.claude-3-sonnet-20240229-v1:0` - Response generation
- `openai.gpt-oss-20b-1:0` - Query generation and classification

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-course-exercises
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install boto3 pinecone-client
```

For Module 5:
```bash
cd "Module 5"
pip install -r requirements.txt
```

## ⚙️ Configuration

Set the following environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export PINECONE_API_KEY=your_pinecone_api_key
export OPENAI_API_KEY=your_openai_api_key  # Only for Module 5
```

Or create a `.env` file:
```
AWS_DEFAULT_REGION=us-east-1
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

### AWS Credentials

Ensure your AWS credentials are configured:
```bash
aws configure
```

## 📖 Usage

### Initialize the Vector Databases

First, populate the Pinecone indices with FAQ data:

```bash
python ex-00-initial-setup.py
```

This creates and populates four Pinecone indices:
- `ecommerce-index` - General e-commerce FAQs
- `product-index` - Product-specific information
- `finance-index` - Payment and billing FAQs
- `tech-index` - Technical support FAQs

### Run Individual Exercises

Each exercise can be run independently:

```bash
# Basic RAG
python ex-01-rag-chatbox.py

# Multi-Query RAG
python ex-02-multi-query-rag.py

# Fusion RAG
python ex-03-fusion-rag.py

# HyDE RAG
python ex-04-hyde-rag.py

# Prompt Routing
python ex-05-prompt-routing.py

# Database Routing
python ex-06-database-routing.py
```

### Modify Queries

Edit the `query` variable in the `main()` function of each file to test different questions:

```python
def main():
    query = "what is your return policy?"  # Change this
    response = rag_chatbot(query)
    print(f"Bot: {response}")
```

## 🏭 Module 5: Production RAG API

A production-ready FastAPI application with caching and containerization.

### Features

- **REST API**: POST endpoint for RAG queries
- **Response Caching**: Stores previous responses to reduce latency and costs
- **Health Check**: GET endpoint to verify service status
- **Docker Support**: Containerized for easy deployment

### Local Development

```bash
cd "Module 5"
uvicorn app.main:app --reload --port 8000
```

### Docker Deployment

1. **Build the image**
```bash
docker build -t ragcourseexercises:latest .
```

2. **Run the container**
```bash
docker run -p 8081:80 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e PINECONE_API_KEY=$PINECONE_API_KEY \
  ragcourseexercises:latest
```

### API Endpoints

**Health Check**
```bash
GET http://localhost:8081/
```

Response:
```json
{
  "message": "RAG API is running"
}
```

**Query Endpoint**
```bash
POST http://localhost:8081/query/
Content-Type: application/json

{
  "query": "What is your return policy?"
}
```

Response:
```json
{
  "response": "Our return policy allows...",
  "source": "RAG"  // or "cache" if cached
}
```

### Testing the API

```bash
curl -X POST http://localhost:8081/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Do you offer student discounts?"}'
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.9+**: Primary programming language
- **AWS Bedrock**: AI model hosting and inference
  - Amazon Titan Embeddings v1 (1536-dim vectors)
  - Anthropic Claude 3 Sonnet
  - OpenAI GPT OSS 20B
- **Pinecone**: Vector database for semantic search
- **FastAPI**: Web framework for REST API (Module 5)
- **Docker**: Containerization and deployment

### Python Libraries
- `boto3`: AWS SDK for Python
- `pinecone-client`: Pinecone vector database client
- `fastapi`: Modern web framework
- `uvicorn`: ASGI server
- `openai`: OpenAI API client (Module 5)

## 📚 Learning Objectives

By completing these exercises, you will learn:

1. **Vector Database Operations**: Storing and querying embeddings in Pinecone
2. **Semantic Search**: Finding relevant documents using vector similarity
3. **Query Augmentation**: Generating multiple query variations for better retrieval
4. **Document Ranking**: Implementing Reciprocal Rank Fusion
5. **Intent Classification**: Routing queries based on detected intent
6. **Prompt Engineering**: Crafting effective system prompts for different scenarios
7. **API Development**: Building production-ready RAG services
8. **Caching Strategies**: Optimizing response times and reducing costs
9. **Containerization**: Deploying AI applications with Docker

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -r requirements.txt
```

**Vector Dimension Mismatch**
- All exercises use Amazon Titan v1 (1536 dimensions)
- If you see dimension errors, ensure all files use `amazon.titan-embed-text-v1`

**OpenAI AttributeError (Module 5)**
- Ensure `openai>=1.0.0` is installed
- Rebuild Docker image with `--no-cache` flag

**Pinecone Index Not Found**
- Run `ex-00-initial-setup.py` to create and populate indices
- Verify index names match in your Pinecone dashboard

**AWS Credentials**
```bash
# Configure AWS CLI
aws configure

# Test access
aws bedrock list-foundation-models --region us-east-1
```

## 📝 Notes

- Each exercise includes detailed docstrings explaining the RAG pattern
- Debug output is included to help understand the flow
- Adjust `top_k`, `temperature`, and `max_tokens` parameters as needed
- Response quality depends on the quality of your FAQ data

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**
