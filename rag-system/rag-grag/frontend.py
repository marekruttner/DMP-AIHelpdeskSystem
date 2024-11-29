import os
import hashlib
import json
import psycopg2
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can restrict it to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# PostgreSQL connection
DB_CONFIG = {
    "dbname": "chatdb",
    "user": "admin",
    "password": "adminadmin",
    "host": "localhost",  # Use the service name from docker-compose
    "port": "5432"
}

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.load()

# Connect to Neo4j
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "testtest"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Initialize Hugging Face model
model_name = "Seznam/retromae-small-cs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize LLM and conversation history
llm = Ollama(model="llama3.1:8b")
conversation_history = []  # Tracks user queries and responses

# Pydantic models for request/response validation
class UserCredentials(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    user_id: int
    query: str

class RegistrationResponse(BaseModel):
    message: str

class LoginResponse(BaseModel):
    user_id: Optional[int]
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[str]

# Function to generate embeddings for a list of texts
def generate_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

# Save conversation to PostgreSQL
def save_conversation(user_id, query, response):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        sanitized_query = query.encode("utf-8", "replace").decode("utf-8")
        sanitized_response = response.encode("utf-8", "replace").decode("utf-8")
        cursor.execute("""
            INSERT INTO user_conversations (user_id, conversation)
            VALUES (%s, %s)
        """, (user_id, f"User: {sanitized_query}\nAI: {sanitized_response}"))
        connection.commit()
    except Exception as e:
        print(f"Error saving conversation: {e}")
    finally:
        cursor.close()
        connection.close()

# Fetch conversation history for a user
@app.get("/chat/history/{user_id}")
def get_chat_history(user_id: int):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT conversation FROM user_conversations WHERE user_id = %s ORDER BY id ASC
        """, (user_id,))
        result = cursor.fetchall()
        history = [record[0] for record in result]
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")
    finally:
        cursor.close()
        connection.close()

# Fetch relevant documents
def get_relevant_docs(query):
    query_embedding = generate_embeddings([query])[0]

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    search_results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["document_id"]
    )

    relevant_doc_ids = [hit.entity.get("document_id") for hit in search_results[0]]
    documents = []
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Document)
            WHERE d.doc_id IN $doc_ids
            RETURN d.content AS content, d.metadata AS metadata
        """, doc_ids=relevant_doc_ids)
        for record in result:
            metadata = json.loads(record['metadata'])
            documents.append({
                'content': record['content'],
                'metadata': metadata,
                'filename': metadata.get('filename', 'Unknown')
            })
    return documents

# User registration endpoint
@app.post("/register", response_model=RegistrationResponse)
def register_user(credentials: UserCredentials):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()

    hashed_password = hashlib.sha256(credentials.password.encode()).hexdigest()
    try:
        cursor.execute("""
            INSERT INTO users (username, password)
            VALUES (%s, %s)
        """, (credentials.username, hashed_password))
        connection.commit()
        return {"message": "Registration successful"}
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        cursor.close()
        connection.close()

# User login endpoint
@app.post("/login", response_model=LoginResponse)
def login_user(credentials: UserCredentials):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT id, password FROM users WHERE username = %s
        """, (credentials.username,))
        result = cursor.fetchone()
        if result:
            user_id, stored_hash = result
            input_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
            if input_hash == stored_hash:
                return {"user_id": user_id, "message": "Login successful"}
        raise HTTPException(status_code=401, detail="Invalid username or password")
    finally:
        cursor.close()
        connection.close()

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def generate_response(request: QueryRequest):
    if not request.new_chat:
        # Use conversation history if not starting a new chat
        conversation = "\n".join([
            f"User: {q}\nAI: {r}" for q, r in conversation_history[-50:]
        ])
    else:
        conversation = ""  # Start with an empty conversation

    documents = get_relevant_docs(request.query)
    max_docs = 5
    documents = documents[:max_docs]

    context = ""
    for doc in documents:
        content = doc['content']
        filename = doc['filename']
        content = content.encode('utf-8', errors='replace').decode('utf-8')
        filename = filename.encode('utf-8', errors='replace').decode('utf-8')
        context += f"{content}\n(Source: {filename})\n\n"

    prompt = f"""
        You are a helpdesk assistant who assists users based on information from the provided documents.

        Your primary goal is to help users solve their problems by providing simple, clear, and step-by-step instructions suitable for non-technical individuals.

        Previous conversation:
        {conversation}

        Context:
        {context}

        Question:
        {request.query}

        Your kind and helpful Answer:
    """

    response = llm.invoke(prompt).strip()
    response = response.encode('utf-8', errors='replace').decode('utf-8')

    if not request.new_chat:
        conversation_history.append((request.query, response))  # Save to in-memory history
    save_conversation(request.user_id, request.query, response)

    source_names = ", ".join([doc['filename'] for doc in documents])
    source_names = source_names.encode('utf-8', errors='replace').decode('utf-8')
    formatted_response = f"{response}\n\nSources:\n{source_names.replace(',', '\n')}"

    return {"response": formatted_response, "sources": source_names}