import os
import hashlib
import json
import psycopg2
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain_community.llms import Ollama

# PostgreSQL connection
DB_CONFIG = {
    "dbname": "chatdb",
    "user": "admin",
    "password": "adminadmin",
    "host": "localhost",
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
    cursor.execute("""
        INSERT INTO user_conversations (user_id, conversation)
        VALUES (%s, %s)
    """, (user_id, f"User: {query}\nAI: {response}"))
    connection.commit()
    cursor.close()
    connection.close()


# Fetch relevant documents
def get_relevant_docs(query):
    query_embedding = generate_embeddings([query])[0]

    spinner = Halo(text='Searching for relevant documents...', spinner='dots')
    spinner.start()

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    search_results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["document_id"]
    )
    spinner.succeed("Relevant documents found.")

    relevant_doc_ids = [hit.entity.get("document_id") for hit in search_results[0]]
    documents = []
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Document)
            WHERE d.doc_id IN $doc_ids
            RETURN d.content AS content, d.metadata AS metadata
        """, doc_ids=relevant_doc_ids)
        for record in result:
            documents.append({'content': record['content'], 'metadata': json.loads(record['metadata'])})
    return documents


# User registration
def register_user():
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    print("Sign Up: Enter your details")
    username = input("Username: ")
    password = input("Password: ")
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute("""
            INSERT INTO users (username, password)
            VALUES (%s, %s)
        """, (username, hashed_password))
        connection.commit()
        print("Registration successful. You can now log in.")
    except psycopg2.IntegrityError:
        print("Username already exists. Try a different one.")
    finally:
        cursor.close()
        connection.close()


# Chat functionality
llm = Ollama(model="llama3.1:8b")
conversation_history = []


def generate_response(query, user_id):
    documents = get_relevant_docs(query)
    max_docs = 3
    documents = documents[:max_docs]

    context = ""
    for doc in documents:
        content = doc['content']
        metadata = doc['metadata']
        source_info = f"(Source: {metadata.get('filename', 'Unknown')})"
        context += f"{content}\n{source_info}\n\n"

    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history[-50:]])
    prompt = f"""
        You are a helpdesk assistant who assists users based on information from the provided documents.

        Your primary goal is to help users solve their problems by providing simple, clear, and step-by-step instructions suitable for non-technical individuals.

        Assume that the user is experiencing difficulties and needs your assistance, so use a kind, patient, and empathetic tone.

        In the context, you have information from documents; use them to answer the user's questions.

        FOLLOW THESE INSTRUCTIONS:

        - Use the same language as the user's input.
        - Provide step-by-step instructions in simple language, avoiding technical jargon.
        - Ensure your explanations are clear, accurate, and easy to follow.
        - Be patient and empathetic throughout the conversation.
        - If you cannot answer clearly, politely ask the user for more detailed information.
        - Always verify the information in the provided context before answering.
        - Do not provide information that is not included in the provided context.
        - When providing your answer, refer to the page from which the source document is taken, including the URL of the document if available.
        - Only use information available in the provided context.
        - Avoid making assumptions or providing information beyond what is given in the documents.

        Previous conversation:
        {conversation}

        Context:
        {context}

        Question:
        {query}

        Your kind and helpful Answer:
    """

    spinner = Halo(text='Generating response...', spinner='dots')
    spinner.start()
    response = llm.invoke(prompt)
    spinner.succeed("Response generated.")
    conversation_history.append((query, response))
    save_conversation(user_id, query, response)
    return response


# Start chat
def chat():
    print("GraphRAG + RAG Chat System. Type 'exit' to stop.")
    print("1. Log In\n2. Sign Up")
    choice = input("Select an option: ")
    if choice == '2':
        register_user()
        return
    user_id = int(input("Enter your user ID: "))
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Exiting chat...")
            break
        response = generate_response(query, user_id)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    chat()

driver.close()
