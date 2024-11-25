import os
import hashlib
import json
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain_community.llms import Ollama

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
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

def get_relevant_docs(query):
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]

    spinner = Halo(text='Searching for relevant documents...', spinner='dots')
    spinner.start()

    # Step 1: Milvus search to find top K relevant documents
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    search_results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["document_id"]
    )
    spinner.succeed("Relevant documents found.")

    # Extract document IDs
    relevant_doc_ids = [hit.entity.get("document_id") for hit in search_results[0]]

    # Fetch document contents and metadata from Neo4j
    spinner.start("Fetching document contents from Neo4j...")
    documents = []
    with driver.session() as session:
        result = session.run("""
        MATCH (d:Document)
        WHERE d.doc_id IN $doc_ids
        RETURN d.content AS content, d.metadata AS metadata
        """, doc_ids=relevant_doc_ids)
        for record in result:
            documents.append({'content': record['content'], 'metadata': json.loads(record['metadata'])})
    spinner.succeed("Document contents retrieved.")

    return documents

# Chat functionality

llm = Ollama(model="llama3.1:8b")
conversation_history = []

def generate_response(query):
    # Fetch relevant documents
    documents = get_relevant_docs(query)

    # Limit the number of documents (e.g., top N)
    max_docs = 3
    documents = documents[:max_docs]

    # Combine document contents into the context with metadata
    context = ""
    for doc in documents:
        content = doc['content']
        metadata = doc['metadata']
        source_info = f"(Source: {metadata.get('filename', 'Unknown')})"
        context += f"{content}\n{source_info}\n\n"

    # Include the conversation history in the prompt
    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history[-50:]])  # Include last 50 exchanges

    # Construct the prompt
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
    return response

# Start chat
def chat():
    print("GraphRAG + RAG Chat System. Type 'exit' to stop.")
    while True:
        query = input("#################\nYou: ")
        if query.lower() == 'exit':
            print("Exiting chat...")
            break
        response = generate_response(query)
        print(f"\n#################\nAI: {response}\n")

if __name__ == "__main__":
    chat()

# Close Neo4j driver when finished
driver.close()
