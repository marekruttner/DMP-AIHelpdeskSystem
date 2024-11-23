from pymilvus import Collection, connections
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.llms import Ollama

# Connect to Milvus and Neo4j
connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.load()

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "testtest"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Initialize Hugging Face model
model_name = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embeddings

def get_relevant_docs(query):
    query_embedding = generate_embedding(query)

    spinner = Halo(text='Searching for relevant documents...', spinner='dots')
    spinner.start()

    # Step 1: Milvus search
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["document_id"]
    )
    initial_relevant_docs = [hit.entity.get("document_id") for hit in search_results[0]]
    spinner.succeed("Relevant documents found.")

    # Use a set to avoid duplicates
    relevant_doc_ids = set(initial_relevant_docs)

    # Step 2: Fetch precomputed related documents in batch
    spinner.start("Fetching precomputed related documents from Neo4j...")
    with driver.session() as session:
        result = session.run("""
        MATCH (d:Document)-[:SIMILAR_TO {type: 'PRECOMPUTED'}]->(related:Document)
        WHERE d.doc_id IN $doc_ids
        RETURN related.doc_id AS doc_id
        """, doc_ids=list(relevant_doc_ids))
        for record in result:
            relevant_doc_ids.add(record['doc_id'])
    spinner.succeed("Precomputed related documents retrieved.")

    # Step 3: Fetch dynamically related documents in batch
    spinner.start("Fetching dynamically related documents from Neo4j...")
    with driver.session() as session:
        result = session.run("""
        MATCH (d:Document)-[:SIMILAR_TO]->(related:Document)
        WHERE d.doc_id IN $doc_ids AND related.doc_id <> d.doc_id AND NOT EXISTS {
            MATCH (d)-[:SIMILAR_TO {type: 'PRECOMPUTED'}]->(related)
        }
        RETURN related.doc_id AS doc_id
        """, doc_ids=list(relevant_doc_ids))
        for record in result:
            relevant_doc_ids.add(record['doc_id'])
    spinner.succeed("Dynamically related documents retrieved.")

    # Step 4: Fetch all document contents in a single query
    spinner.start("Fetching document contents from Neo4j...")
    document_contents = []
    with driver.session() as session:
        result = session.run("""
        MATCH (d:Document)
        WHERE d.doc_id IN $doc_ids
        RETURN d.content AS content
        """, doc_ids=list(relevant_doc_ids))
        for record in result:
            document_contents.append(record['content'])
    spinner.succeed("Document contents retrieved.")

    return document_contents

# Chat functionality

llm = Ollama(model="llama3.1:8b")
conversation_history = []

def generate_response(query):
    # Fetch relevant documents
    relevant_docs_content = get_relevant_docs(query)

    # Limit the number of documents (e.g., top 2)
    max_docs = 2
    relevant_docs_content = relevant_docs_content[:max_docs]

    # Combine document contents into the context
    context = "\n".join(relevant_docs_content)

    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history])

    prompt = f"""
    You are a helpdesk assistant who assists users based on information from the provided documents.

    Your primary goal is to help users solve their problems by providing simple, clear, and step-by-step instructions suitable for non-technical individuals.

    Assume that the user is experiencing difficulties and needs your assistance, so use a kind, patient, and empathetic tone.

    In the context, you have information from documents; use them to answer the user's questions.

    FOLLOW THESE INSTRUCTIONS:

    - Use the same language as the user's input.
    - Provide step-by-step instructions in simple language, avoiding technical jargon.
    - Ensure your explanations are clear and easy to follow.
    - Be patient and empathetic throughout the conversation.
    - If you cannot answer clearly, politely ask the user for more detailed information.
    - When providing your answer, refer to the page from which the source document is taken. URL of ducument.
    - Only use information available in the provided context.

    Previous conversation: {conversation}

    Context: {context}

    Question: {query}

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
