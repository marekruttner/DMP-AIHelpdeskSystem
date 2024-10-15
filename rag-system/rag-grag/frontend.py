from pymilvus import Collection, connections
from langchain_community.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase
from halo import Halo  # For spinner animations

# Connect to Milvus and Neo4j
connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.load()

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "testtest"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")


# Retrieve relevant documents from Milvus and Neo4j
def get_relevant_docs(query):
    query_embedding = embedding_model.embed_query(query)

    spinner = Halo(text='Searching for relevant documents...', spinner='dots')
    spinner.start()

    # Search in Milvus for the top-k similar embeddings
    search_results = collection.search([query_embedding], "embedding", limit=5,
                                       output_fields=["document_id", "metadata"])
    relevant_docs = [result.entity for result in search_results[0]]

    spinner.succeed("Relevant documents found.")

    # Expand with similar documents from Neo4j
    relevant_docs_set = set([doc["document_id"] for doc in relevant_docs])
    spinner.start("Searching for related documents in Neo4j...")
    for doc_id in relevant_docs_set:
        with driver.session() as session:
            result = session.run("""
            MATCH (d:Document {doc_id: $doc_id})-[:SIMILAR]->(related:Document)
            RETURN related.doc_id
            """, doc_id=doc_id)
            relevant_docs_set.update([record['related.doc_id'] for record in result])

    spinner.succeed("Related documents retrieved.")
    return list(relevant_docs_set)


# Chat functionality
llm = Ollama(model="llama3.1:8b")
conversation_history = []


def generate_response(query):
    relevant_docs = get_relevant_docs(query)
    context = " ".join(relevant_docs)

    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history])

    prompt = f"""
    You are a helpdesk assistant that helps the user based on information from provided documents.
    If you don't know how to answer based on the documents, don't answer.

    Previous Conversation:\n{conversation}\n\nContext: {context}\n\nQuery: {query}\nAnswer:"""

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
