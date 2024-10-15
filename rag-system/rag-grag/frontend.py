from pymilvus import Collection, connections
from langchain_community.llms import Ollama
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


def get_relevant_docs(query):
    query_embedding = embedding_model.embed_query(query)

    spinner = Halo(text='Searching for relevant documents...', spinner='dots')
    spinner.start()

    # Step 1: Define search parameters and search in Milvus for the top-k similar embeddings
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}  # Adjust "ef" as per your requirement
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,  # Search parameters
        limit=5,
        output_fields=["document_id"]  # Removed "metadata" to avoid the error
    )

    # Extract the document IDs from search results
    relevant_docs = [result.id for result in search_results[0]]  # Accessing 'id' directly from the Hit object

    spinner.succeed("Relevant documents found.")

    # Step 2: Retrieve the document content from Neo4j for relevant documents
    document_contents = []
    spinner.start("Fetching document contents from Neo4j...")
    for doc_id in relevant_docs:
        with driver.session() as session:
            result = session.run("""
            MATCH (d:Document {doc_id: $doc_id})
            RETURN d.content AS content
            """, doc_id=doc_id)
            for record in result:
                document_contents.append(record['content'])  # Collect document content

    spinner.succeed("Document contents retrieved.")

    # Step 3: Retrieve static (precomputed) related documents from Neo4j
    spinner.start("Searching for precomputed related documents in Neo4j...")
    for doc_id in relevant_docs:
        with driver.session() as session:
            result = session.run("""
            MATCH (d:Document {doc_id: $doc_id})-[:SIMILAR_TO {type: 'PRECOMPUTED'}]->(related:Document)
            RETURN related.doc_id
            """, doc_id=doc_id)
            relevant_docs.extend([record['related.doc_id'] for record in result])

    spinner.succeed("Precomputed related documents retrieved.")

    # Step 4: Dynamically find additional relationships based on query context
    spinner.start("Dynamically finding additional relationships in Neo4j...")
    for doc_id in relevant_docs:
        with driver.session() as session:
            result = session.run("""
            MATCH (d:Document {doc_id: $doc_id})-[:SIMILAR_TO]->(related:Document)
            WHERE related.doc_id <> $doc_id AND NOT EXISTS { MATCH (d)-[:SIMILAR_TO {type: 'PRECOMPUTED'}]->(related) }
            RETURN related.doc_id
            """, doc_id=doc_id)
            relevant_docs.extend([record['related.doc_id'] for record in result])

    spinner.succeed("Dynamically related documents retrieved.")

    return document_contents  # Return document contents instead of IDs


# Chat functionality
llm = Ollama(model="llama3.1:8b")
conversation_history = []


def generate_response(query):
    relevant_docs_content = get_relevant_docs(query)
    context = "\n".join(relevant_docs_content)  # Combine document contents into the context

    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history])

    prompt = f"""
    You are a helpdesk assistant that helps the user based on information from provided documents. 
    Expect that user have some difficulties and need your help so have kind tone voice. 
    
    In the context you have informations from documents and use them for answering.
    

    FOLLOW THESE INSTRUCTIONS:
    - use same language as is on the input
    - ask user for detail informations if you can not answer clearly


    \n\nPrevious conversation: {conversation}\n\nContext: {context}\n\nQuestion: {query}\nYour kind helpful Answer:"""

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
