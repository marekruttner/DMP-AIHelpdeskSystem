import os
import hashlib
from tqdm import tqdm
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from neo4j import GraphDatabase
from halo import Halo  # For spinner animations
from sklearn.metrics.pairwise import cosine_similarity

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define Milvus collection schema
fields = [
    FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # assuming embeddings are 768-dimensional
    FieldSchema(name="metadata", dtype=DataType.JSON)
]
schema = CollectionSchema(fields, description="Document Embeddings")

# Create or load the Milvus collection
collection_name = "document_embeddings"
if collection_name not in utility.list_collections():  # Corrected check using utility.list_collections()
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Create an index for the "embedding" field before loading the collection
index_params = {
    "index_type": IndexType.HNSW,  # Choose the index type (you can also use FLAT, IVF_FLAT, etc.)
    "metric_type": "COSINE",  # Use the cosine similarity for vector comparison
    "params": {"M": 16, "efConstruction": 200}  # Parameters for HNSW (can be customized based on needs)
}

# Create index on the embedding field
print("Creating index on the embedding field...")
index = Index(collection, "embedding", index_params)
print("Index created successfully.")

# Now load the collection after creating the index
collection.load()  # Load the collection to be ready for search and insert

# Connect to Neo4j
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "testtest"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))


# Function to rename long file names
def rename_long_filename(file_path, max_length=255):
    if len(file_path) > max_length:
        directory, filename = os.path.split(file_path)
        hashed_name = hashlib.sha1(filename.encode()).hexdigest()[:10]
        ext = os.path.splitext(filename)[1]
        new_filename = f"{hashed_name}{ext}"
        new_file_path = os.path.join(directory, new_filename)
        os.rename(file_path, new_file_path)
        print(f"Renamed '{file_path}' to '{new_file_path}' due to long filename.")
        return new_file_path
    return file_path


# Function to load PDF files
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# Function to load Markdown files
def load_md(file_path):
    loader = TextLoader(file_path)
    return loader.load()


# Store embeddings in Milvus
def store_embeddings(document_id, embedding, metadata):
    # Insert embeddings into Milvus
    collection.insert([[document_id], [embedding], [metadata]])  # Correct format for Milvus insertion
    collection.flush()  # Ensure data is saved


# Store graph relationships in Neo4j
def create_document_node(doc_id, content, metadata):
    with driver.session() as session:
        session.run("""
        CREATE (d:Document {doc_id: $doc_id, content: $content, title: $metadata.title, author: $metadata.author})
        """, doc_id=doc_id, content=content, metadata=metadata)


def create_similarity_relationship(doc_id_1, doc_id_2, similarity_score):
    with driver.session() as session:
        session.run("""
        MATCH (d1:Document {doc_id: $doc_id_1}), (d2:Document {doc_id: $doc_id_2})
        CREATE (d1)-[:SIMILAR {score: $similarity_score}]->(d2)
        """, doc_id_1=doc_id_1, doc_id_2=doc_id_2, similarity_score=similarity_score)


# Load documents and build graph
def load_documents_and_build_graph(directory):
    docs = []
    spinner = Halo(text='Loading documents...', spinner='dots')
    spinner.start()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_path = rename_long_filename(file_path)

        try:
            if filename.endswith(".pdf"):
                new_docs = load_pdf(file_path)
                docs.extend(new_docs)
            elif filename.endswith(".md"):
                new_docs = load_md(file_path)
                docs.extend(new_docs)

            # Add documents to Neo4j as nodes
            for doc in new_docs:
                create_document_node(doc.metadata.get("id", filename), doc.page_content, doc.metadata)

        except Exception as e:
            spinner.fail(f"Error loading file {file_path}: {e}")
    spinner.succeed("Documents loaded successfully.")

    # Compute similarities between documents and store relationships
    spinner.start("Computing similarities and building graph...")
    for i, doc_a in enumerate(docs):
        for j, doc_b in enumerate(docs):
            if i != j:
                similarity = compute_similarity(doc_a.page_content, doc_b.page_content)
                if similarity > 0.7:  # Threshold for connecting documents
                    create_similarity_relationship(doc_a.metadata.get("id", ""), doc_b.metadata.get("id", ""),
                                                   similarity)
    spinner.succeed("Graph built successfully.")

    return docs


# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")


# Compute similarity between document contents
def compute_similarity(content_a, content_b):
    embedding_a = embedding_model.embed_documents([content_a])
    embedding_b = embedding_model.embed_documents([content_b])
    return cosine_similarity(embedding_a, embedding_b)[0][0]  # Compute cosine similarity


# Generate embeddings with a progress bar
def generate_embeddings(documents):
    document_texts = [doc.page_content for doc in documents]
    print("Generating embeddings for documents...")
    embeddings = embedding_model.embed_documents(document_texts)  # Batch embedding to improve performance
    return embeddings


# Store embeddings in Milvus
def store_embeddings_in_milvus(documents, embeddings):
    for doc, embedding in zip(documents, embeddings):
        store_embeddings(doc.metadata.get("id", ""), embedding, doc.metadata)


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

    # Expand the set with similar documents from Neo4j
    relevant_docs_set = set([doc["document_id"] for doc in relevant_docs])
    spinner.start("Searching for related documents in Neo4j...")
    for doc_id in relevant_docs_set:
        with driver.session() as session:
            result = session.run("""
            MATCH (d:Document {doc_id: $doc_id})-[:SIMILAR]->(related:Document)
            RETURN related.doc_id
            """, doc_id=doc_id)
            # Add error handling for Neo4j query
            if result.peek() is None:
                print(f"No related documents found for doc_id: {doc_id}")
                continue
            relevant_docs_set.update([record['related.doc_id'] for record in result])
    spinner.succeed("Related documents retrieved.")
    return list(relevant_docs_set)


# Generate a response using Ollama
llm = Ollama(model="llama3.1:8b")

# List to store conversation history
conversation_history = []


# Generate a response to the query using retrieved context
def generate_response(query):
    relevant_docs = get_relevant_docs(query)
    context = " ".join(relevant_docs)

    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history])

    prompt = f"""
                You are a helpdesk assistant that helps the user based on information from provided documents.
                If you don't know how to answer based on the documents, don't answer.

                Instructions:
                - Answer only questions mentioned in documents
                - Use Czech language to answer
                - Refer to documents with information that supports your answer

                Previous Conversation:\n{conversation}\n\nContext: {context}\n\nQuery: {query}\nAnswer:"""

    spinner = Halo(text='Generating response...', spinner='dots')
    spinner.start()
    response = llm.invoke(prompt)
    spinner.succeed("Response generated.")
    conversation_history.append((query, response))
    return response


# Chat functionality in the terminal
def chat():
    print("GraphRAG + RAG Chat System. Type 'exit' to stop.")
    while True:
        query = input("#################\nYou: ")
        if query.lower() == 'exit':
            print("Exiting chat...")
            break
        response = generate_response(query)
        print(f"\n#################\nAI: {response}\n")


# Start chat
if __name__ == "__main__":
    chat()

# Close Neo4j driver when finished
driver.close()
