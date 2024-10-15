import os
import hashlib
import json  # For stringifying metadata if necessary
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase
from halo import Halo  # Halo for terminal progress indicators

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define Milvus collection schema (Updated to 1024 dimensions)
fields = [
    FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Updated to 1024 dimensions
]
schema = CollectionSchema(fields, description="Document Embeddings")

# Clear existing collection if it exists and recreate it
collection_name = "document_embeddings"
if collection_name in utility.list_collections():
    with Halo(text=f"Collection '{collection_name}' exists. Dropping it to clear the database...", spinner="dots") as spinner:
        collection = Collection(name=collection_name)
        collection.drop()
        spinner.succeed(f"Collection '{collection_name}' dropped successfully.")

# Create a new collection
collection = Collection(name=collection_name, schema=schema)

# Create an index for the "embedding" field before loading the collection
index_params = {
    "index_type": IndexType.HNSW,
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}

# Create index on the embedding field
with Halo(text="Creating index on the embedding field...", spinner="dots") as spinner:
    index = Index(collection, "embedding", index_params)
    spinner.succeed("Index created successfully.")

# Load the collection to be ready for search and insert
collection.load()

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

# Load PDF files
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Load Markdown files
def load_md(file_path):
    loader = TextLoader(file_path)
    return loader.load()

# Clear Neo4j graph (all nodes and relationships)
def clear_neo4j_graph():
    with driver.session() as session:
        with Halo(text="Clearing Neo4j graph...", spinner="dots") as spinner:
            session.run("MATCH (n) DETACH DELETE n")
            spinner.succeed("Neo4j graph cleared successfully.")

# Store embeddings in Milvus (only embeddings, metadata handled separately)
def store_embeddings(embeddings):
    # Insert data into Milvus: only embedding list
    collection.insert([embeddings])
    collection.flush()

# Store graph relationships and metadata in Neo4j
def create_document_node(doc_id, content, metadata):
    with driver.session() as session:
        # Optional: Stringify metadata if needed
        metadata_json = json.dumps(metadata)
        session.run("""
        CREATE (d:Document {doc_id: $doc_id, content: $content, metadata: $metadata_json})
        """, doc_id=doc_id, content=content, metadata_json=metadata_json)

# Create relationship between similar documents in Neo4j (Precomputed)
def create_similarity_relationship(doc_id_1, doc_id_2, similarity_score):
    with driver.session() as session:
        session.run("""
        MATCH (d1:Document {doc_id: $doc_id_1}), (d2:Document {doc_id: $doc_id_2})
        CREATE (d1)-[:SIMILAR_TO {score: $similarity_score, type: 'PRECOMPUTED'}]->(d2)
        """, doc_id_1=doc_id_1, doc_id_2=doc_id_2, similarity_score=similarity_score)

# Function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    return sum(a * b for a, b in zip(embedding1, embedding2)) / (
            (sum(a ** 2 for a in embedding1) ** 0.5) * (sum(b ** 2 for b in embedding2) ** 0.5)
    )

# Compute and store similarities for documents
def compute_similarity_and_store(doc_id, embedding, doc_ids, embeddings):
    for i, existing_embedding in enumerate(embeddings):
        existing_doc_id = doc_ids[i]
        similarity_score = cosine_similarity(embedding, existing_embedding)

        if similarity_score > similarity_threshold:
            create_similarity_relationship(doc_id, existing_doc_id, similarity_score)

# Similarity threshold for precomputed relationships
similarity_threshold = 0.75  # Can be adjusted as needed

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Embed and store documents
def process_documents(directory):
    docs = []
    embeddings = []
    doc_ids = []

    # Clear the Neo4j graph before processing
    clear_neo4j_graph()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_path = rename_long_filename(file_path)

        try:
            new_docs = []
            with Halo(text=f"Processing file {filename}...", spinner="dots") as spinner:
                if filename.endswith(".pdf"):
                    new_docs = load_pdf(file_path)
                    docs.extend(new_docs)
                elif filename.endswith(".md"):
                    new_docs = load_md(file_path)
                    docs.extend(new_docs)

                spinner.succeed(f"File {filename} processed successfully.")

            # Process each document
            for doc in new_docs:
                doc_id = hash(doc.page_content)  # Unique document ID based on content
                content = doc.page_content
                metadata = doc.metadata  # Metadata stored in Neo4j

                # Get embedding from the document content
                with Halo(text=f"Generating embedding for document {doc_id}...", spinner="dots") as spinner:
                    embedding = embedding_model.embed_documents([content])[0]

                    # Check if the embedding has the correct length (1024)
                    if len(embedding) != 1024:
                        raise ValueError(
                            f"Embedding size mismatch: expected 1024, got {len(embedding)} for document {doc_id}")

                    embeddings.append(embedding)
                    doc_ids.append(doc_id)

                    spinner.succeed(f"Embedding generated for document {doc_id}.")

                # Store document metadata in Neo4j
                with Halo(text=f"Storing metadata for document {doc_id} in Neo4j...", spinner="dots") as spinner:
                    create_document_node(doc_id, content, metadata)
                    spinner.succeed(f"Metadata stored for document {doc_id} in Neo4j.")

                # Compute similarity with other documents and store relationships
                compute_similarity_and_store(doc_id, embedding, doc_ids, embeddings)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Store all embeddings in Milvus at once
    if embeddings:
        with Halo(text=f"Storing {len(embeddings)} documents in Milvus...", spinner="dots") as spinner:
            store_embeddings(embeddings)
            spinner.succeed(f"Stored {len(embeddings)} documents successfully in Milvus.")
    else:
        print("No documents to store.")

# Run the backend process
if __name__ == "__main__":
    # Specify your directory where documents are stored
    directory = "/home/marek/rag-documents/"
    process_documents(directory)

# Close Neo4j driver when finished
driver.close()
