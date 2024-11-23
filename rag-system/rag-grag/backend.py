import os
import hashlib
import json
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define Milvus collection schema with auto_id=False
fields = [
    FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
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


model_name = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


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
    # Implement your PDF loading logic here
    pass

# Load Markdown files
def load_md(file_path):
    # Implement your Markdown loading logic here
    pass

# Clear Neo4j graph (all nodes and relationships)
def clear_neo4j_graph():
    with driver.session() as session:
        with Halo(text="Clearing Neo4j graph...", spinner="dots") as spinner:
            session.run("MATCH (n) DETACH DELETE n")
            spinner.succeed("Neo4j graph cleared successfully.")

# Function to generate consistent document IDs
def generate_doc_id(content):
    sha256_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    doc_id = int(sha256_hash, 16) % (2 ** 63 - 1)
    return doc_id

# Store embeddings in Milvus
def store_embeddings(doc_ids, embeddings):
    collection.insert([doc_ids, embeddings])
    collection.flush()

# Store graph relationships and metadata in Neo4j
def create_document_node(doc_id, content, metadata):
    with driver.session() as session:
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
    return float(torch.nn.functional.cosine_similarity(
        torch.tensor(embedding1), torch.tensor(embedding2), dim=0))

# Compute and store similarities for documents
def compute_similarity_and_store(doc_id, embedding, doc_ids, embeddings):
    for i, existing_embedding in enumerate(embeddings):
        existing_doc_id = doc_ids[i]
        similarity_score = cosine_similarity(embedding, existing_embedding)

        if similarity_score > similarity_threshold:
            create_similarity_relationship(doc_id, existing_doc_id, similarity_score)

# Similarity threshold for precomputed relationships
similarity_threshold = 0.7  # Can be adjusted as needed

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    return embeddings

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
                # Implement your document loading logic here
                # For example, if your documents are plain text files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    new_docs.append({'page_content': content, 'metadata': {'filename': filename}})
                spinner.succeed(f"File {filename} processed successfully.")

            # Process each document
            for doc in new_docs:
                doc_id = generate_doc_id(doc['page_content'])
                content = doc['page_content']
                metadata = doc['metadata']  # Metadata stored in Neo4j

                # Get embedding from the document content
                with Halo(text=f"Generating embedding for document {doc_id}...", spinner="dots") as spinner:
                    embedding = generate_embedding(content)

                    # Check if the embedding has the correct length (256)
                    if len(embedding) != 256:
                        raise ValueError(
                            f"Embedding size mismatch: expected 256, got {len(embedding)} for document {doc_id}")

                    embeddings.append(embedding)
                    doc_ids.append(doc_id)

                    spinner.succeed(f"Embedding generated for document {doc_id}.")

                # Store document metadata in Neo4j
                with Halo(text=f"Storing metadata for document {doc_id} in Neo4j...", spinner="dots") as spinner:
                    create_document_node(doc_id, content, metadata)
                    spinner.succeed(f"Metadata stored for document {doc_id} in Neo4j.")

                # Compute similarity with other documents and store relationships
                compute_similarity_and_store(doc_id, embedding, doc_ids[:-1], embeddings[:-1])

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Store all embeddings in Milvus at once
    if embeddings:
        with Halo(text=f"Storing {len(embeddings)} documents in Milvus...", spinner="dots") as spinner:
            store_embeddings(doc_ids, embeddings)
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
