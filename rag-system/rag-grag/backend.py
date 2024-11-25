import os
import hashlib
import json
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

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


model_name = "Seznam/retromae-small-cs"
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
    collection.insert([doc_ids, embeddings.tolist()])
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

# Similarity threshold for precomputed relationships
similarity_threshold = 0.7  # Can be adjusted as needed

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

# Embed and store documents
def process_documents(directory):
    docs = []
    embeddings = []
    doc_ids = []
    contents = []
    metadatas = []

    # Clear the Neo4j graph before processing
    clear_neo4j_graph()

    # Collect all documents
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_path = rename_long_filename(file_path)

        try:
            with Halo(text=f"Processing file {filename}...", spinner="dots") as spinner:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_id = generate_doc_id(content)
                    metadata = {'filename': filename}
                    docs.append({'doc_id': doc_id, 'content': content, 'metadata': metadata})
                    contents.append(content)
                    doc_ids.append(doc_id)
                    metadatas.append(metadata)
                spinner.succeed(f"File {filename} processed successfully.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Store document metadata in Neo4j
    with Halo(text=f"Storing {len(docs)} documents metadata in Neo4j...", spinner="dots") as spinner:
        for i in range(len(docs)):
            doc_id = doc_ids[i]
            content = contents[i]
            metadata = metadatas[i]
            create_document_node(doc_id, content, metadata)
        spinner.succeed(f"Metadata for {len(docs)} documents stored in Neo4j.")

    # Generate embeddings in batches
    with Halo(text=f"Generating embeddings for {len(contents)} documents...", spinner="dots") as spinner:
        embeddings = generate_embeddings(contents)
        spinner.succeed(f"Embeddings generated for {len(contents)} documents.")

    # Store all embeddings in Milvus at once
    if embeddings.any():
        with Halo(text=f"Storing {len(embeddings)} documents in Milvus...", spinner="dots") as spinner:
            store_embeddings(doc_ids, embeddings)
            collection.flush()
            spinner.succeed(f"Stored {len(embeddings)} documents successfully in Milvus.")
    else:
        print("No documents to store.")

    # Perform similarity search for each document
    with Halo(text=f"Performing similarity search and storing relationships in Neo4j...", spinner="dots") as spinner:
        for i in range(len(docs)):
            doc_id = doc_ids[i]
            embedding = embeddings[i]

            # Perform similarity search in Milvus
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            search_embedding = [embedding.tolist()]
            res = collection.search(
                data=search_embedding,
                anns_field="embedding",
                param=search_params,
                limit=10,
                expr=None,
                output_fields=["document_id"]
            )
            hits = res[0]
            for hit in hits:
                similar_doc_id = hit.entity.get('document_id')
                similarity_score = 1 - hit.distance  # Since distance is 1 - cosine similarity
                # Exclude self and apply similarity threshold
                if similar_doc_id != doc_id and similarity_score > similarity_threshold:
                    create_similarity_relationship(doc_id, similar_doc_id, similarity_score)
        spinner.succeed("Similarity search and relationships stored in Neo4j.")

# Run the backend process
if __name__ == "__main__":
    # Specify your directory where documents are stored
    directory = "/home/marek/rag-documents/"
    process_documents(directory)

# Close Neo4j driver when finished
driver.close()
