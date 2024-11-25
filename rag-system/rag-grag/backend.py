import os
import hashlib
import json
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re
from PyPDF2 import PdfReader

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

# Model setup
model_name = "Seznam/retromae-small-cs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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

# Create relationship between documents in Neo4j
def create_relationship(doc_id_1, doc_id_2, relationship_type, extra_data=None):
    with driver.session() as session:
        result = session.run("""
        MATCH (d1:Document {doc_id: $doc_id_1}), (d2:Document {doc_id: $doc_id_2})
        CREATE (d1)-[:RELATED {type: $relationship_type, extra: $extra_data}]->(d2)
        RETURN d1, d2
        """, doc_id_1=doc_id_1, doc_id_2=doc_id_2, relationship_type=relationship_type, extra_data=json.dumps(extra_data))
        if result.peek() is None:
            print(f"Failed to create relationship: {doc_id_1} -> {doc_id_2}")

# Extract links and references from content
def extract_links_and_references(content, doc_map):
    links = re.findall(r"\[.*?\]\((.*?)\)", content)
    relationships = []
    for link in links:
        for ref_doc, ref_doc_id in doc_map.items():
            if link in ref_doc:
                relationships.append((ref_doc_id, "LINK"))
    return links, relationships

# Read content from a file (Markdown or PDF)
def read_file_content(file_path):
    if file_path.endswith(".md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
    return None

# Enrich metadata with more details
def extract_metadata(file_path, content):
    metadata = {
        "filename": os.path.basename(file_path),
        "size": os.path.getsize(file_path),
        "word_count": len(content.split()),
        "links": []
    }
    if file_path.endswith(".md"):
        metadata["type"] = "Markdown"
    elif file_path.endswith(".pdf"):
        metadata["type"] = "PDF"
    return metadata

# Embed and store documents
def process_documents(directory):
    docs = {}
    embeddings = []
    doc_ids = []

    # Clear the Neo4j graph before processing
    clear_neo4j_graph()

    # Collect all documents
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            content = read_file_content(file_path)
            if content:
                doc_id = generate_doc_id(content)
                metadata = extract_metadata(file_path, content)
                links, _ = extract_links_and_references(content, docs)
                metadata["links"] = [link.split("/")[-1] for link in links]  # Extract the file name from the URL
                docs[filename] = doc_id
                create_document_node(doc_id, content, metadata)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Parse links and references
    for filename, doc_id in docs.items():
        file_path = os.path.join(directory, filename)
        content = read_file_content(file_path)
        if content:
            _, relationships = extract_links_and_references(content, docs)
            for target_doc_id, rel_type in relationships:
                create_relationship(doc_id, target_doc_id, rel_type, extra_data={"source": filename})

# Main function to execute the document processing
if __name__ == "__main__":
    directory = "/home/marek/rag-documents/"
    if os.path.isdir(directory):
        with Halo(text="Starting document processing...", spinner="dots") as spinner:
            try:
                process_documents(directory)
                spinner.succeed("Document processing completed successfully.")
            except Exception as e:
                spinner.fail(f"An error occurred during processing: {e}")
    else:
        print(f"Invalid directory: {directory}")

# Close Neo4j driver when finished
driver.close()
