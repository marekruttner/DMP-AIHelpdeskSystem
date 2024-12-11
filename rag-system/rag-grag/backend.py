import os
import hashlib
import json
import re

import torch
import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, Index, IndexType, utility
from neo4j import GraphDatabase
from halo import Halo
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader

# Global variables
collection = None
driver = None
tokenizer = None
model = None

def init_milvus_collection(host="localhost", port="19530", collection_name="document_embeddings"):
    """
    Initialize the Milvus connection, create or recreate the document_embeddings collection and index.
    """
    global collection
    # Connect to Milvus
    connections.connect("default", host=host, port=port)

    # Define Milvus collection schema with auto_id=False
    fields = [
        FieldSchema(name="document_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
    ]
    schema = CollectionSchema(fields, description="Document Embeddings")

    # Clear existing collection if it exists and recreate it
    if collection_name in utility.list_collections():
        with Halo(text=f"Collection '{collection_name}' exists. Dropping it to clear the database...", spinner="dots") as spinner:
            existing_collection = Collection(name=collection_name)
            existing_collection.drop()
            spinner.succeed(f"Collection '{collection_name}' dropped successfully.")

    # Create a new collection
    collection = Collection(name=collection_name, schema=schema)

    # Create an index for the "embedding" field
    index_params = {
        "index_type": IndexType.HNSW,
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }

    with Halo(text="Creating index on the embedding field...", spinner="dots") as spinner:
        Index(collection, "embedding", index_params)
        spinner.succeed("Index created successfully.")

    # Load the collection to be ready for search and insert
    collection.load()

def init_neo4j(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="testtest"):
    """
    Initialize the Neo4j driver.
    """
    global driver
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def init_model(model_name="Seznam/retromae-small-cs"):
    """
    Initialize the Hugging Face transformer model and tokenizer for embedding documents.
    """
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

def clear_neo4j_graph():
    """
    Clear all nodes and relationships from the Neo4j graph.
    """
    with driver.session() as session:
        with Halo(text="Clearing Neo4j graph...", spinner="dots") as spinner:
            session.run("MATCH (n) DETACH DELETE n")
            spinner.succeed("Neo4j graph cleared successfully.")

def generate_doc_id(content):
    """
    Generate a consistent integer doc_id based on the content hash.
    """
    sha256_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    doc_id = int(sha256_hash, 16) % (2 ** 63 - 1)
    return doc_id

def store_embeddings(doc_ids, embeddings):
    """
    Store embeddings in Milvus.
    """
    embeddings = np.array(embeddings)  # Ensure embeddings is a NumPy array
    collection.insert([doc_ids, embeddings.tolist()])
    collection.flush()

def create_document_node(doc_id, content, metadata):
    """
    Create a document node in Neo4j.
    """
    metadata_json = json.dumps(metadata)
    with driver.session() as session:
        session.run("""
        CREATE (d:Document {
            doc_id: $doc_id,
            content: $content,
            metadata: $metadata_json,
            is_global: true,
            workspace_id: null
        })
        """, doc_id=doc_id, content=content, metadata_json=metadata_json)

def create_relationship(doc_id_1, doc_id_2, relationship_type, extra_data=None):
    """
    Create a relationship between two documents in Neo4j.
    """
    if extra_data and "score" in extra_data:
        extra_data["score"] = float(extra_data["score"])  # Ensure float32 is converted to float
    with driver.session() as session:
        result = session.run("""
        MATCH (d1:Document {doc_id: $doc_id_1}), (d2:Document {doc_id: $doc_id_2})
        CREATE (d1)-[:RELATED {type: $relationship_type, extra: $extra_data}]->(d2)
        RETURN d1, d2
        """, doc_id_1=doc_id_1, doc_id_2=doc_id_2, relationship_type=relationship_type, extra_data=json.dumps(extra_data or {}))
        if result.peek() is None:
            print(f"Failed to create relationship: {doc_id_1} -> {doc_id_2}")

def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def compute_similarity_and_store(doc_id, embedding, doc_ids, embeddings):
    """
    Compute similarity between the given embedding and existing embeddings, store relationships if above threshold.
    """
    for i, existing_embedding in enumerate(embeddings):
        existing_doc_id = doc_ids[i]
        similarity_score = cosine_similarity(embedding, existing_embedding)
        if similarity_score > 0.7:  # Similarity threshold
            create_relationship(doc_id, existing_doc_id, "SIMILAR_TO", extra_data={"score": similarity_score})

def extract_links_and_references(content, doc_map):
    """
    Extract links from markdown-like syntax [text](link) and find referenced documents.
    """
    links = re.findall(r"\[.*?\]\((.*?)\)", content)
    relationships = []
    for link in links:
        for ref_doc, ref_doc_id in doc_map.items():
            if link in ref_doc:
                relationships.append((ref_doc_id, "LINK"))
    return links, relationships

def read_file_content(file_path):
    """
    Read the content of a file, supported are .md and .pdf, others are considered text.
    """
    if file_path.endswith(".md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
    else:
        # For other files, assume text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None

def extract_metadata(file_path, content):
    """
    Extract metadata from the file.
    """
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
    else:
        metadata["type"] = "Text"
    return metadata

def process_documents(directory):
    """
    Process and embed documents from a directory, store them in Milvus and Neo4j, and compute relationships.
    """
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
                metadata["links"] = [link.split("/")[-1] for link in links]
                docs[filename] = doc_id
                create_document_node(doc_id, content, metadata)

                # Generate embeddings
                with Halo(text=f"Generating embeddings for {filename}...", spinner="dots") as spinner:
                    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(embedding)
                    doc_ids.append(doc_id)
                    spinner.succeed(f"Embedding generated for {filename}.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Compute similarities and store relationships
    for i, embedding in enumerate(embeddings):
        compute_similarity_and_store(doc_ids[i], embedding, doc_ids[:i], embeddings[:i])

    # Store embeddings in Milvus
    if embeddings:
        with Halo(text=f"Storing {len(embeddings)} documents in Milvus...", spinner="dots") as spinner:
            store_embeddings(doc_ids, embeddings)
            spinner.succeed(f"Stored {len(embeddings)} documents successfully in Milvus.")
    else:
        print("No embeddings to store.")

    # Store links explicitly found in the documents
    with Halo(text="Storing explicit document links in Neo4j...", spinner="dots") as spinner:
        for filename, doc_id in docs.items():
            links, relationships = extract_links_and_references(read_file_content(os.path.join(directory, filename)), docs)
            for target_doc_id, _ in relationships:
                create_relationship(doc_id, target_doc_id, "LINK", extra_data={"source": filename})
        spinner.succeed("Explicit document links stored successfully.")

def initialize_all(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="testtest", milvus_host="localhost", milvus_port="19530"):
    """
    Initialize all connections and models required.
    """
    init_milvus_collection(host=milvus_host, port=milvus_port, collection_name="document_embeddings")
    init_neo4j(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
    init_model()

# Optional: enable standalone testing
if __name__ == "__main__":
    directory = "/path/to/your/documents"
    if os.path.isdir(directory):
        initialize_all()
        with Halo(text="Starting document processing...", spinner="dots") as spinner:
            try:
                process_documents(directory)
                spinner.succeed("Document processing completed successfully.")
            except Exception as e:
                spinner.fail(f"An error occurred during processing: {e}")
    else:
        print(f"Invalid directory: {directory}")
