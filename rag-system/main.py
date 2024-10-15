import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from tqdm import tqdm  # Progress bar library


# Function to rename long file names
def rename_long_filename(file_path, max_length=255):
    # Check if the file path length exceeds the allowed limit
    if len(file_path) > max_length:
        # Extract directory and filename
        directory, filename = os.path.split(file_path)

        # Create a hash for the original filename to ensure uniqueness
        hashed_name = hashlib.sha1(filename.encode()).hexdigest()[:10]  # Take the first 10 characters of the hash

        # Create a new short filename using the hash and file extension
        ext = os.path.splitext(filename)[1]
        new_filename = f"{hashed_name}{ext}"

        # Full path with the new filename
        new_file_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"Renamed '{file_path}' to '{new_file_path}' due to long filename.")

        return new_file_path
    return file_path


# Define function to load PDF files
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# Define function to load Markdown files
def load_md(file_path):
    loader = TextLoader(file_path)
    return loader.load()


# Load documents from a directory
def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Rename the file if the name is too long
        file_path = rename_long_filename(file_path)

        # Handle different file types
        try:
            if filename.endswith(".pdf"):
                docs.extend(load_pdf(file_path))
            elif filename.endswith(".md"):
                docs.extend(load_md(file_path))
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    return docs


# Example usage
directory = "/home/marek/rag-documents/"
documents = load_documents(directory)

from langchain_community.embeddings import OllamaEmbeddings

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Generate embeddings with a progress bar
document_texts = [doc.page_content for doc in documents]

# Adding progress bar for embeddings generation
print("Generating embeddings for documents...")
embeddings = []
for doc_text in tqdm(document_texts, desc="Embedding Progress", unit="doc"):
    embeddings.append(embedding_model.embed_documents([doc_text]))

# Flatten embeddings into a single list
embeddings = [embedding for sublist in embeddings for embedding in sublist]

from langchain_chroma import Chroma

# Initialize ChromaDB
chroma_db = Chroma(embedding_function=embedding_model)

# Add documents to ChromaDB
chroma_db.add_texts(texts=document_texts, metadatas=[doc.metadata for doc in documents])


# Retrieve relevant documents from ChromaDB
def get_relevant_docs(query):
    query_embedding = embedding_model.embed_query(query)
    return chroma_db.similarity_search_by_vector(query_embedding)


# Generate a response using Ollama
from langchain_community.llms import Ollama

# Initialize Ollama language model for text generation
llm = Ollama(model="llama3.1:8b")

# List to store conversation history
conversation_history = []


# Generate a response to the query using retrieved context
def generate_response(query):
    # Retrieve relevant documents
    relevant_docs = get_relevant_docs(query)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Add the previous conversation to the context
    conversation = "\n".join([f"User: {q}\nAI: {r}" for q, r in conversation_history])

    # Create a prompt with context for LLM
    prompt = f"""
                You are a helpdesk assistant that helps the user based on information from provided documents. 
                If you don't know how to answer based on the documents, don't answer. 

                Instructions:
                - Answer only questions mentioned in documents
                - Use Czech language to answer
                - Refer to documents with information that supports your answer

                Previous Conversation:\n{conversation}\n\nContext: {context}\n\nQuery: {query}\nAnswer:"""

    # Generate response using Ollama
    response = llm.invoke(prompt)

    # Add the new conversation to the history
    conversation_history.append((query, response))

    return response


# Chat functionality in the terminal
def chat():
    print("RAG Chat System. Type 'exit' to stop.")
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
