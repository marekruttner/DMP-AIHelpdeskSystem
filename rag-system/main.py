from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from tqdm import tqdm  # Progress bar library

# Define function to load PDF files
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Define function to load Markdown files
def load_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    loader = TextLoader(text)
    return loader.load()

# Load documents from a directory
def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            docs.extend(load_pdf(file_path))
        elif filename.endswith(".md"):
            docs.extend(load_md(file_path))
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

# Finalize embeddings into a single list (optional, depends on your logic)
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
    prompt = f"Previous Conversation:\n{conversation}\n\nContext: {context}\n\nQuery: {query}\nAnswer:"

    # Generate response using Ollama
    response = llm(prompt)

    # Add the new conversation to the history
    conversation_history.append((query, response))

    return response


# Chat functionality in the terminal
def chat():
    print("RAG Chat System. Type 'exit' to stop.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Exiting chat...")
            break
        response = generate_response(query)
        print(f"AI: {response}")

# Start chat
if __name__ == "__main__":
    chat()
