import os
import hashlib
import json
import psycopg2
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Union
import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.webhook import WebhookClient
import hmac
import threading  # Import threading module

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can restrict it to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# PostgreSQL connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Connect to Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection("document_embeddings")
collection.load()

# Connect to Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Hugging Face model
model_name = "Seznam/retromae-small-cs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize LLM and conversation history
llm = Ollama(model="llama3.1:8b")

# Create locks for thread safety
model_lock = threading.Lock()
llm_lock = threading.Lock()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Slack Bot configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Remove the unused global variable
# conversation_history = []  # Tracks user queries and responses

# Pydantic models for request validation
class UserCredentials(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    query: str
    new_chat: Optional[bool] = True
    chat_id: Optional[str] = None


class RegistrationResponse(BaseModel):
    message: str


class LoginResponse(BaseModel):
    access_token: str
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[str]
    chat_id: Optional[str]


###### UTILITIES ######

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return int(user_id)
    except JWTError:
        raise credentials_exception


# Function to generate embeddings for a list of texts
def generate_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with model_lock:  # Ensure thread-safe access to the model
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)


# Save conversation to PostgreSQL
def save_conversation(user_id, chat_id, query, response):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        sanitized_query = query.encode("utf-8", "replace").decode("utf-8")
        sanitized_response = response.encode("utf-8", "replace").decode("utf-8")
        cursor.execute("""
            INSERT INTO user_conversations (user_id, chat_id, conversation)
            VALUES (%s, %s, %s)
        """, (user_id, chat_id, f"User: {sanitized_query}\nAI: {sanitized_response}"))
        connection.commit()
    except Exception as e:
        print(f"Error saving conversation: {e}")
    finally:
        cursor.close()
        connection.close()


# Fetch relevant documents
def get_relevant_docs(query):
    query_embedding = generate_embeddings([query])[0]

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    search_results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["document_id"]
    )

    relevant_doc_ids = [hit.entity.get("document_id") for hit in search_results[0]]
    documents = []
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Document)
            WHERE d.doc_id IN $doc_ids
            RETURN d.content AS content, d.metadata AS metadata
        """, doc_ids=relevant_doc_ids)
        for record in result:
            metadata = json.loads(record['metadata'])
            documents.append({
                'content': record['content'],
                'metadata': metadata,
                'filename': metadata.get('filename', 'Unknown')
            })
    return documents

# Slack signature verification
async def verify_slack_signature(request: Request):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    if abs(int(timestamp) - int(datetime.now().timestamp())) > 60 * 5:
        return False

    request_body = await request.body()  # Fetch raw body as bytes
    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    computed_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    slack_signature = request.headers.get("X-Slack-Signature")
    return hmac.compare_digest(computed_signature, slack_signature)

async def process_slack_command(user_query: str, channel_id: str):
    try:
        # Process the query using your chat backend
        response = generate_response(QueryRequest(query=user_query, new_chat=True), 1)

        # Send the final response to Slack
        slack_client.chat_postMessage(
            channel=channel_id,
            text=response.response
        )
    except Exception as e:
        print(f"Error processing Slack command: {e}")
        slack_client.chat_postMessage(
            channel=channel_id,
            text="Sorry, something went wrong while processing your request."
        )

###### ENDPOINTS ######

# Fetch all chats for a user

@app.get("/chats", response_model=dict)
def get_user_chats(current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT chat_id, MAX(conversation) AS latest_message
            FROM user_conversations
            WHERE user_id = %s
            GROUP BY chat_id
            ORDER BY MAX(id) DESC
        """, (current_user_id,))
        result = cursor.fetchall()
        chats = [{"chat_id": row[0], "latest_message": row[1]} for row in result]
        return {"chats": chats}
    finally:
        cursor.close()
        connection.close()


# Fetch conversation history for a specific chat
@app.get("/chat/history/{chat_id}", response_model=dict)
def get_chat_history(chat_id: str, current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT conversation FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
        """, (current_user_id, chat_id))
        result = cursor.fetchall()
        history = []
        for record in result:
            # Split by line, e.g., "User: ...\nAI: ..."
            conversations = record[0].split("\n")
            history.extend(conversations)
        return {"history": history}
    finally:
        cursor.close()
        connection.close()


# User registration endpoint
@app.post("/register", response_model=RegistrationResponse)
def register_user(credentials: UserCredentials):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()

    hashed_password = hashlib.sha256(credentials.password.encode()).hexdigest()
    try:
        cursor.execute("""
            INSERT INTO users (username, password)
            VALUES (%s, %s)
        """, (credentials.username, hashed_password))
        connection.commit()
        return {"message": "Registration successful"}
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        cursor.close()
        connection.close()


# User login endpoint
@app.post("/login", response_model=LoginResponse)
def login_for_access_token(credentials: UserCredentials):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, password FROM users WHERE username = %s", (credentials.username,))
        result = cursor.fetchone()
        if result:
            user_id, stored_hash = result
            input_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
            if input_hash == stored_hash:
                access_token = create_access_token(data={"sub": str(user_id)},
                                                   expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
                return {"access_token": access_token, "message": "Login successful"}
        raise HTTPException(status_code=401, detail="Invalid username or password")
    finally:
        cursor.close()
        connection.close()


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def generate_response(request: QueryRequest, current_user_id: int = Depends(get_current_user)):
    # Log the incoming request data
    print(f"Request received: {request.dict()}")

    if request.new_chat:
        # Create a new chat_id for a new conversation
        chat_id = str(uuid.uuid4())
        conversation = ""  # Start fresh for a new chat
    else:
        # Validate chat_id is provided when not starting a new chat
        chat_id = request.chat_id
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required when new_chat is False")

        # Fetch conversation history for the chat
        connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT conversation FROM user_conversations WHERE user_id = %s AND chat_id = %s ORDER BY id ASC
            """, (current_user_id, chat_id))
            result = cursor.fetchall()
            conversation = "\n".join([record[0] for record in result])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")
        finally:
            cursor.close()
            connection.close()

    documents = get_relevant_docs(request.query)
    max_docs = 5
    documents = documents[:max_docs]

    context = ""
    for doc in documents:
        content = doc['content']
        filename = doc['filename']
        content = content.encode('utf-8', errors='replace').decode('utf-8')
        filename = filename.encode('utf-8', errors='replace').decode('utf-8')
        context += f"{content}\n(Source: {filename})\n\n"

    prompt = f"""
        You are a helpdesk assistant who assists users based on information from the provided documents.

        Your primary goal is to help users solve their problems by providing simple, clear, and step-by-step instructions suitable for non-technical individuals.

        Previous conversation:
        {conversation}

        Context:
        {context}

        Question:
        {request.query}

        Your kind and helpful Answer:
    """

    with llm_lock:  # Ensure thread-safe access to the LLM
        response = llm.invoke(prompt).strip()
    response = response.encode('utf-8', errors='replace').decode('utf-8')

    save_conversation(current_user_id, chat_id, request.query, response)

    source_names = ", ".join([doc['filename'] for doc in documents])
    source_names = source_names.encode('utf-8', errors='replace').decode('utf-8')
    formatted_response = f"{response}\n\nSources:\n{source_names.replace(',', '\n')}"

    return {"response": formatted_response, "sources": source_names, "chat_id": chat_id}


@app.post("/slack/events")
async def slack_events(request: Request):
    if not await verify_slack_signature(request):  # Use await for the async function
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    body = await request.json()
    event = body.get("event", {})
    if event.get("type") == "message" and not event.get("bot_id"):
        user_query = event.get("text")
        channel_id = event.get("channel")

        # Call your chat endpoint
        try:
            # Here you should invoke your own chat logic, not Slack API
            response = generate_response(QueryRequest(query=user_query, new_chat=True), 1)

            # Send response back to Slack
            slack_client.chat_postMessage(
                channel=channel_id,
                text=response.response
            )
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    return JSONResponse(content={"message": "Event received"})


@app.post("/slack/command")
async def slack_command(request: Request, background_tasks: BackgroundTasks):
    if not await verify_slack_signature(request):
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    form_data = await request.form()
    user_query = form_data.get("text")
    channel_id = form_data.get("channel_id")

    # Send an immediate response to Slack
    background_tasks.add_task(process_slack_command, user_query, channel_id)
    return JSONResponse(content={"response_type": "ephemeral", "text": "Processing your request..."})
