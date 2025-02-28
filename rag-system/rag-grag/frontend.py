import os
import hashlib
import json
import psycopg2
import numpy as np
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException, Depends, UploadFile, Form, Request, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Union
import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import hmac
import threading
from dotenv import load_dotenv
import re
import logging

# ------------------------------------------------------------------------
# IMPORTANT: import your factory method from the storage_integration script
# ------------------------------------------------------------------------
from storage_integrations import integrate_data_into_datalake, get_datalake

# Import the improved backend module (with multi-vector embeddings, etc.)
import backend

# For summarizing long conversations (optional huggingface approach)
try:
    from transformers import pipeline

    conversation_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except:
    conversation_summarizer = None

# FastAPI app initialization
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# PostgreSQL connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Slack Bot configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Locks for concurrency
model_lock = threading.Lock()
llm_lock = threading.Lock()

# Initialize backend (Milvus, Neo4j, Models)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testtest")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

backend.initialize_all(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    milvus_host=MILVUS_HOST,
    milvus_port=MILVUS_PORT
)

# Initialize LLM (you can choose any model in Ollama)
llm = Ollama(model="phi4:latest")

################################################################################
# Configurable Constants
################################################################################

MAX_CONVERSATION_CHARS = 3000
MAX_CONTEXT_CHARS = 3000

# If conversation grows beyond this length, we do an automatic summary
CONVERSATION_SUMMARY_TRIGGER = 4000

################################################################################
# Pydantic Models
################################################################################

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

class UpdateRoleRequest(BaseModel):
    username: str
    new_role: str

class CreateWorkspaceRequest(BaseModel):
    name: str

class AssignUserRequest(BaseModel):
    user_id: int

class StorageConfig(BaseModel):
    datalake_type: str
    config: dict

# NEW MODELS BELOW
class PublicQueryRequest(BaseModel):
    query: str
    new_chat: Optional[bool] = True
    chat_id: Optional[str] = None
    user_id: Optional[Union[str, int]] = None  # can be string or int if desired

class RatingRequest(BaseModel):
    chat_id: str
    rating: int  # e.g. 1..5
    comment: Optional[str] = None

################################################################################
# Auth / Helpers
################################################################################

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

async def get_current_user_with_role(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception

        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT id, role, workspace_id FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if not result:
            raise credentials_exception
        return {"user_id": result[0], "role": result[1], "workspace_id": result[2]}
    except JWTError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def save_conversation(user_id, chat_id, query, response):
    """
    Saves user+assistant messages in the DB.
    We store them in user_conversations as raw text.
    Potentially used later for summarization if it becomes too large.
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        sanitized_query = query.encode("utf-8", "replace").decode("utf-8")
        sanitized_response = response.encode("utf-8", "replace").decode("utf-8")
        cursor.execute(
            """
            INSERT INTO user_conversations (user_id, chat_id, conversation)
            VALUES (%s, %s, %s)
            """,
            (user_id, chat_id, f"User: {sanitized_query}\nAI: {sanitized_response}")
        )
        connection.commit()
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail="Error saving conversation.")
    finally:
        cursor.close()
        connection.close()

def role_required(required_roles: list):
    def decorator(current_user=Depends(get_current_user_with_role)):
        if current_user["role"] not in required_roles:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        return current_user
    return decorator

################################################################################
# Query Refinement & Summaries
################################################################################

def refine_query(original_query: str, conversation_context: str) -> str:
    """
    Use the LLM to produce a short, direct refined query in the same language as the user.
    """
    prompt_for_refinement = f"""
        You are a query refiner. Given the conversation so far and the user's latest query,
        rewrite the latest query into a short, direct query that will best match relevant documents
        in the knowledge base. If the conversation is in Czech, keep it in Czech; if in English,
        keep it in English. Avoid any extra explanation or commentary. 
        Simply return the refined query that has all important information.

        Conversation so far:
        {conversation_context}

        User's latest query: {original_query}

        Refined query (no additional text, just the query):
        """.strip()

    with llm_lock:
        refined_query = llm.invoke(prompt_for_refinement).strip()
    return refined_query

def summarize_conversation(conversation_text: str) -> str:
    """
    Summarizes the conversation if conversation_summarizer is available;
    otherwise, do a naive approach.
    """
    if conversation_summarizer:
        try:
            result = conversation_summarizer(conversation_text, max_length=100, min_length=50, do_sample=False)
            summary = result[0]["summary_text"]
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarization error: {e}")

    # fallback naive approach
    truncated = conversation_text[:500] + "..."
    return f"Summary of conversation: {truncated}"

def maybe_summarize_long_conversation(user_id: int, chat_id: str):
    """
    If the conversation is too long, we automatically summarize it
    and store that summary as a new conversation entry (like a running memory).
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation 
            FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (user_id, chat_id)
        )
        rows = cursor.fetchall()
        full_conv = "\n".join([r[0] for r in rows])
        if len(full_conv) > CONVERSATION_SUMMARY_TRIGGER:
            # Summarize
            summary = summarize_conversation(full_conv)
            # Insert the summary as a new "turn"
            cursor.execute(
                """
                INSERT INTO user_conversations (user_id, chat_id, conversation)
                VALUES (%s, %s, %s)
                """,
                (user_id, chat_id, f"AI (conversation summary): {summary}")
            )
            connection.commit()
            logger.info(f"Conversation for chat_id {chat_id} summarized.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error in maybe_summarize_long_conversation: {e}")
        raise HTTPException(status_code=500, detail="Error summarizing conversation.")
    finally:
        cursor.close()
        connection.close()

################################################################################
# Multi-Vector + Graph Retrieval
################################################################################

def hybrid_search(query: str, top_k: int = 5) -> list:
    """
    Demonstrates a multi-vector retrieval (semantic + lexical) approach.
    1) Embed the query using both models
    2) Search in both Milvus collections
    3) Merge results (by average or max of similarity)
    4) Return top_k doc_ids
    """
    # Step 1: embed query in both models
    with model_lock:
        sem_emb = backend.semantic_embedding_model.encode([query], show_progress_bar=False)[0].astype(np.float32)
        lex_emb = backend.lexical_embedding_model.encode([query], show_progress_bar=False)[0].astype(np.float32)

    # Step 2: do Milvus searches
    sem_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    lex_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    sem_results = backend.semantic_collection.search(
        data=[sem_emb.tolist()],
        anns_field="embedding",
        param=sem_search_params,
        limit=top_k * 2,
        output_fields=["document_id"]
    )[0]

    lex_results = backend.lexical_collection.search(
        data=[lex_emb.tolist()],
        anns_field="embedding",
        param=lex_search_params,
        limit=top_k * 2,
        output_fields=["document_id"]
    )[0]

    # Step 3: unify and rank
    score_map = {}

    for hit in sem_results:
        doc_id = hit.entity.get("document_id")
        score = hit.score
        if doc_id not in score_map:
            score_map[doc_id] = []
        score_map[doc_id].append(score)

    for hit in lex_results:
        doc_id = hit.entity.get("document_id")
        score = hit.score
        if doc_id not in score_map:
            score_map[doc_id] = []
        score_map[doc_id].append(score)

    doc_id_scores = []

    for doc_id, scores in score_map.items():
        avg_score = sum(scores) / len(scores)
        doc_id_scores.append((doc_id, avg_score))

    doc_id_scores.sort(key=lambda x: x[1], reverse=True)
    top_doc_ids = [t[0] for t in doc_id_scores[:top_k]]
    logger.info(f"Hybrid search top doc_ids: {top_doc_ids}")
    return top_doc_ids

def clean_cypher_query(query: str) -> str:
    """
    Cleans the Cypher query by removing code fencing and the 'cypher' keyword if present.
    """
    # Remove Markdown code fences
    query = re.sub(r'^```.*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'\n```$', '', query, flags=re.MULTILINE)

    # Remove 'cypher' keyword if it exists at the start
    query = query.strip()
    if query.lower().startswith('cypher'):
        query = query[len('cypher'):].strip()

    return query

def generate_cypher_query(refined_query: str) -> str:
    """
    Use LLM to generate a possible Cypher query to find relevant docs in the graph.
    """
    prompt = f"""
    You are a Cypher query generator. The user asked the following refined query:
    '{refined_query}'

    We have a Neo4j graph with :Document, :Topic, :Entity nodes and relationships:
    - (Document)-[:HAS_TOPIC]->(Topic)
    - (Document)-[:MENTIONS]->(Entity)
    - (Document)-[:RELATED {{type: 'SIMILAR_TO'}}]->(Document)

    Generate an efficient Cypher query to find the top relevant Document nodes based on the user's query.
    Ensure that the query does not create Cartesian products by maintaining connected patterns.
    Only output the Cypher query without any additional text, code fencing, or comments.
    """.strip()

    with llm_lock:
        possible_cypher = llm.invoke(prompt).strip()

    # Clean the response to remove any unintended formatting
    possible_cypher = clean_cypher_query(possible_cypher)
    logger.info(f"Generated Cypher query: {possible_cypher}")
    return possible_cypher

def validate_cypher_query(query: str):
    """
    Validates that the Cypher query starts with a valid Cypher keyword.
    """
    valid_start_keywords = [
        "MATCH", "CREATE", "DELETE", "MERGE", "RETURN", "WITH",
        "OPTIONAL MATCH", "CALL", "UNWIND", "LOAD CSV", "FOREACH",
        "DETACH DELETE"
    ]
    query_upper = query.upper()
    if not any(query_upper.startswith(keyword) for keyword in valid_start_keywords):
        raise ValueError("Invalid Cypher query syntax.")

def run_cypher_query(query_text: str, top_k: int = 5) -> list:
    """
    Attempts to run a given Cypher query, expecting it to return a list of doc_ids
    from Document nodes. We'll parse them out.
    """
    doc_ids = []
    try:
        validate_cypher_query(query_text)
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return doc_ids

    with backend.driver.session() as session:
        try:
            logger.info(f"Executing Cypher query: {query_text}")
            result = session.run(query_text)
            for rec in result:
                if "doc_id" in rec.keys():
                    doc_ids.append(rec["doc_id"])
                else:
                    val = next(iter(rec.values()), None)
                    if isinstance(val, int):
                        doc_ids.append(val)
            logger.info(f"Retrieved doc_ids from Cypher query: {doc_ids}")
        except Exception as e:
            logger.error(f"Cypher query failed or invalid: {e}")
    doc_ids = list(set(doc_ids))
    logger.info(f"Final doc_ids after deduplication: {doc_ids[:top_k]}")
    return doc_ids[:top_k]

def retrieve_docs_from_neo4j(doc_ids: list, public_only: bool = True) -> list:
    """
    Given a list of doc_ids, fetch their content and metadata from Neo4j,
    then keep only docs with metadata["is_public"] == True if public_only is True.
    """
    if not doc_ids:
        return []

    with backend.driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            WHERE d.doc_id IN $doc_ids
            RETURN d.content AS content, d.metadata AS metadata
            """,
            doc_ids=doc_ids
        )
        documents = []
        for record in result:
            meta_raw = record["metadata"]
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except:
                meta = {}
            # Only append if is_public == True or user has access to private documents
            if not public_only or meta.get("is_public") is True:
                logger.info(f"Document '{meta.get('filename', 'Unknown')}' is {'public' if meta.get('is_public') else 'private'}.")
                documents.append({
                    "content": record["content"],
                    "filename": meta.get("filename", "Unknown"),
                    "is_public": meta.get("is_public", False)
                })
    logger.info(f"Total documents retrieved: {len(documents)}")
    return documents

def get_hybrid_plus_cypher_docs(refined_query: str, top_k: int = 5, use_cypher_expansion: bool = True, public_only: bool = True) -> list:
    """
    1. Multi-vector search in Milvus
    2. (Optional) Generate a Cypher query to find relevant docs in Neo4j
    3. Merge doc_ids, retrieve content from Neo4j (based on public_only)
    """
    doc_ids_hybrid = hybrid_search(refined_query, top_k=top_k)
    logger.info(f"Hybrid search returned doc_ids: {doc_ids_hybrid}")

    doc_ids_cypher = []
    if use_cypher_expansion:
        possible_cypher = generate_cypher_query(refined_query)
        doc_ids_cypher = run_cypher_query(possible_cypher, top_k=top_k)
        logger.info(f"Cypher expansion returned doc_ids: {doc_ids_cypher}")

    all_ids = list(set(doc_ids_hybrid + doc_ids_cypher))
    logger.info(f"Merged doc_ids: {all_ids}")

    docs = retrieve_docs_from_neo4j(all_ids, public_only=public_only)

    if not docs and use_cypher_expansion:
        logger.info("No documents found with Cypher expansion. Falling back to hybrid search only.")
        docs = retrieve_docs_from_neo4j(doc_ids_hybrid, public_only=public_only)

    logger.info(f"Total documents retrieved: {len(docs)}")
    return docs

################################################################################
# Slack Helpers
################################################################################

async def verify_slack_signature(request: Request):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    if not timestamp:
        logger.warning("Missing Slack timestamp header.")
        return False
    try:
        timestamp = int(timestamp)
    except ValueError:
        logger.warning("Invalid Slack timestamp header.")
        return False

    # Check if the request is too old
    if abs(int(datetime.now().timestamp()) - timestamp) > 60 * 5:
        logger.warning("Slack request timestamp is too old.")
        return False

    request_body = await request.body()
    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    computed_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    slack_signature = request.headers.get("X-Slack-Signature")
    if not slack_signature:
        logger.warning("Missing Slack signature header.")
        return False

    return hmac.compare_digest(computed_signature, slack_signature)

async def process_slack_command(user_query: str, channel_id: str):
    try:
        response = generate_response(
            QueryRequest(query=user_query, new_chat=True),
            current_user={"user_id": 1, "role": "superadmin", "workspace_id": None}
        )
        slack_client.chat_postMessage(channel=channel_id, text=response.response)
    except Exception as e:
        logger.error(f"Error processing Slack command: {e}")
        slack_client.chat_postMessage(
            channel=channel_id,
            text="Sorry, something went wrong while processing your request."
        )

def get_storage_settings():
    config_path = "storage_config.json"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=500, detail="Storage configuration not found.")
    with open(config_path, "r") as f:
        return json.load(f)

################################################################################
# Existing Endpoints (Unchanged)
################################################################################

@app.get("/chats", response_model=dict)
def get_user_chats(current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT chat_id, MAX(conversation) AS latest_message
            FROM user_conversations
            WHERE user_id = %s
            GROUP BY chat_id
            ORDER BY MAX(id) DESC
            """,
            (current_user_id,)
        )
        result = cursor.fetchall()
        chats = [{"chat_id": row[0], "latest_message": row[1]} for row in result]
        return {"chats": chats}
    finally:
        cursor.close()
        connection.close()

@app.get("/chat/history/{chat_id}", response_model=dict)
def get_chat_history(chat_id: str, current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (current_user_id, chat_id)
        )
        result = cursor.fetchall()
        history = []
        for record in result:
            conversations = record[0].split("\n")
            history.extend(conversations)
        return {"history": history}
    finally:
        cursor.close()
        connection.close()

@app.post("/register")
def register_user(username: str = Form(...), password: str = Form(...)):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        connection.commit()
        logger.info(f"User '{username}' registered successfully.")
        return {"message": "Registration successful"}
    except psycopg2.errors.UniqueViolation:
        logger.warning(f"Registration failed: Username '{username}' already exists.")
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        cursor.close()
        connection.close()

@app.post("/login")
def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, role FROM users WHERE username = %s AND password = %s", (username, hashed_password))
        result = cursor.fetchone()
        if result:
            user_id, user_role = result
            access_token = create_access_token(data={"sub": str(user_id)},
                                               expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
            logger.info(f"User '{username}' logged in successfully.")
            return {"access_token": access_token, "message": "Login successful", "role": user_role}
        logger.warning(f"Login failed for username '{username}'.")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    finally:
        cursor.close()
        connection.close()

@app.post("/workspaces")
def create_workspace(
        request: CreateWorkspaceRequest,
        current_user=Depends(role_required(["admin", "superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO workspaces (name) VALUES (%s) RETURNING id", (request.name,))
        workspace_id = cursor.fetchone()[0]
        connection.commit()
        logger.info(f"Workspace '{request.name}' created with ID {workspace_id}.")
        return {"message": "Workspace created", "workspace_id": workspace_id}
    finally:
        cursor.close()
        connection.close()

@app.post("/workspaces/{workspace_id}/assign-user")
def assign_user_to_workspace(
        workspace_id: int,
        request: AssignUserRequest,
        current_user=Depends(role_required(["admin", "superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO user_workspaces (user_id, workspace_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (request.user_id, workspace_id)
        )
        connection.commit()
        logger.info(f"User {request.user_id} assigned to workspace {workspace_id}.")
        return {"message": f"User {request.user_id} assigned to workspace {workspace_id}"}
    finally:
        cursor.close()
        connection.close()

@app.post("/documents")
def upload_document(
        file: UploadFile = File(...),
        scope: str = Form(...),
        chat_id: Optional[str] = Form(None),
        current_user=Depends(role_required(["user", "admin", "superadmin"]))
):
    if scope not in ["chat", "profile", "workspace", "system"]:
        logger.warning(f"Invalid scope '{scope}' provided.")
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "chat" and not chat_id:
        logger.warning("chat_id is required for chat scope.")
        raise HTTPException(status_code=400, detail="chat_id is required for chat scope")

    if scope == "workspace" and current_user["role"] not in ["admin", "superadmin"]:
        logger.warning("Permission denied for workspace scope.")
        raise HTTPException(status_code=403, detail="Admins and Superadmins only")
    if scope == "system" and current_user["role"] != "superadmin":
        logger.warning("Permission denied for system scope.")
        raise HTTPException(status_code=403, detail="Superadmins only")

    try:
        file_content = file.file.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    doc_id = hashlib.sha256(file_content.encode()).hexdigest()
    metadata = {"filename": file.filename, "scope": scope}

    # Store doc in Neo4j (omitting workspace logic for brevity)
    with backend.driver.session() as session:
        try:
            session.run(
                """
                CREATE (d:Document {
                    doc_id: $doc_id,
                    content: $content,
                    metadata: $metadata,
                    is_public: false,
                    workspace_id: null
                })
                """,
                doc_id=doc_id,
                content=file_content,
                metadata=json.dumps(metadata)
            )
            logger.info(f"Document '{file.filename}' uploaded successfully with doc_id {doc_id}.")
        except Exception as e:
            logger.error(f"Error uploading document to Neo4j: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload document.")

    # Ensure that :Topic nodes have a 'title' property
    ensure_topic_title_exists()

    return {"message": f"Document uploaded successfully with scope {scope}"}

################################################################################
# NEW: Rating endpoint (no auth required)
################################################################################

@app.post("/rate_response")
def rate_response(rating: RatingRequest):
    """
    Allows public or logged-in users to rate a particular chat_id response.
    Insert a row in user_conversation_ratings or a similar table.
    """
    # Basic validation
    if rating.rating < 1 or rating.rating > 5:
        logger.warning(f"Invalid rating '{rating.rating}' received for chat_id {rating.chat_id}.")
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO user_conversation_ratings (chat_id, rating, comment)
            VALUES (%s, %s, %s)
            """,
            (rating.chat_id, rating.rating, rating.comment)
        )
        connection.commit()
        logger.info(f"Rating {rating.rating} for chat_id {rating.chat_id} saved successfully.")
        return {"message": "Rating saved successfully."}
    except Exception as e:
        logger.error(f"Error saving rating for chat_id {rating.chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()

################################################################################
# Slack Events and Commands
################################################################################

@app.post("/slack/events")
async def slack_events(request: Request):
    if not await verify_slack_signature(request):
        logger.warning("Invalid Slack signature.")
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    body = await request.json()
    event = body.get("event", {})
    if event.get("type") == "message" and not event.get("bot_id"):
        user_query = event.get("text")
        channel_id = event.get("channel")
        try:
            response = generate_response(
                QueryRequest(query=user_query, new_chat=True),
                current_user={"user_id": 1, "role": "superadmin", "workspace_id": None}
            )
            slack_client.chat_postMessage(channel=channel_id, text=response.response)
            logger.info(f"Sent response to Slack channel {channel_id}.")
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")

    return JSONResponse(content={"message": "Event received"})

@app.post("/slack/command")
async def slack_command(request: Request, background_tasks: BackgroundTasks):
    if not await verify_slack_signature(request):
        logger.warning("Invalid Slack signature for command.")
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    form_data = await request.form()
    user_query = form_data.get("text")
    channel_id = form_data.get("channel_id")
    background_tasks.add_task(process_slack_command, user_query, channel_id)
    logger.info(f"Processing Slack command from channel {channel_id}.")
    return JSONResponse(content={"response_type": "ephemeral", "text": "Processing your request..."})

################################################################################
# Admin and Storage Endpoints
################################################################################

@app.post("/update-role")
def update_role(
        request: UpdateRoleRequest,
        current_user=Depends(role_required(["admin", "superadmin"]))
):
    if request.new_role not in ["user", "admin", "superadmin"]:
        logger.warning(f"Invalid role '{request.new_role}' attempted to be set.")
        raise HTTPException(status_code=400, detail="Invalid role")

    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET role = %s WHERE username = %s RETURNING id",
            (request.new_role, request.username)
        )
        if cursor.rowcount == 0:
            logger.warning(f"Update role failed: User '{request.username}' not found.")
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        logger.info(f"Role updated for user '{request.username}' to '{request.new_role}'.")
        return {"message": f"Role updated for user {request.username} to {request.new_role}"}
    finally:
        cursor.close()
        connection.close()

@app.get("/admin/users")
def get_all_users(current_user=Depends(role_required(["superadmin"]))):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, username, role FROM users ORDER BY id ASC")
        result = cursor.fetchall()
        users = [{"id": row[0], "username": row[1], "role": row[2]} for row in result]
        logger.info("Retrieved all users for admin.")
        return {"users": users}
    finally:
        cursor.close()
        connection.close()

@app.post("/admin/users/{user_id}/change-username")
def change_username(
        user_id: int,
        new_username: str = Form(...),
        current_user=Depends(role_required(["superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET username = %s WHERE id = %s RETURNING id",
            (new_username, user_id)
        )
        if cursor.rowcount == 0:
            logger.warning(f"Change username failed: User ID '{user_id}' not found.")
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        logger.info(f"Username for user ID '{user_id}' changed to '{new_username}'.")
        return {"message": "Username updated successfully"}
    finally:
        cursor.close()
        connection.close()

@app.post("/admin/users/{user_id}/change-password")
def change_password(
        user_id: int,
        new_password: str = Form(...),
        current_user=Depends(role_required(["superadmin"]))
):
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s RETURNING id",
            (hashed_password, user_id)
        )
        if cursor.rowcount == 0:
            logger.warning(f"Change password failed: User ID '{user_id}' not found.")
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        logger.info(f"Password for user ID '{user_id}' updated successfully.")
        return {"message": "Password updated successfully"}
    finally:
        cursor.close()
        connection.close()

@app.get("/admin/users/{user_id}/chats")
def get_user_chats_admin(
        user_id: int,
        current_user=Depends(role_required(["superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT chat_id, MAX(conversation) AS latest_message
            FROM user_conversations
            WHERE user_id = %s
            GROUP BY chat_id
            ORDER BY MAX(id) DESC
            """,
            (user_id,)
        )
        result = cursor.fetchall()
        chats = [{"chat_id": row[0], "latest_message": row[1]} for row in result]
        logger.info(f"Retrieved chats for user ID '{user_id}' by admin.")
        return {"chats": chats}
    finally:
        cursor.close()
        connection.close()

@app.post("/embed-documents")
def embed_documents(
        directory: str = Form(...),
        current_user=Depends(role_required(["admin", "superadmin"]))
):
    if not os.path.isdir(directory):
        logger.warning(f"Invalid directory '{directory}' provided for embedding.")
        raise HTTPException(status_code=400, detail="Invalid directory")

    try:
        backend.process_documents(directory)
        logger.info(f"Documents in directory '{directory}' processed and embedded successfully.")
        return {"message": f"Documents in {directory} processed and embedded successfully."}
    except Exception as e:
        logger.error(f"Error embedding documents from directory '{directory}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure-storage")
def configure_storage(
        storage_config: StorageConfig,
        current_user=Depends(role_required(["superadmin"]))
):
    config_path = "storage_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(storage_config.dict(), f)
        logger.info(f"Storage configured successfully for {storage_config.datalake_type}.")
        return {"message": f"Storage configured successfully for {storage_config.datalake_type}"}
    except Exception as e:
        logger.error(f"Error configuring storage: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure storage.")

@app.post("/local-datalake/upload-file")
def local_datalake_upload_file(
        file: UploadFile = File(...),
        is_public: bool = Form(False),
        workspace_id: Optional[int] = Form(None),
        current_user=Depends(role_required(["admin", "superadmin"]))
):
    """
    Endpoint to upload a file from the user's computer into the local datalake,
    tag it with is_public or workspace_id, and embed it using backend's logic.
    """

    # 1) Read file into bytes
    filename = file.filename
    try:
        file_bytes = file.file.read()
        if not file_bytes:
            logger.warning(f"Empty or unreadable file '{filename}' uploaded.")
            raise HTTPException(status_code=400, detail="File is empty or unreadable.")
    except Exception as e:
        logger.error(f"Error reading file '{filename}': {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # 2) Store in local datalake with metadata (is_public, workspace_id, etc.)
    local_dlake = get_datalake("local")  # specifically want local

    # We'll store uploaded files in "uploaded/" subfolder in local datalake
    local_path = f"uploaded/{filename}"

    # Create metadata
    metadata = {
        "filename": filename,
        "uploaded_by": current_user["user_id"],
        "is_public": is_public,
        "workspace_id": workspace_id,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    try:
        # Save file + metadata
        local_dlake.save_file_with_metadata(file_bytes, local_path, metadata)
        logger.info(f"File '{filename}' uploaded to local datalake at '{local_path}'.")
    except Exception as e:
        logger.error(f"Error saving file '{filename}' to local datalake: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file to datalake.")

    # 3) Embed it by calling the existing logic in backend
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_filepath = os.path.join(tmpdir, filename)
        # Write file bytes to a temporary file
        try:
            with open(temp_filepath, "wb") as f_out:
                f_out.write(file_bytes)
            logger.info(f"Temporary file created at '{temp_filepath}' for embedding.")
        except Exception as e:
            logger.error(f"Error writing temporary file '{temp_filepath}': {e}")
            raise HTTPException(status_code=500, detail="Failed to write temporary file.")

        # Extract text from the file
        content = backend.read_file_content(temp_filepath)
        if not content:
            logger.warning(f"Could not parse text from the file '{filename}'.")
            raise HTTPException(status_code=400, detail="Could not parse text from the file.")

        # Split into chunks
        chunks = backend.chunk_text_with_langchain(content, chunk_size=1000, chunk_overlap=200)

        # Create base metadata for the Document nodes
        file_level_meta = {
            "filename": filename,
            "is_public": is_public,
            "workspace_id": workspace_id,
            "size": len(file_bytes),
            "word_count": len(content.split())
        }

        # We'll gather doc_ids + embeddings for doc-doc similarity
        doc_ids = []
        sem_embeddings = []

        for ctext in chunks:
            try:
                doc_id = backend.process_chunk(ctext, file_level_meta)
                doc_ids.append(doc_id)
                sem_emb = backend.semantic_embedding_model.encode([ctext], show_progress_bar=False)[0]
                sem_embeddings.append(sem_emb)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue

        # Link these new chunks if they're similar
        try:
            backend.compute_batch_similarities(doc_ids, sem_embeddings, threshold=0.7)
            logger.info(f"Embedded and linked chunks for file '{filename}'.")
        except Exception as e:
            logger.error(f"Error computing batch similarities: {e}")
            raise HTTPException(status_code=500, detail="Failed to compute batch similarities.")

    return {
        "message": f"File '{filename}' uploaded and embedded successfully.",
        "is_public": is_public,
        "workspace_id": workspace_id,
        "local_path": local_path
    }

################################################################################
# NEW ENDPOINTS FOR PUBLIC CHAT + RATINGS
################################################################################

@app.post("/admin/copy-google-drive-to-local")
def copy_google_drive_to_local(
        folder_id: str = Form(...),
        is_public: bool = Form(False),
        current_user=Depends(role_required(["superadmin"]))
):
    """
    Copy files from Google Drive into the local datalake folder,
    then set is_public in their metadata.
    We call integrate_data_into_datalake("google_drive", "local", folder_id=...),
    then update .metadata.json files to reflect is_public.
    """
    try:
        integrate_data_into_datalake(
            provider="google_drive",
            datalake_type="local",
            folder_id=folder_id
        )
        logger.info(f"Copied files from Google Drive folder '{folder_id}' to local datalake.")
    except Exception as e:
        logger.error(f"Error copying files from Google Drive to local datalake: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    base_path = os.environ.get("LOCAL_DATALAKE_PATH", "local_datalake")
    google_drive_dir = os.path.join(base_path, "google_drive")

    for root, dirs, files in os.walk(google_drive_dir):
        for filename in files:
            if filename.endswith(".metadata.json"):
                meta_path = os.path.join(root, filename)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata["is_public"] = is_public
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    logger.info(f"Updated metadata file '{meta_path}' with is_public={is_public}.")
                except Exception as ex:
                    logger.error(f"Could not update metadata file {meta_path}: {ex}")

    return {
        "message": "Successfully copied files from Google Drive to local datalake",
        "is_public": is_public
    }

@app.post("/admin/configure-storage-dashboard")
def configure_storage_dashboard(
        datalake_type: str = Form(...),
        current_user=Depends(role_required(["superadmin"]))
):
    """
    Minimal wrapper around /configure-storage for a simpler Admin UI
    that only sets the datalake type and leaves config empty or default.
    """
    dummy_config = {}
    payload = StorageConfig(datalake_type=datalake_type, config=dummy_config)

    config_path = "storage_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(payload.dict(), f)
        logger.info(f"Storage dashboard configured for datalake type '{datalake_type}'.")
        return {"message": f"Storage configured successfully for {datalake_type}"}
    except Exception as e:
        logger.error(f"Error configuring storage dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure storage.")

################################################################################
# Shutdown Event
################################################################################

@app.on_event("shutdown")
def shutdown_event():
    backend.driver.close()
    logger.info("Neo4j driver closed on shutdown.")

################################################################################
# Complete Updated Chat Endpoint
################################################################################

@app.post("/chat", response_model=ChatResponse)
def generate_response(
        request: QueryRequest,
        current_user=Depends(get_current_user_with_role)
):
    """
    Main chat endpoint with multi-vector retrieval + optional graph expansions,
    conversation summarization, and query refinement.
    """
    logger.info(f"Request received: {request.dict()}")

    # 1) Get or create chat_id
    if request.new_chat:
        chat_id = str(uuid.uuid4())
        conversation_so_far = ""
        logger.info(f"Creating new chat with chat_id: {chat_id}")
    else:
        if not request.chat_id:
            logger.error("chat_id is required when new_chat=False")
            raise HTTPException(status_code=400, detail="chat_id is required when new_chat is False")
        chat_id = request.chat_id
        logger.info(f"Using existing chat_id: {chat_id}")

        connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT conversation 
                FROM user_conversations 
                WHERE user_id = %s AND chat_id = %s 
                ORDER BY id ASC
                """,
                (current_user["user_id"], chat_id)
            )
            rows = cursor.fetchall()
            conversation_so_far = "\n".join([r[0] for r in rows])
            logger.info(f"Retrieved conversation history for chat_id {chat_id}.")
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")
        finally:
            cursor.close()
            connection.close()

    # 2) Possibly summarize
    maybe_summarize_long_conversation(current_user["user_id"], chat_id)

    # Reload if summary was added
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation 
            FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (current_user["user_id"], chat_id)
        )
        rows = cursor.fetchall()
        conversation_so_far = "\n".join([r[0] for r in rows])
        logger.info(f"Reloaded conversation history for chat_id {chat_id}.")
    finally:
        cursor.close()
        connection.close()

    # 3) Refine query
    refined_query = refine_query(request.query, conversation_so_far)
    logger.info(f"Refined query: {refined_query}")

    # 4) Retrieve docs (public and private based on user)
    # Set public_only=False to include private documents for logged-in users
    docs = get_hybrid_plus_cypher_docs(refined_query, top_k=3, use_cypher_expansion=True, public_only=False)

    if not docs:
        logger.warning(f"No documents found for the query: {refined_query}")
        raise HTTPException(status_code=404, detail="No documents found for the query.")

    # Build short context
    context_text = ""
    for doc in docs:
        piece = f"{doc['content']}\n(Source: {doc['filename']})\n\n"
        if len(context_text) + len(piece) < MAX_CONTEXT_CHARS:
            context_text += piece
        else:
            context_text += "... [truncated]"
            break

    truncated_conversation = conversation_so_far
    if len(truncated_conversation) > MAX_CONVERSATION_CHARS:
        truncated_conversation = truncated_conversation[:MAX_CONVERSATION_CHARS] + " ... [truncated]"
        logger.info(f"Truncated conversation for chat_id {chat_id}.")

    # 5) Construct prompt
    prompt = f"""
        You are a helpful assistant who provides concise, step-by-step solutions.
        Conversation so far:
        {truncated_conversation}

        Relevant context:
        {context_text}

        Question:
        {request.query}

        Your concise answer:
    """.strip()

    logger.info("Constructed prompt for LLM.")

    # 6) LLM response
    try:
        with llm_lock:
            response_text = llm.invoke(prompt).strip()
        response_text = response_text.encode('utf-8', errors='replace').decode('utf-8')
        logger.info("LLM response generated successfully.")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise HTTPException(status_code=500, detail="Error generating response from LLM.")

    # 7) Save turn
    try:
        save_conversation(current_user["user_id"], chat_id, request.query, response_text)
        logger.info("Conversation turn saved successfully.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error saving conversation: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error saving conversation.")

    # Return doc filenames
    source_names = [doc['filename'] for doc in docs]
    sources_str = "\n".join(source_names)
    final_answer = f"{response_text}\n\nSources:\n{sources_str}"
    logger.info("Final answer prepared with sources.")

    return ChatResponse(
        response=final_answer,
        sources=", ".join(source_names),
        chat_id=chat_id
    )

################################################################################
# Complete Updated Public Chat Endpoint
################################################################################

@app.post("/public_chat", response_model=ChatResponse)
def public_chat(request: PublicQueryRequest):
    """
    Public/Anonymous version of the /chat endpoint.
    - Does not require auth token.
    - If request.user_id is provided, that is used. Otherwise default=0 (anonymous).
    - Only public docs (is_public=True) are used in the context.
    - Chat history is stored in user_conversations with that user_id.
    """
    logger.info(f"Public chat request received: {request.dict()}")

    # 1) Determine user_id
    if request.user_id is not None:
        try:
            user_id = int(request.user_id)
            logger.info(f"Using provided user_id: {user_id}")
        except ValueError:
            user_id = 0
            logger.warning(f"Invalid user_id '{request.user_id}' provided. Defaulting to anonymous (user_id=0).")
    else:
        user_id = 0
        logger.info("No user_id provided. Defaulting to anonymous (user_id=0).")

    # 2) Get or create chat_id and retrieve conversation history
    if request.new_chat:
        chat_id = str(uuid.uuid4())
        conversation_so_far = ""
        logger.info(f"Creating new public chat with chat_id: {chat_id}")
    else:
        if not request.chat_id:
            logger.error("chat_id is required when new_chat=False in public_chat.")
            raise HTTPException(status_code=400, detail="chat_id is required when new_chat=False")
        chat_id = request.chat_id
        logger.info(f"Using existing public chat_id: {chat_id}")

        connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT conversation 
                FROM user_conversations 
                WHERE user_id = %s AND chat_id = %s 
                ORDER BY id ASC
                """,
                (user_id, chat_id)
            )
            rows = cursor.fetchall()
            conversation_so_far = "\n".join([r[0] for r in rows])
            logger.info(f"Retrieved conversation history for public chat_id {chat_id}.")
        except Exception as e:
            logger.error(f"Error fetching chat history for public chat: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")
        finally:
            cursor.close()
            connection.close()

    # 3) Summarize conversation if it's too long
    maybe_summarize_long_conversation(user_id, chat_id)

    # 4) Reload conversation if a summary was added
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation 
            FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (user_id, chat_id)
        )
        rows = cursor.fetchall()
        conversation_so_far = "\n".join([r[0] for r in rows])
        logger.info(f"Reloaded conversation history for public chat_id {chat_id}.")
    finally:
        cursor.close()
        connection.close()

    # 5) Refine the user's query based on the conversation so far
    refined_query = refine_query(request.query, conversation_so_far)
    logger.info(f"Refined public query: {refined_query}")

    # 6) Retrieve public documents using hybrid and Cypher retrieval methods
    # Set public_only=True to include only public documents
    docs = get_hybrid_plus_cypher_docs(
        refined_query,
        top_k=3,
        use_cypher_expansion=True,  # Ensure only public documents are retrieved
        public_only=True
    )

    if not docs:
        logger.warning(f"No public documents found for the public query: {refined_query}")
        raise HTTPException(status_code=404, detail="No public documents found for the query.")

    # 7) Build the context text from public documents
    context_text = ""
    for doc in docs:
        piece = f"{doc['content']}\n(Source: {doc['filename']})\n\n"
        if len(context_text) + len(piece) < MAX_CONTEXT_CHARS:
            context_text += piece
        else:
            context_text += "... [truncated]"
            break

    # 8) Truncate conversation if it's too long
    truncated_conversation = conversation_so_far
    if len(truncated_conversation) > MAX_CONVERSATION_CHARS:
        truncated_conversation = truncated_conversation[:MAX_CONVERSATION_CHARS] + " ... [truncated]"
        logger.info(f"Truncated conversation for public chat_id {chat_id}.")

    # 9) Construct the prompt for the LLM
    prompt = f"""
    You are a helpful assistant. 
    Conversation so far:
    {truncated_conversation}

    Relevant context:
    {context_text}

    User's query:
    {request.query}

    Your concise answer:
    """.strip()

    logger.info("Constructed prompt for public LLM.")

    # 10) Invoke the LLM to generate a response
    try:
        with llm_lock:
            response_text = llm.invoke(prompt).strip()
        response_text = response_text.encode('utf-8', errors='replace').decode('utf-8')
        logger.info("Public LLM response generated successfully.")
    except Exception as e:
        logger.error(f"Error invoking LLM for public chat: {e}")
        raise HTTPException(status_code=500, detail="Error generating response from LLM.")

    # 11) Save the conversation turn
    try:
        save_conversation(user_id, chat_id, request.query, response_text)
        logger.info("Public conversation turn saved successfully.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error saving public conversation: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error saving conversation.")

    # 12) Prepare the final answer with sources
    source_names = [doc['filename'] for doc in docs]
    sources_str = "\n".join(source_names)
    final_answer = f"{response_text}\n\nSources:\n{sources_str}"
    logger.info("Final public answer prepared with sources.")

    return ChatResponse(
        response=final_answer,
        sources=", ".join(source_names),
        chat_id=chat_id
    )

################################################################################
# Function to Ensure :Topic Nodes Have 'title' Property
################################################################################

def ensure_topic_title_exists():
    """
    Ensures that all :Topic nodes have a 'title' property.
    If a :Topic node lacks the 'title', set it to a default value or based on other properties.
    """
    with backend.driver.session() as session:
        try:
            # Find :Topic nodes without 'title' property
            result = session.run(
                """
                MATCH (t:Topic)
                WHERE NOT exists(t.title)
                RETURN t
                """
            )
            topics = result.fetchall()
            for record in topics:
                topic_node = record["t"]
                # Set a default title or derive from existing properties
                # For demonstration, setting a generic title. Modify as needed.
                default_title = "Untitled Topic"
                # Use Neo4j's internal ID for precise matching
                neo4j_id = topic_node.id
                session.run(
                    """
                    MATCH (t:Topic)
                    WHERE ID(t) = $neo4j_id
                    SET t.title = $title
                    """,
                    neo4j_id=neo4j_id,
                    title=default_title
                )
                logger.info(f"Set default title for Topic node with Neo4j ID {neo4j_id}.")
        except Exception as e:
            logger.error(f"Error ensuring 'title' exists on :Topic nodes: {e}")
