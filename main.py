from dotenv import load_dotenv
import os

# Load environment variables FIRST, before any other imports that need them
# Try to load from .env in the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Disable LangSmith tracing if no API key is set (suppresses warning)
if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from youtube_transcript_api.proxies import GenericProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel,HttpUrl
from yt_to_id import get_youtube_video_id
import hashlib
from pathlib import Path
#import gradio as gr
from langdetect import detect
from langchainhub import Client as HubClient
# LangChain 0.2.x agent imports
# Note: These are available in langchain>=0.2.16,<1.0.0
# In LangChain 1.0.x, the API changed significantly
try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    # If imports fail, it might be LangChain 1.0.x installed
    # Pin to langchain<1.0.0 in requirements.txt
    raise ImportError(
        "Failed to import create_react_agent and AgentExecutor from langchain.agents.\n"
        "This code requires langchain>=0.2.16,<1.0.0.\n"
        "Please ensure requirements.txt specifies: langchain>=0.2.16,<1.0.0\n"
        "Then reinstall: pip install -r requirements.txt --force-reinstall"
    )
from langchain_core.tools import tool, BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from requests import Session as RequestsSession
import requests
from requests.cookies import RequestsCookieJar
from http.cookies import SimpleCookie

from fastapi import FastAPI, Path, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
import secrets
import base64
from typing import Optional, Dict, Any, Tuple, List, Callable, Set
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import (
    get_db, User, SessionLocal,
    VideoConversation, ConversationMessage, VideosCatalog  # Schema 2 models
)
from schema2_crud import (
    get_or_create_conversation,
    get_conversation_by_id,
    insert_conversation_turn,
    get_recent_messages_for_context,
    get_or_create_catalog_entry,
    get_conversation_messages,
    serialize_message_for_api,
    verify_conversation_ownership,
    get_user_conversation_ids,
)
from langchain_core.messages import HumanMessage, AIMessage
try:
    import bcrypt
except ImportError:
    print("Warning: bcrypt not installed. Install with: pip install bcrypt")
    bcrypt = None

from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import time
import asyncio

try:
    import jwt
except ImportError:
    print("Warning: PyJWT not installed. Install with: pip install PyJWT")
    jwt = None

app = FastAPI(
    title="FastAPI RAG Application",
    description="YouTube Video Q&A with AI Agent",
    version="1.0.0"
)

# CORS Configuration
# Properly handle CORS for both development and production
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "*").strip()
if CORS_ORIGINS_STR == "*":
    # Development mode - allow all origins
    # Note: Cannot use allow_credentials=True with allow_origins=["*"]
    # This is a browser security restriction
    allow_origins = ["*"]
    allow_credentials = False
    print("[INFO] CORS: Allowing all origins (development mode)")
else:
    # Production mode - specific origins
    allow_origins = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]
    allow_credentials = True
    print(f"[INFO] CORS: Allowing origins: {', '.join(allow_origins)}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers to frontend
)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "change-me":
    raise ValueError(
        "[ERROR] SECRET_KEY must be set to a strong random value!\n"
        "   Generate one with: openssl rand -hex 32\n"
        "   Set it in .env file or deployment environment"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Debug logging configuration
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if DEBUG_LOGGING else logging.INFO)

# User Storage now handled by PostgreSQL database

# Password hashing utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    if bcrypt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="bcrypt not installed. Please install with: pip install bcrypt"
        )
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if bcrypt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="bcrypt not installed. Please install with: pip install bcrypt"
        )
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# JWT token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    if jwt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT not installed. Please install with: pip install PyJWT"
        )
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Verify JWT token and return (username, user_id).
    
    Phase 2.6: Enhanced to return both username and user_id if present.
    Maintains backward compatibility - returns (username, None) for old tokens.
    
    Returns:
        Tuple of (username: Optional[str], user_id: Optional[str])
    """
    if jwt is None:
        return None, None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: Optional[str] = payload.get("user_id")  # Phase 2.6: Extract user_id if present
        
        # Debug logging (Phase 1.2): Log token verification details
        if DEBUG_LOGGING or username is None:
            log_data = {
                "event": "token_verification",
                "username": username,
                "has_user_id": "user_id" in payload,
                "user_id": user_id,
                "token_expired": False,
                "timestamp": datetime.utcnow().isoformat()
            }
            if "exp" in payload:
                exp_time = datetime.utcnow().timestamp()
                log_data["token_expired"] = payload["exp"] < exp_time
                log_data["expires_at"] = datetime.fromtimestamp(payload["exp"]).isoformat()
            logger.debug(f"[AUTH] Token verification: {json.dumps(log_data)}")
        
        if username is None:
            return None, None
        return username, user_id
    except jwt.ExpiredSignatureError:
        if DEBUG_LOGGING:
            logger.debug(f"[AUTH] Token expired: {json.dumps({'event': 'token_expired', 'timestamp': datetime.utcnow().isoformat()})}")
        return None, None
    except jwt.PyJWTError as e:
        if DEBUG_LOGGING:
            logger.debug(f"[AUTH] Token validation error: {json.dumps({'event': 'token_error', 'error_type': type(e).__name__, 'timestamp': datetime.utcnow().isoformat()})}")
        return None, None

# OAuth2 Configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read": "Read access to resources",
        "write": "Write access to resources",
        "admin": "Administrative access"
    }
)

# OAuth2 Authentication Functions
def authenticate_user(username: str, password: str, db: Session):
    """Authenticate user with username and password"""
    # Check if user exists in database
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        return user
    return None

def get_current_user_oauth2(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Get current user from OAuth2 token.
    
    Phase 2.7: Enhanced to use user_id from token if available for faster lookup.
    Validates token user_id matches database user.id for security.
    Falls back to username lookup for backward compatibility.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username, token_user_id = verify_token(token)
    if username is None:
        raise credentials_exception
    
    # Phase 2.7: Use user_id from token if available (faster lookup)
    start_time = time.time()
    if token_user_id:
        # Try lookup by user_id first (faster)
        try:
            import uuid as uuid_lib
            user_id_uuid = uuid_lib.UUID(token_user_id)
            user = db.query(User).filter(User.id == user_id_uuid).first()
            
            # Security check: Validate token user_id matches username
            if user and user.username != username:
                if DEBUG_LOGGING:
                    logger.warning(f"[AUTH] Token user_id mismatch: token_user_id={token_user_id}, token_username={username}, db_username={user.username}")
                # Fallback to username lookup for security
                user = None
        except (ValueError, TypeError):
            # Invalid UUID format in token - fallback to username lookup
            user = None
    else:
        user = None
    
    # Fallback to username lookup (backward compatibility or if user_id lookup failed)
    if not user:
        user = db.query(User).filter(User.username == username).first()
    
    lookup_time = (time.time() - start_time) * 1000  # Convert to ms
    
    if DEBUG_LOGGING or user is None:
        log_data = {
            "event": "user_lookup",
            "username": username,
            "token_user_id": token_user_id,
            "user_found": user is not None,
            "user_id": str(user.id) if user else None,
            "lookup_method": "user_id" if token_user_id and user else "username",
            "lookup_time_ms": round(lookup_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"[AUTH] User lookup: {json.dumps(log_data)}")
    
    if user is None:
        raise credentials_exception
    
    # Phase 2.7: Validate token user_id matches database user.id (security check)
    if token_user_id and str(user.id) != token_user_id:
        if DEBUG_LOGGING:
            logger.warning(f"[AUTH] Token user_id validation failed: token_user_id={token_user_id}, db_user_id={user.id}")
        # Still allow access but log warning (token might be from old format)
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return username

def get_current_user_with_scopes(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user with scopes from OAuth2 token"""
    username = get_current_user_oauth2(token, db)
    user = db.query(User).filter(User.username == username).first()
    return {
        "username": username,
        "scopes": user.scopes if user else []
    }

def require_scope(required_scope: str):
    """Dependency to require specific OAuth2 scope"""
    def scope_checker(current_user: dict = Depends(get_current_user_with_scopes)):
        if required_scope not in current_user["scopes"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker

# Legacy Basic Authentication (for backward compatibility)
def get_current_user_basic(username: str = Depends(oauth2_scheme)):
    """Legacy basic authentication - kept for backward compatibility"""
    # This is a placeholder - in practice, you'd implement basic auth here
    # For now, we'll redirect to OAuth2
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Please use OAuth2 authentication. Use /token endpoint to get access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )



# User Authentication Models
class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

# ============================================================================
# SCHEMA 2 PYDANTIC MODELS
# ============================================================================

class ConversationInput(BaseModel):
    """Input model for Schema 2 conversation requests"""
    video_url: Optional[HttpUrl] = None
    video_id: Optional[str] = None
    conversation_id: Optional[str] = None  # For resuming existing conversation
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "query": "What is this video about?",
                "conversation_id": None
            }
        }


class ConversationResponse(BaseModel):
    """Response model for Schema 2 conversation"""
    conversation_id: str
    video_id: str
    video_title: str
    user_message: str
    assistant_message: str
    message_index: int
    created_at: str


class ConversationMessageResponse(BaseModel):
    """Response model for individual message"""
    id: str
    conversation_id: str
    user_id: str
    video_id: str
    message_index: int
    role: str  # "user" | "assistant" | "system"
    content: str
    content_length: int
    tokens_estimate: Optional[int]
    created_at: str









#---------------------------------------------------------------------------------------------

# Environment variables already loaded at top of file

# Validate required API keys at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set. OpenAI features will fail.")
    print("   Set OPENAI_API_KEY in .env file to enable AI functionality.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("[WARN] PINECONE_API_KEY not set. Vector search will fail.")
    print("   Set PINECONE_API_KEY in .env file to enable RAG functionality.")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
if not GOOGLE_SEARCH_API_KEY:
    print("[WARN] GOOGLE_SEARCH_API_KEY not set. Google search tool will be disabled.")

GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
if not GOOGLE_SEARCH_ENGINE_ID:
    print("[WARN] GOOGLE_SEARCH_ENGINE_ID not set. Google search tool will be disabled.")

# Time-sensitive keywords for tool selection logic
TIME_SENSITIVE_KEYWORDS = ["latest", "recent", "new", "current", "2024", "2025", "now", "today", "update", "what is new"]

def is_time_sensitive_query(query: str) -> bool:
    """
    Check if a query is time-sensitive based on keywords.
    
    Args:
        query: The user query string
        
    Returns:
        True if query contains time-sensitive keywords, False otherwise
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in TIME_SENSITIVE_KEYWORDS)


YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip() or None
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "").strip() or None
YOUTUBE_COOKIES_HEADER = os.getenv("YOUTUBE_COOKIES_HEADER", "").strip() or None

YOUTUBE_PROXIES = None
if YOUTUBE_PROXY_URL:
    try:
        YOUTUBE_PROXIES = GenericProxyConfig(
            http_url=YOUTUBE_PROXY_URL,
            https_url=YOUTUBE_PROXY_URL,
        )
        # Mask password if present in URL
        safe_url = YOUTUBE_PROXY_URL.split('@')[-1] if '@' in YOUTUBE_PROXY_URL else YOUTUBE_PROXY_URL
        print(f"[INFO] Using YouTube proxy endpoint: {safe_url}")
    except Exception as e:
        # In production, if proxy is required, fail fast
        safe_url = YOUTUBE_PROXY_URL.split('@')[-1] if '@' in YOUTUBE_PROXY_URL else YOUTUBE_PROXY_URL
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise ValueError(
                f"Failed to configure required YouTube proxy: {e}. "
                "Set YOUTUBE_PROXY_URL correctly or disable proxy requirement."
            )
        print(f"[WARN] Could not configure YouTube proxy '{safe_url}': {e}")
        YOUTUBE_PROXIES = None

YOUTUBE_COOKIES = None
if YOUTUBE_COOKIES_FILE:
    try:
        YOUTUBE_COOKIES = Path(YOUTUBE_COOKIES_FILE).read_text(encoding="utf-8").strip()
        print(f"[INFO] Loaded YouTube cookies from file: {YOUTUBE_COOKIES_FILE}")
    except Exception as e:
        print(f"[WARN] Could not read YouTube cookies file '{YOUTUBE_COOKIES_FILE}': {e}")

if not YOUTUBE_COOKIES and YOUTUBE_COOKIES_HEADER:
    YOUTUBE_COOKIES = YOUTUBE_COOKIES_HEADER
    print("[INFO] Using YouTube cookies provided via environment header")

def parse_cookie_string(cookie_str: str) -> RequestsCookieJar:
    """Parse cookie string into RequestsCookieJar for robust cookie handling"""
    jar = RequestsCookieJar()
    cookies = SimpleCookie()
    cookies.load(cookie_str)
    for key, morsel in cookies.items():
        jar.set(key, morsel.value, domain='.youtube.com')
    return jar

# Global HTTP client for connection pooling
GLOBAL_HTTP_CLIENT = None
if YOUTUBE_COOKIES:
    GLOBAL_HTTP_CLIENT = RequestsSession()
    # Use cookie jar for robust cookie handling
    cookie_jar = parse_cookie_string(YOUTUBE_COOKIES)
    GLOBAL_HTTP_CLIENT.cookies.update(cookie_jar)

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))  # Uses OPENAI_API_KEY from environment

# ============================================================================
# AGENT PROMPT CONSTRUCTION
# ============================================================================

def build_react_agent_prompt(base_template: Optional[str] = None) -> PromptTemplate:
    """
    Build ReAct agent prompt with simplified state machine instructions.
    
    Simplified from 200+ lines to ~50 lines focusing on:
    - Format rules
    - JSON-based state machine
    - Tool selection
    - Final answer format
    - Loop prevention
    
    Args:
        base_template: Optional base template from LangChain hub. If None, uses complete fallback.
    
    Returns:
        PromptTemplate configured with simplified agent instructions.
    """
    # Get current date/time for context
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "January 15, 2025"
    current_year = datetime.now().year
    
    # Simplified instructions (~50 lines total)
    simplified_instructions = f"""

CURRENT DATE/TIME CONTEXT:
- Today's date: {current_date}
- Current year: {current_year}
- Use this information to understand what "latest", "recent", "new", "current" means - these refer to information from {current_year}, not past years.

CRITICAL: TOOL SELECTION (READ THIS FIRST):
- BEFORE choosing any tool, analyze the query for time-sensitive keywords: "latest", "recent", "new", "current", "2024", "2025", "now", "today", "update", "what is new"
- Check if the query contains ANY of these time-sensitive keywords (this is the same logic used by tools internally)
- IMPORTANT: If query contains "latest", "recent", "new", "current", "update" BUT ALSO contains a past date (before {current_year}), the user likely wants CURRENT information from {current_year}, not historical information from that past date. In this case, modify your search query to focus on current/latest information, not the past date.
- Examples:
  * "Tell me about what is new in the AI world" → Contains "new" → Call google_search_tool_v2 FIRST
  * "Latest updates in AI" → Contains "latest" → Call google_search_tool_v2 FIRST
  * "Update to {current_year} please" → Contains "update" and "{current_year}" → Call google_search_tool_v2 FIRST
  * "Recent developments" → Contains "recent" → Call google_search_tool_v2 FIRST
- If query contains ANY of these time-sensitive keywords → You MUST call google_search_tool_v2 FIRST (not youtube_rag_tool_general)
- Only after getting web search results, you may optionally call youtube_rag_tool_general for additional context
- For NON-time-sensitive queries → Call youtube_rag_tool_v2 or youtube_rag_tool_general FIRST

CHAT CONTEXT (MAINTAIN CONVERSATION FLOW):
- You are in a conversation with a user. ALWAYS check chat_history to understand the conversation context.
- The chat_history contains previous messages in the conversation. Use it to understand what the user is referring to in follow-up questions.

HOW TO USE CHAT HISTORY:
1. When user asks a follow-up question, look at chat_history to see what they're referring to
2. Combine the context from chat_history with the current query to understand the full intent
3. If chat_history shows a previous topic, the current query likely relates to that topic

CONCRETE EXAMPLES:

Example 1 - Follow-up with update request:
  chat_history:
    Human: "Tell me about AI developments in 2023"
    Assistant: "In 2023, AI saw major developments including GPT-4 release..."
  Current query: "Update to 2025 please"
  → You should: Search for 2025 AI developments (not repeat 2023 info)
  → Action: google_search_tool_v2
  → Action Input: "AI developments in 2025"

Example 2 - Follow-up asking for more detail:
  chat_history:
    Human: "What is machine learning?"
    Assistant: "Machine learning is a subset of AI..."
  Current query: "Can you be more specific about neural networks?"
  → You should: Combine "machine learning" context with "neural networks" query
  → Action: youtube_rag_tool_general
  → Action Input: "neural networks in machine learning"

Example 3 - Follow-up with clarification:
  chat_history:
    Human: "Explain async programming"
    Assistant: "Async programming allows concurrent execution..."
  Current query: "What about asyncio in Python specifically?"
  → You should: Focus on Python's asyncio (not general async programming)
  → Action: youtube_rag_tool_general
  → Action Input: "Python asyncio async programming"

DETECTING FOLLOW-UP QUESTIONS:
- Phrases like "update to", "be more specific", "what about", "tell me more", "how about" indicate follow-ups
- If current query is very short or vague, check chat_history for context
- When user asks for updates/changes → They likely want CURRENT information, so use google_search_tool_v2

CRITICAL: Always combine chat_history context with current query to understand what the user is actually asking. Don't treat each query in isolation.

FORMAT RULES:
- Action and Action Input must be on SEPARATE lines
- Do NOT use function call syntax like tool_name("input")
- Example: Action: youtube_rag_tool_v2
          Action Input: "what is asyncio?"

STATE MACHINE (CRITICAL - Parse JSON):

1. Tool returns Observation that looks like:
   ===JSON_RESPONSE_START===
   {{{{ "status": "next_action_required", "next_action": "youtube_rag_tool_general", "context": "..." }}}}
   ===JSON_RESPONSE_END===
   You MUST parse this JSON. Extract 'status' and 'next_action' fields.

2. You MUST extract fields from the JSON:
   - Find text between ===JSON_RESPONSE_START=== and ===JSON_RESPONSE_END===
   - Parse that JSON: status = "next_action_required"  # From the JSON
   - Extract: next_action = "youtube_rag_tool_general"  # From the JSON
   - Extract: context = "..."  # From the JSON

3. Actions based on status:
   - If status == "final_answer_required" → STOP immediately, provide Final Answer using context
   - If status == "next_action_required" → Check "next_action" field:
     * CRITICAL: You MUST call the tool specified in "next_action" field (if you haven't called it yet)
     * If next_action == "google_search_tool_v2" and you haven't called it → Call google_search_tool_v2
     * If next_action == "youtube_rag_tool_general" and you haven't called it → Call youtube_rag_tool_general
     * If next_action == "youtube_rag_tool_v2" and you haven't called it → Call youtube_rag_tool_v2
     * If next_action is a tool you already called → Provide Final Answer using the context you have
   - If status == "no_results" → Try alternative tool or provide answer
     * Example: If youtube_rag_tool_general returns "no_results", try google_search_tool_v2 for current information
     * If all tools return "no_results", provide Final Answer explaining that no information was found
   - If status == "error" → Handle error based on error type:
     * Check the 'context' field for error details
     * If error is recoverable (e.g., "no results", "service temporarily unavailable"):
       - Try alternative tool if available
       - Example: Observation: {{{{ "status": "error", "context": "No relevant information found. Consider using google_search_tool_v2 if this is a current events question." }}}}
         → Thought: The tool suggests trying google_search_tool_v2. I should call it.
         → Action: google_search_tool_v2
     * If error is fatal (e.g., "API key invalid", "service not configured"):
       - Provide Final Answer explaining the limitation
       - Example: Observation: {{{{ "status": "error", "context": "Web search is not configured. Please provide GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID." }}}}
         → Thought: I cannot access web search. I should provide Final Answer using available video context or explain the limitation.
         → Final Answer: [Your answer using available context, or explanation of limitation]
     * Always check error context before deciding on recovery action

4. Example flow (following next_action):
   Action: youtube_rag_tool_general
   Action Input: "how to be specific"
   Observation: ===JSON_RESPONSE_START===
                {{{{ "status": "next_action_required", "next_action": "google_search_tool_v2", "context": "..." }}}}
                ===JSON_RESPONSE_END===
   Thought: I see status="next_action_required" and next_action="google_search_tool_v2". The tool response tells me to call google_search_tool_v2 next. I should follow this instruction and call google_search_tool_v2.
   Action: google_search_tool_v2
   Action Input: "how to be specific in communication"

TOOL SELECTION REMINDER:
- This is a reminder - you should have already checked for time-sensitive keywords above
- TIME-SENSITIVE queries → google_search_tool_v2 FIRST (always)
- GENERAL queries → youtube_rag_tool_v2 or youtube_rag_tool_general FIRST

FINAL ANSWER FORMAT:
- When status is "final_answer_required", your Thought MUST be: "I now know the final answer"
- Synthesize comprehensively from ALL context in the "context" field
- Include specific details, examples, and key points
- Format: Final Answer: [Your comprehensive answer here]
- After "Final Answer:", STOP IMMEDIATELY - do not output anything else

LOOP PREVENTION:
- Before calling ANY tool, check your scratchpad for "Action: [tool_name]"
- If you see "Action: google_search_tool_v2" already in scratchpad → DO NOT call it again
- If you see "Action: youtube_rag_tool_general" already in scratchpad → DO NOT call it again  
- If you see "Action: youtube_rag_tool_v2" already in scratchpad → DO NOT call it again
- Maximum 1 call per tool per query - this is enforced programmatically
- If you try to call a tool twice, you will get a plain-text error message → Then provide Final Answer immediately
- CRITICAL: If you see "ERROR: Tool X already called" in Observation → STOP calling tools immediately, provide Final Answer using ALL context from previous Observations
- When you see an error about a tool already being called, your Thought MUST be: "I see an error that this tool was already called. I should provide Final Answer now using the context I have from previous tool calls."
"""
    
    # Base template from hub or fallback
    if base_template:
        # Check if base template includes chat_history placeholder
        # If not, we need to add it
        if "{chat_history}" not in base_template:
            # Insert chat_history section before the question/input section
            # Look for common patterns like "{input}" or "Question:" to insert before
            if "{input}" in base_template:
                base_template = base_template.replace("{input}", "{chat_history}\n\nQuestion: {input}")
            elif "Question:" in base_template:
                base_template = base_template.replace("Question:", "{chat_history}\n\nQuestion:")
            else:
                # Fallback: add at the beginning
                base_template = "{chat_history}\n\n" + base_template
        
        # Enhance hub template with simplified instructions
        enhanced_template = base_template + simplified_instructions
        return PromptTemplate.from_template(enhanced_template)
    else:
        # Complete fallback prompt with simplified instructions
        fallback_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action (JSON string with {{{{ "status" }}}} and {{{{ "context" }}}} fields)
... (this Thought/Action/Action Input/Observation can repeat N times)

When Observation contains JSON with {{{{ "status": "final_answer_required" }}}}, you MUST stop calling tools immediately:
Example:
Action: youtube_rag_tool_v2
Action Input: "What is discussed in this video?"
Observation: {{{{ "status": "final_answer_required", "context": "..." }}}}
Thought: I now know the final answer
Final Answer: [Your comprehensive answer here]

CRITICAL: After you write "Final Answer:", you MUST IMMEDIATELY STOP.
""" + simplified_instructions + """
Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""
        return PromptTemplate.from_template(fallback_template)


# Load agent prompt once at startup (cache it)
try:
    hub_client = HubClient()
    base_prompt = hub_client.pull("hwchase17/react")
    # Enhance prompt with explicit tool format examples
    # PromptTemplate already imported at top of file
    
    # Handle different return types from hub_client.pull()
    original_template = None
    
    if isinstance(base_prompt, PromptTemplate):
        # Direct PromptTemplate object
        original_template = base_prompt.template
    elif isinstance(base_prompt, str):
        # Could be JSON string (serialized PromptTemplate) or plain string
        try:
            import json
            # Try to parse as JSON (LangChain serialization format)
            prompt_data = json.loads(base_prompt)
            if isinstance(prompt_data, dict) and 'kwargs' in prompt_data and 'template' in prompt_data['kwargs']:
                original_template = prompt_data['kwargs']['template']
                print("[INFO] Extracted template from JSON serialized PromptTemplate")
            else:
                # Plain string - check if it has required variables
                required_vars = ['{tools}', '{tool_names}', '{agent_scratchpad}']
                if all(var in base_prompt for var in required_vars):
                    original_template = base_prompt
                    print("[INFO] Using base_prompt as plain string template")
                else:
                    raise ValueError("String template missing required variables")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Not JSON or missing required variables - will use fallback
            original_template = None
            print(f"[WARN] Could not extract template from base_prompt string, will use fallback")
    else:
        # Unexpected type
        print(f"[WARN] base_prompt is unexpected type: {type(base_prompt)}, will use fallback")
        original_template = None
    
    if original_template:
        # We have a valid template - use function to build prompt
        REACT_AGENT_PROMPT = build_react_agent_prompt(base_template=original_template)
        print("[OK] Agent prompt loaded from LangChain hub and enhanced with format instructions")
    else:
        # No valid template extracted - use fallback prompt via function
        print("[WARN] Could not extract valid template from hub, using fallback prompt")
        REACT_AGENT_PROMPT = build_react_agent_prompt(base_template=None)
        print("[OK] Fallback prompt loaded")
except Exception as e:
    print(f"[WARN] Could not load agent prompt from hub: {e}")
    print("   Using fallback prompt")
    # Fallback prompt if hub is unavailable - use function to build it
    REACT_AGENT_PROMPT = build_react_agent_prompt(base_template=None)

#Pinecone Database setting-----------------------------------------------------------------------------
# Initialize Pinecone client - using shared index with namespaces
SHARED_INDEX_NAME = "augustus-videos-integrated"
CATALOG_INDEX_NAME = os.getenv("CATALOG_INDEX_NAME", "agentic-catalog-index")

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("[OK] Pinecone client initialized successfully")
        print(f"   Using shared index: {SHARED_INDEX_NAME}")
    except Exception as e:
        print(f"[WARN] Pinecone initialization failed: {e}")
        print("   Pinecone features will not be available.")
        pc = None
else:
    print("[WARN] Pinecone not configured (PINECONE_API_KEY not set)")
    pc = None

# Retrieval Quality Configuration
ENABLE_SCORE_THRESHOLDING = True   # Set to False to disable score filtering (useful for debugging score distributions)
MIN_SIMILARITY_SCORE = 0.0         # Minimum reranking score to include (start with 0.0, tune based on actual score distributions)
MAX_CHUNKS_PER_VIDEO = 10           # Maximum chunks from same video (ensures diversity)
MAX_CONTEXT_LENGTH = 40000          # Maximum characters in context (prevents LLM overflow)
BASE_TOP_K = 12                    # Base number of results for youtube_rag_tool (already used: 12*2 = 24 candidates)
FALLBACK_TOP_K = 10                # Base number of results for fallback RAG (already used: 10*2 = 20 candidates)
MAX_QUERY_LENGTH = 2000            # Maximum characters allowed in user query (prevents context window exhaustion)

# Context Quality Assessment Configuration (for hybrid approach)
MIN_CONTEXT_QUALITY_AVG_SCORE = 0.3  # Minimum average score for "good" context quality
MIN_CONTEXT_QUALITY_MAX_SCORE = 0.5  # Minimum max score for "good" context quality
MIN_CONTEXT_QUALITY_DOC_COUNT = 2    # Minimum number of docs for "good" context quality

parser = StrOutputParser()



#Embedding generation-----------------------
# NOTE: OpenAI embeddings removed - using Pinecone integrated embeddings instead
# embedding=OpenAIEmbeddings(model='text-embedding-3-large')  # No longer needed

# Note: Tools and agent are created per-request inside the endpoint
# This ensures proper context isolation and prevents race conditions




#Fetching video transcript
def fetchTranscript(url:str)->list:
    video_id=get_youtube_video_id(url)

    try:
        # Reuse global client if available, otherwise create new one
        http_client = GLOBAL_HTTP_CLIENT
        if not http_client and YOUTUBE_COOKIES:
            # Fallback: create new client if global doesn't exist and cookies are needed
            # youtube-transcript-api removed the cookies_config parameter; instead we
            # attach cookies via a custom requests.Session.
            http_client = RequestsSession()
            cookie_jar = parse_cookie_string(YOUTUBE_COOKIES)
            http_client.cookies.update(cookie_jar)

        client = YouTubeTranscriptApi(
            proxy_config=YOUTUBE_PROXIES,
            http_client=http_client,
        )
        print(f"[INFO] Fetching transcript for video: {video_id}")
        transcripts = client.list(video_id)
        
        transcript_obj = transcripts.find_transcript(["en", "en-US", "en-GB", "hi"])
        transcript_list = transcript_obj.fetch()
        transcript = " ".join(chunk.text for chunk in transcript_list)
        if detect(transcript) == "hi":
            prompt_translate = PromptTemplate(
                template="<task>\nUse your intelligence to translate this Hindi text to english maintaining semantic meaning, correct english spelling and consistency in translation.</task>\n <context> \n {content}</context> ",
                input_variables=['content']
            )
            chain_translate = prompt_translate | model | parser
            transcript = chain_translate.invoke({'content':transcript})
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No captions/subtitles available for this video. Please try a video with captions enabled."
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript is available for this video in the requested languages."
        )
    except CouldNotRetrieveTranscript as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "YouTube is blocking transcript requests from this server. "
                "Configure a residential proxy via YOUTUBE_PROXY_URL or provide authenticated cookies "
                "via YOUTUBE_COOKIES_FILE / YOUTUBE_COOKIES_HEADER environment variables. "
                f"Original error: {str(e)}"
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch video transcript: {str(e)}"
        )

    l = [transcript, video_id]
    return l


def generate_video_synopsis(transcript: str, video_title: str = "YouTube Video") -> Tuple[str, List[str]]:
    """
    Generate 250-300 char synopsis and 3-5 topic keywords from transcript using LLM.
    
    Args:
        transcript: Full video transcript
        video_title: Video title for context
    
    Returns:
        Tuple of (synopsis: str, topics: List[str])
    """
    try:
        # Truncate transcript for cost efficiency (first ~4000 chars usually sufficient)
        transcript_preview = transcript[:4000] if len(transcript) > 4000 else transcript
        
        synopsis_prompt = f"""Analyze this video transcript and provide:
1. A concise synopsis (250-300 characters) summarizing the main topic and key points
2. 3-5 topic keywords (single words or short phrases, lowercase)

Video Title: {video_title}

Transcript:
{transcript_preview}

Format your response as:
SYNOPSIS: [your 250-300 char synopsis]
TOPICS: [keyword1, keyword2, keyword3, ...]"""
        
        response = model.invoke(synopsis_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response
        synopsis = ""
        topics = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('SYNOPSIS:'):
                synopsis = line.replace('SYNOPSIS:', '').strip()
            elif line.startswith('TOPICS:'):
                topics_str = line.replace('TOPICS:', '').strip()
                topics = [t.strip().lower() for t in topics_str.split(',') if t.strip()]
        
        # Validate and trim synopsis
        if not synopsis:
            synopsis = f"Video about {video_title}"[:300]
        elif len(synopsis) > 300:
            synopsis = synopsis[:297] + "..."
        elif len(synopsis) < 250:
            # Pad if too short (acceptable, not critical)
            pass
        
        # Validate topics
        if not topics or len(topics) < 3:
            topics = ["video", "content", "information"]
        elif len(topics) > 5:
            topics = topics[:5]
        
        return synopsis, topics
    
    except Exception as e:
        print(f"[WARN] Synopsis generation failed: {e}, using defaults")
        return f"Video: {video_title}"[:300], ["video", "content"]


def assess_context_quality(docs: List[Document]) -> Tuple[bool, str]:
    """
    Assess whether retrieved context quality is sufficient for final answer.
    
    Uses multiple quality indicators:
    - Number of documents
    - Average and maximum similarity scores
    - Score distribution
    
    Args:
        docs: List of Document objects with _score in metadata
    
    Returns:
        Tuple of (is_sufficient: bool, reason: str)
        - is_sufficient: True if context quality is good enough for final answer
        - reason: Explanation of quality assessment
    """
    if not docs:
        return False, "No documents retrieved"
    
    # Extract scores
    scores = []
    for doc in docs:
        score = doc.metadata.get('_score', 0.0)
        if isinstance(score, (int, float)):
            scores.append(score)
    
    if not scores:
        return False, "No valid scores in documents"
    
    doc_count = len(docs)
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    
    # Quality assessment criteria
    has_sufficient_docs = doc_count >= MIN_CONTEXT_QUALITY_DOC_COUNT
    has_good_avg_score = avg_score >= MIN_CONTEXT_QUALITY_AVG_SCORE
    has_good_max_score = max_score >= MIN_CONTEXT_QUALITY_MAX_SCORE
    
    # Context is sufficient if:
    # 1. We have enough docs AND (good avg score OR good max score)
    # OR 2. Very high max score even with fewer docs (top-quality match)
    if has_sufficient_docs and (has_good_avg_score or has_good_max_score):
        return True, f"Quality context: {doc_count} docs, avg={avg_score:.3f}, max={max_score:.3f}"
    elif max_score >= 0.7:  # Exceptionally high score
        return True, f"High-quality match: max={max_score:.3f} (single excellent match)"
    else:
        reason = f"Low-quality context: {doc_count} docs, avg={avg_score:.3f}, max={max_score:.3f} (below thresholds)"
        return False, reason


def format_docs(retrieved_docs, max_length=None):
    """
    Format retrieved documents into context string.
    
    Args:
        retrieved_docs: List of Document objects
        max_length: Maximum context length in characters (optional, defensive check)
    
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return ""
    
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Defensive check: truncate if somehow exceeds limit
    if max_length and len(context_text) > max_length:
        print(f"[WARN] Context exceeded limit ({len(context_text)} > {max_length}), truncating")
        context_text = context_text[:max_length] + "..."
    
    return context_text


def format_citations_from_docs(docs: List[Document], db_session_or_factory: Callable[[], Session] | Session) -> Dict[str, Dict[str, str]]:
    """
    Extract unique video citations from documents.
    
    Args:
        docs: List of Document objects with video_id in metadata
        db_session_or_factory: Either a Session object or a callable that returns a Session
                              (for use in tool closures where session may become stale)
    
    Returns:
        Dict mapping video_id to {title, url, channel} for citations
    """
    video_ids = set()
    for doc in docs:
        vid_id = doc.metadata.get('video_id')
        if vid_id and vid_id != "GENERAL":
            video_ids.add(vid_id)
    
    # Get database session (create new one if factory provided, otherwise use passed session)
    if callable(db_session_or_factory):
        db = db_session_or_factory()
        should_close = True
    else:
        db = db_session_or_factory
        should_close = False
    
    try:
        citations = {}
        for vid_id in video_ids:
            catalog = db.query(VideosCatalog).filter(VideosCatalog.video_id == vid_id).first()
            if catalog:
                # Trim title to ~90 chars
                title = catalog.video_title[:90] + "..." if len(catalog.video_title) > 90 else catalog.video_title
                citations[vid_id] = {
                    "title": title,
                    "url": catalog.video_url,
                    "channel": catalog.channel_title or ""
                }
            else:
                print(f"[WARN] No catalog entry found for video_id: {vid_id}")
        
        # Check for duplicate titles (add channel name if duplicates)
        titles = [c["title"] for c in citations.values()]
        if len(titles) != len(set(titles)):
            # Has duplicates - append channel names
            for vid_id, info in citations.items():
                if info["channel"]:
                    info["title"] = f"{info['title']} ({info['channel']})"
        
        return citations
    finally:
        # Only close session if we created it (via factory)
        if should_close:
            db.close()


def process_retrieval_results(docs, min_score=None, max_per_video=None, max_context_length=None):
    """
    Post-process retrieval results with score thresholding, diversity, and context management.
    
    Args:
        docs: List of Document objects from retrieval
        min_score: Minimum similarity score threshold (default: MIN_SIMILARITY_SCORE)
        max_per_video: Maximum chunks per video (default: MAX_CHUNKS_PER_VIDEO)
        max_context_length: Maximum context length in characters (default: MAX_CONTEXT_LENGTH)
    
    Returns:
        List of Document objects after filtering and limiting
    """
    if not docs:
        return []
    
    # Use defaults if not provided
    min_score = min_score if min_score is not None else MIN_SIMILARITY_SCORE
    max_per_video = max_per_video if max_per_video is not None else MAX_CHUNKS_PER_VIDEO
    max_context_length = max_context_length if max_context_length is not None else MAX_CONTEXT_LENGTH
    
    # Step 1: Filter by score threshold
    # CRITICAL: Add defensive error handling for score access
    # LOG ACTUAL SCORES BEFORE FILTERING (for debugging score distributions)
    if docs:
        raw_scores = []
        for doc in docs:
            score = doc.metadata.get('_score', 0.0)
            if isinstance(score, (int, float)):
                raw_scores.append(score)
        
        if raw_scores:
            print(f"[RETRIEVAL] Raw scores before filtering: count={len(raw_scores)}, min={min(raw_scores):.3f}, max={max(raw_scores):.3f}, mean={sum(raw_scores)/len(raw_scores):.3f}, median={sorted(raw_scores)[len(raw_scores)//2]:.3f}")
        else:
            print(f"[RETRIEVAL] Warning: No valid scores found in {len(docs)} documents")
    
    # Apply score thresholding only if enabled
    if ENABLE_SCORE_THRESHOLDING:
        filtered_docs = []
        for doc in docs:
            score = doc.metadata.get('_score', 0.0)
            # Defensive check: validate score type (AGENTS.md line 696: scores are floats)
            if not isinstance(score, (int, float)):
                score = 0.0  # Default if invalid type
            if score >= min_score:
                filtered_docs.append(doc)
        
        # If all results filtered, log warning and check if we should use fallback
        if not filtered_docs:
            print(f"[WARN] All {len(docs)} results filtered out by score threshold ({min_score})")
            # Fallback: if threshold is too aggressive, use top results anyway (prevents complete failure)
            # This ensures we always have some context even if scores are low
            if docs and min_score > 0:
                # Sort by score and take top results as fallback
                sorted_fallback = sorted(docs, key=lambda x: x.metadata.get('_score', 0), reverse=True)
                fallback_count = min(3, len(sorted_fallback))  # Take top 3 as fallback
                filtered_docs = sorted_fallback[:fallback_count]
                print(f"[RETRIEVAL] Fallback: Using top {fallback_count} results despite low scores")
                if fallback_count > 0:
                    fallback_scores = [d.metadata.get('_score', 0) for d in filtered_docs]
                    print(f"[RETRIEVAL] Fallback scores: {[f'{s:.3f}' for s in fallback_scores]}")
            else:
                return []  # No fallback possible
    else:
        # Score thresholding disabled - use all results
        filtered_docs = docs
        print(f"[RETRIEVAL] Score thresholding disabled, using all {len(docs)} results")
    
    # Step 2: Apply diversity (limit chunks per video)
    # Sort by score first (highest first) - maintains reranking quality
    sorted_docs = sorted(filtered_docs, key=lambda x: x.metadata.get('_score', 0), reverse=True)
    
    diverse_docs = []
    video_count = {}
    for doc in sorted_docs:
        vid_id = doc.metadata.get('video_id')
        if vid_id:
            count = video_count.get(vid_id, 0)
            if count < max_per_video:
                diverse_docs.append(doc)
                video_count[vid_id] = count + 1
        else:
            # Include if no video_id (shouldn't happen, but safe)
            diverse_docs.append(doc)
    
    # Step 3: Limit context length
    total_length = 0
    final_docs = []
    for doc in diverse_docs:
        doc_length = len(doc.page_content)
        if total_length + doc_length <= max_context_length:
            final_docs.append(doc)
            total_length += doc_length
        else:
            # Truncate last doc if there's meaningful space remaining
            remaining = max_context_length - total_length
            if remaining > 100:  # Only if meaningful content remains (at least 100 chars)
                truncated_doc = Document(
                    page_content=doc.page_content[:remaining] + "...",
                    metadata=doc.metadata
                )
                final_docs.append(truncated_doc)
            break
    
    # Log processing stats
    print(f"[RETRIEVAL] Processed {len(docs)} -> {len(filtered_docs)} (score) -> {len(diverse_docs)} (diversity) -> {len(final_docs)} (context) results")
    if final_docs:
        scores = [d.metadata.get('_score', 0) for d in final_docs]
        print(f"[RETRIEVAL] Final score range: {min(scores):.3f} - {max(scores):.3f}, total context: {total_length} chars")
    
    return final_docs


# Helper function to get shared Pinecone index
def get_shared_index():
    """
    Get the shared Pinecone index for all users.
    Data isolation is achieved through namespaces.
    
    Returns:
        Pinecone Index object for the shared index
        
    Raises:
        HTTPException: If Pinecone is not available or index doesn't exist
    """
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )
    
    if not pc.has_index(SHARED_INDEX_NAME):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Shared index '{SHARED_INDEX_NAME}' not found. Run: pc index create -n {SHARED_INDEX_NAME} -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content"
        )
    
    return pc.Index(SHARED_INDEX_NAME)


def get_catalog_index():
    """
    Get Pinecone catalog index for video synopsis search.
    
    Returns:
        Pinecone Index object for the catalog index
        
    Raises:
        HTTPException: If Pinecone is not available or catalog index doesn't exist
    """
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )
    
    if not pc.has_index(CATALOG_INDEX_NAME):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Catalog index '{CATALOG_INDEX_NAME}' not found. Create it with: pc index create -n {CATALOG_INDEX_NAME} -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content"
        )
    
    return pc.Index(CATALOG_INDEX_NAME)


# Helper function to generate namespace for user
def get_user_namespace(username: str) -> str:
    """
    Generate namespace for user: user_{username}
    Ensures data isolation per user within the shared index.
    
    Args:
        username: The username to generate namespace for
        
    Returns:
        Namespace string in format: user_{sanitized_username}
    """
    # Sanitize username for namespace (similar to old index name logic)
    sanitized = username.lower().replace("_", "-").replace(" ", "-")
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '-')
    sanitized = sanitized.strip('-')
    return f"user_{sanitized}"


def check_video_exists_in_pinecone(index, namespace: str, video_id: str) -> bool:
    """
    Check if any records exist for a given video_id in namespace.
    
    Uses search with metadata filter to check existence efficiently.
    This approach works reliably with the current Pinecone SDK (index.list() 
    returns a generator, not an object with .records attribute).
    
    Args:
        index: Pinecone Index object (from get_shared_index())
        namespace: Namespace string (from get_user_namespace())
        video_id: YouTube video ID to check
        
    Returns:
        True if any records with video_id exist, False otherwise
        
    Note:
        Uses index.search() with filter (AGENTS.md lines 825-841).
        Matches existing pattern used at line 1455.
        Returns False on any error (conservative: assumes not found, will fetch transcript).
    """
    try:
        # Use search with metadata filter for existence check
        # AGENTS.md line 825-841: Dynamic filter pattern
        # Matches existing pattern at line 1455: {"video_id": video_id}
        query_dict = {
            "top_k": 1,  # Only need to check existence (minimal overhead)
            "inputs": {
                "text": "video"  # Generic query text for embeddings (needs valid text, not "dummy")
            },
            "filter": {"video_id": video_id}  # Filter by video_id (shorthand format, matches line 1455)
        }
        
        # Search without reranking (unnecessary overhead for existence check)
        # AGENTS.md line 662-674 says "always rerank" but that's for quality, not existence checks
        result = exponential_backoff_retry(
            lambda: index.search(
                namespace=namespace,
                query=query_dict
                # No rerank parameter - we only need existence check
            )
        )
        
        # Access results via dict-style (matches existing pattern at line 1481, 2034)
        # AGENTS.md line 221: reranked_results['result']['hits']
        # Even without reranking, results have same structure: result['result']['hits']
        return len(result['result']['hits']) > 0
    except Exception as e:
        print(f"[WARN] Error checking video existence for {video_id}: {e}")
        # Conservative: assume not found, will fetch transcript (graceful degradation)
        return False


# Utility functions for Pinecone operations
def exponential_backoff_retry(func, max_retries=5):
    """
    Retry with exponential backoff for transient errors.
    
    Args:
        func: Function to retry (lambda or callable)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Result of func() if successful
        
    Raises:
        Exception: If all retries fail or non-retryable error occurs
    """
    from pinecone.exceptions import PineconeException
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except PineconeException as e:
            status_code = getattr(e, 'status', None)
            if status_code and (status_code >= 500 or status_code == 429):
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 60)  # Exponential backoff, cap at 60s
                    time.sleep(delay)
                else:
                    raise
            else:
                raise  # Don't retry client errors (4xx except 429)


def batch_upsert_records(index, namespace, records, batch_size=96):
    """
    Upsert records in batches with retry logic (for integrated embeddings).
    Pinecone automatically generates embeddings from the 'content' field.
    
    CRITICAL CONSTRAINTS (AGENTS.md line 641-647, 693):
    - Text records: MAX 96 per batch, 2MB total per batch
    - Always use namespaces (AGENTS.md line 656-657)
    - Use exponential_backoff_retry for transient errors (AGENTS.md line 715-733)
    
    Args:
        index: Pinecone Index object
        namespace: Namespace string (REQUIRED - AGENTS.md line 656-657)
        records: List of record dictionaries with '_id', 'content', and metadata
        batch_size: Number of records per batch (max 96 for text records - AGENTS.md line 693)
    """
    import time
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        # CRITICAL: Method signature is index.upsert_records(namespace, records) - AGENTS.md line 416
        exponential_backoff_retry(
            lambda: index.upsert_records(namespace, batch)
        )
        time.sleep(0.01)  # Rate limiting (reduced from 0.1s for better performance)


def upsert_catalog_to_pinecone(
    video_id: str,
    synopsis: str,
    video_title: str,
    topics: List[str],
    namespace: str,
    channel_title: Optional[str] = None
) -> bool:
    """
    Upsert single catalog entry to Pinecone catalog index.
    Uses user-specific namespace for per-user catalog isolation.
    
    Args:
        video_id: YouTube video ID (11 chars)
        synopsis: Video synopsis (250-300 chars)
        video_title: Video title
        topics: List of topic keywords (3-5 items)
        namespace: User namespace for catalog isolation
        channel_title: Optional channel name
    
    Returns:
        True if successful, False otherwise
    """
    try:
        catalog_index = get_catalog_index()
        
        # Prepare catalog record
        record = {
            "_id": video_id,
            "content": synopsis,  # This gets embedded by Pinecone
            "video_id": video_id,
            "video_title": video_title,
            "topics": topics,
            "channel_title": channel_title or ""
        }
        
        # Upsert to catalog index (user-specific namespace for isolation)
        exponential_backoff_retry(
            lambda: catalog_index.upsert_records(namespace, [record])
        )
        
        print(f"[OK] Synced catalog entry to Pinecone for video {video_id} in namespace {namespace}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed to sync catalog to Pinecone for {video_id}: {e}")
        return False


def embed_video_if_needed(
    db: Session,
    video_id: str,
    video_url: str,
    user_id,
    index,
    namespace: str
) -> Tuple[str, str]:
    """
    Embed video into Pinecone if not already embedded.
    Also creates/updates videos_catalog entry with basic metadata.
    
    Args:
        db: Database session
        video_id: YouTube video ID (11 chars)
        video_url: Full YouTube URL
        user_id: User ID for tracking
        index: Pinecone Index object
        namespace: Pinecone namespace for user
    
    Returns:
        Tuple of (video_id, video_title)
    
    Raises:
        HTTPException: If transcript fetch or embedding fails
    """
    # Check if video already embedded in Pinecone
    if check_video_exists_in_pinecone(index, namespace, video_id):
        print(f"[INFO] Video {video_id} already embedded, skipping transcript fetch")
        
        # Try to get title from catalog, fallback to generic
        catalog = db.query(VideosCatalog).filter(
            VideosCatalog.video_id == video_id
        ).first()
        video_title = catalog.video_title if catalog else "YouTube Video"
        
        return video_id, video_title
    
    # Video not found in Pinecone - fetch transcript and embed
    print(f"[INFO] Video {video_id} not found in Pinecone, fetching transcript")
    
    # Fetch transcript
    l = fetchTranscript(video_url)
    transcript = l[0]
    
    # Generate synopsis and topics for catalog
    print(f"[INFO] Generating synopsis for video {video_id}")
    video_title = "YouTube Video"  # Default fallback
    synopsis, topics = generate_video_synopsis(transcript, video_title)
    print(f"[OK] Generated synopsis ({len(synopsis)} chars) and {len(topics)} topics")
    
    # Embed in Pinecone
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    chunks = parent_splitter.create_documents([transcript])
    
    # Prepare records (text only - Pinecone embeds automatically)
    records = []
    for i, chunk in enumerate(chunks):
        vector_id = f"{video_id}:{hashlib.sha1(chunk.page_content.encode('utf-8')).hexdigest()[:16]}"
        records.append({
            "_id": vector_id,
            "content": chunk.page_content,
            "video_id": video_id,
            "doc_id": f"{video_id}:p{i}"
        })
    
    # Defensive check: verify records don't already exist
    already = False
    if records:
        probe_ids = [r["_id"] for r in records[:8]]
        try:
            result = index.fetch(namespace=namespace, ids=probe_ids)
            already = len(result.records) > 0
        except Exception:
            already = False
    
    # Batch upsert with retry - must succeed before creating catalog entry
    try:
        if not already:
            batch_upsert_records(index, namespace, records, batch_size=96)
            print(f"[OK] Embedded video {video_id}")
        else:
            print(f"[INFO] Video {video_id} already embedded (defensive check triggered)")
        
        # Only create catalog entry if Pinecone embedding succeeded
        # Create/update catalog entry with synopsis and topics
        catalog = get_or_create_catalog_entry(
            db=db,
            video_id=video_id,
            video_title=video_title,
            video_url=video_url,
            synopsis=synopsis,  # Now populated with LLM-generated synopsis
            topics=topics,      # Now populated with LLM-generated topics
            embedding_version="llama-text-embed-v2"
        )
        
        # Sync catalog entry to Pinecone catalog index (per-user namespace)
        upsert_catalog_to_pinecone(
            video_id=video_id,
            synopsis=synopsis,
            video_title=video_title,
            topics=topics,
            namespace=namespace,  # User-specific namespace for catalog isolation
            channel_title=None  # Could be extracted from YouTube API in future
        )
        
        return video_id, catalog.video_title
    
    except Exception as e:
        # If Pinecone embedding fails, don't create catalog entry to maintain consistency
        print(f"[ERROR] Pinecone embedding failed for video {video_id}: {e}")
        print(f"[ERROR] Skipping catalog entry creation to maintain data consistency")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed video: {str(e)}"
        )


def shortlist_videos_from_catalog(
    query: str,
    namespace: str,
    top_k: int = 10,
    top_n: int = 5
) -> List[str]:
    """
    Search catalog index to shortlist relevant video IDs.
    Uses user-specific namespace for per-user catalog isolation.
    
    Args:
        query: User query text
        namespace: User namespace for catalog search
        top_k: Candidate pool size (default: 10)
        top_n: Final shortlist size after reranking (default: 5)
    
    Returns:
        List of video_id strings (max top_n items), empty list on error
    """
    try:
        catalog_index = get_catalog_index()
        
        # Search catalog (user-specific namespace for isolation)
        results = exponential_backoff_retry(
            lambda: catalog_index.search(
                namespace=namespace,  # User-specific catalog
                query={
                    "top_k": top_k,
                    "inputs": {"text": query}
                },
                rerank={
                    "model": "bge-reranker-v2-m3",
                    "top_n": top_n,
                    "rank_fields": ["content"]
                }
            )
        )
        
        # Extract video_ids from results
        video_ids = []
        if results and 'result' in results and 'hits' in results['result']:
            for hit in results['result']['hits']:
                vid_id = hit.get('fields', {}).get('video_id')
                if vid_id:
                    video_ids.append(vid_id)
        
        print(f"[CATALOG] Shortlisted {len(video_ids)} videos for query: {query[:50]}... (namespace: {namespace})")
        return video_ids
    
    except Exception as e:
        print(f"[ERROR] Catalog shortlisting failed: {e}")
        return []  # Return empty list, will trigger fallback


def retrieve_for_general_tab(
    chunks_index,
    namespace: str,
    query: str,
    candidate_pool: int = 24,
    top_n: int = 12,
    max_per_video: int = 10,
    min_shortlist_videos: int = 3
) -> Tuple[List[Document], List[str]]:
    """
    General tab retrieval with catalog shortlisting.
    
    Steps:
    1. Shortlist videos via catalog search
    2. If shortlist < min_shortlist_videos, fall back to global chunk search
    3. Otherwise, search chunks filtered by shortlisted video_ids
    4. Apply diversity cap and score thresholding
    
    Args:
        chunks_index: Pinecone chunks index
        namespace: User namespace
        query: User query
        candidate_pool: Chunk candidates before reranking (default: 24)
        top_n: Final chunks after reranking (default: 12)
        max_per_video: Max chunks per video (default: 10)
        min_shortlist_videos: Min videos for filtered search (default: 3)
    
    Returns:
        Tuple of (documents: List[Document], shortlisted_video_ids: List[str])
    """
    # Step 1: Shortlist videos (using user-specific namespace)
    shortlisted_video_ids = shortlist_videos_from_catalog(query, namespace, top_k=10, top_n=5)
    
    # Step 2: Fallback check
    use_global_search = len(shortlisted_video_ids) < min_shortlist_videos
    
    if use_global_search:
        print(f"[GENERAL] Shortlist too small ({len(shortlisted_video_ids)} videos), using global search")
        filter_dict = None
    else:
        print(f"[GENERAL] Using filtered search with {len(shortlisted_video_ids)} shortlisted videos")
        # Pinecone filter syntax for "video_id in list"
        filter_dict = {"video_id": {"$in": shortlisted_video_ids}}
    
    # Step 3: Build query
    query_dict = {
        "top_k": candidate_pool,
        "inputs": {"text": query}
    }
    
    # Only add filter if exists (don't set to None)
    if filter_dict:
        query_dict["filter"] = filter_dict
    
    # Search chunks with reranking
    try:
        results = exponential_backoff_retry(
            lambda: chunks_index.search(
                namespace=namespace,
                query=query_dict,
                rerank={
                    "model": "bge-reranker-v2-m3",
                    "top_n": top_n,
                    "rank_fields": ["content"]
                }
            )
        )
        
        # Parse results
        raw_docs = []
        if results and 'result' in results and 'hits' in results['result']:
            for hit in results['result']['hits']:
                doc = Document(
                    page_content=hit['fields']['content'],
                    metadata={
                        'video_id': hit['fields'].get('video_id'),
                        'doc_id': hit['fields'].get('doc_id'),
                        '_id': hit['_id'],
                        '_score': hit['_score']
                    }
                )
                raw_docs.append(doc)
        
        # Step 4-5: Post-processing (diversity, context length, lenient scoring)
        final_docs = process_retrieval_results(
            raw_docs,
            min_score=0.5,  # Lenient threshold for General tab
            max_per_video=max_per_video,
            max_context_length=MAX_CONTEXT_LENGTH
        )
        
        return final_docs, shortlisted_video_ids
    
    except Exception as e:
        print(f"[ERROR] General tab retrieval failed: {e}")
        return [], shortlisted_video_ids


# ============================================================================
# DATABASE INTEGRITY UTILITIES (Phase 4)
# ============================================================================

def check_conversation_ownership_integrity(db: Session, conversation_id) -> Dict[str, Any]:
    """
    Check if conversation has valid user_id reference.
    
    Phase 4.10: Utility function to check for orphaned conversations.
    Used for diagnostic purposes to identify database integrity issues.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID to check (can be str or UUID)
    
    Returns:
        Dictionary with integrity check results:
        - "exists": bool - Whether conversation exists
        - "conversation_user_id": str - User ID that owns the conversation
        - "user_exists": bool - Whether the user still exists in database
        - "user_username": Optional[str] - Username if user exists
    """
    import uuid as uuid_lib
    # Convert to UUID if string
    if isinstance(conversation_id, str):
        conversation_id = uuid_lib.UUID(conversation_id)
    
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        return {"exists": False}
    
    user = db.query(User).filter(User.id == conversation.user_id).first()
    return {
        "exists": True,
        "conversation_user_id": str(conversation.user_id),
        "user_exists": user is not None,
        "user_username": user.username if user else None
    }


# Health check endpoint (no authentication required)
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check endpoint"""
    # Get CORS configuration for debugging
    cors_origins = os.getenv("CORS_ORIGINS", "NOT SET (defaulting to *)")
    
    health_status = {
        "status": "healthy",
        "cors_configured": cors_origins != "NOT SET (defaulting to *)",
        "cors_origins": cors_origins if cors_origins != "NOT SET (defaulting to *)" else "* (all origins allowed)",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database check
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"error: {str(e)}"
    
    # Pinecone check
    if pc is not None:
        try:
            if pc.has_index(SHARED_INDEX_NAME):
                health_status["checks"]["pinecone"] = "ok"
            else:
                health_status["checks"]["pinecone"] = f"index '{SHARED_INDEX_NAME}' not found"
        except Exception as e:
            health_status["checks"]["pinecone"] = f"error: {str(e)}"
    else:
        health_status["checks"]["pinecone"] = "not_configured"
    
    # OpenAI check (just verify key is set)
    health_status["checks"]["openai"] = "configured" if OPENAI_API_KEY else "not_configured"
    
    return health_status

# CORS test endpoint (no authentication required)
@app.get("/cors-test")
def cors_test():
    """Test endpoint to verify CORS configuration"""
    cors_origins = os.getenv("CORS_ORIGINS", "NOT SET (defaulting to *)")
    return {
        "cors_configured": cors_origins != "NOT SET (defaulting to *)",
        "cors_origins": cors_origins if cors_origins != "NOT SET (defaulting to *)" else "* (all origins allowed)",
        "message": "If you can see this from your frontend, CORS is working correctly!",
        "frontend_domains": [
            "https://www.chataugustus.com",
            "https://augustus-web-five.vercel.app"
        ] if cors_origins != "NOT SET (defaulting to *)" else "All origins allowed"
    }

# User Registration and Authentication Endpoints
@app.post("/signup", response_model=UserResponse)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash the password
    hashed_password = hash_password(user_data.password)
    
    # Create new user
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        is_active=True,
        scopes=["read", "write"]  # Default scopes for new users
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at.isoformat()
    )

@app.post("/signin", response_model=TokenResponse)
def signin(user_data: UserLogin, db: Session = Depends(get_db)):
    """Sign in user and return JWT token (legacy endpoint)"""
    # Check if user exists
    print(f"[INFO] Signing in user: {user_data.username}")
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token with scopes
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Phase 2.5: Include user_id in token payload
    access_token = create_access_token(
        data={
            "sub": user_data.username,
            "user_id": str(user.id),  # Phase 2.5: Add user_id for faster lookups
            "scopes": user.scopes
        }, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@app.post("/api/auth/login", response_model=TokenResponse)
def api_auth_login(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    JSON-based login endpoint used by the frontend bridge.
    Mirrors the behaviour of /signin but under the /api namespace.
    """
    return signin(user_data, db)

# OAuth2 Token Endpoint (RFC 6749 compliant)
@app.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 compliant token endpoint"""
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Phase 2.5: Include user_id in token payload
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": str(user.id),  # Phase 2.5: Add user_id for faster lookups
            "scopes": user.scopes
        }, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: str = Depends(get_current_user_oauth2), db: Session = Depends(get_db)):
    """Get current user information (requires read scope)"""
    user = db.query(User).filter(User.username == current_user).first()
    return UserResponse(
        username=user.username,
        email=user.email,
        created_at=user.created_at.isoformat()
    )

@app.get("/admin/users")
def get_all_users(current_user: dict = Depends(require_scope("admin")), db: Session = Depends(get_db)):
    """Get all users (requires admin scope)"""
    users = db.query(User).all()
    return {
        "users": [
            {
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat(),
                "is_active": user.is_active,
                "scopes": user.scopes
            }
            for user in users
        ]
    }

@app.post("/admin/users/{username}/scopes")
def update_user_scopes(
    username: str,
    scopes: list[str],
    current_user: dict = Depends(require_scope("admin")),
    db: Session = Depends(get_db)
):
    """Update user scopes (requires admin scope)"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.scopes = scopes
    db.commit()
    return {"message": f"Scopes updated for user {username}", "scopes": scopes}

# ============================================================================
# TOOL FACTORY FUNCTIONS
# ============================================================================

def format_structured_tool_response(
    status: str,
    context: str,
    next_action: str = None,
    quality_reason: str = None,
    citations: dict = None,
    metadata: dict = None
) -> str:
    """
    Format tool response as structured JSON string.
    
    Args:
        status: One of "final_answer_required", "next_action_required", "no_results", "error"
        context: The actual content/context for the agent
        next_action: Optional tool name to call next (if status is "next_action_required")
        quality_reason: Optional quality assessment reason
        citations: Optional citation data dictionary
        metadata: Optional tool-specific metadata
    
    Returns:
        JSON string that can be parsed by agent or fallback to natural language if serialization fails
    """
    try:
        response_dict = {
            "status": status,
            "context": context,
            "next_action": next_action,
            "quality_reason": quality_reason,
            "citations": citations,
            "metadata": metadata
        }
        # Remove None values to keep JSON clean
        response_dict = {k: v for k, v in response_dict.items() if v is not None}
        
        # Serialize to JSON
        json_str = json.dumps(response_dict, ensure_ascii=False)
        
        # Add explicit markers so agent recognizes JSON
        return (
            "===JSON_RESPONSE_START===\n"
            f"{json_str}\n"
            "===JSON_RESPONSE_END===\n\n"
            "You MUST parse this JSON. Extract 'status' and 'next_action' fields:\n"
            "- 'status' field tells you what to do next\n"
            "- 'next_action' field (if present) tells you which tool to call\n"
            "- 'context' field contains the information to use in your answer"
        )
    except Exception as e:
        # Fallback to natural language format if JSON serialization fails
        print(f"[WARN] Failed to serialize structured response: {e}, using fallback format")
        fallback = f"OBSERVATION: {status.upper().replace('_', ' ')}\n{context}"
        if next_action:
            fallback += f"\nNEXT_STEP: Call {next_action} NOW"
        return fallback


class AgentState:
    """Tracks agent execution state to prevent loops and enforce stopping conditions"""
    def __init__(self):
        self.tools_called: Set[str] = set()  # Track unique tool calls
        self.tool_call_count: Dict[str, int] = {}  # Count per tool
        self.current_status: Optional[str] = None  # From last tool response
        self.next_action: Optional[str] = None  # From last tool response
        self.final_answer_provided: bool = False
        self.last_tool_output: Optional[str] = None
        self.next_action_violations: int = 0  # Track when agent ignores next_action
        self.max_calls_per_tool: Dict[str, int] = {
            "youtube_rag_tool_v2": 1,
            "google_search_tool_v2": 1,
            "youtube_rag_tool_general": 1
        }
    
    def can_call_tool(self, tool_name: str) -> bool:
        """Check if tool can be called based on history and limits"""
        count = self.tool_call_count.get(tool_name, 0)
        max_calls = self.max_calls_per_tool.get(tool_name, 1)
        return count < max_calls
    
    def record_tool_call(self, tool_name: str, output: str):
        """Record tool call and parse structured response"""
        self.tools_called.add(tool_name)
        self.tool_call_count[tool_name] = self.tool_call_count.get(tool_name, 0) + 1
        self.last_tool_output = output
        # Parse JSON from output
        parsed = parse_structured_tool_response(output)
        if parsed:
            self.current_status = parsed.get("status")
            self.next_action = parsed.get("next_action")
            # If status is final_answer_required, mark that we should stop
            if self.current_status == "final_answer_required":
                self.final_answer_provided = True
                print(f"[STATE] Final answer required detected from {tool_name}")
    
    def should_stop(self) -> bool:
        """Check if agent should stop based on current state"""
        return (self.final_answer_provided or 
                self.current_status == "final_answer_required")
    
    def validate_next_action(self) -> Tuple[bool, Optional[str]]:
        """
        Validate if next_action is valid (not already called).
        
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
            - is_valid: True if next_action can be called, False if already called
            - error_message: Error message if invalid, None if valid
        """
        if not self.next_action:
            return True, None  # No next_action set, so it's valid (no suggestion)
        
        # Check if the suggested next_action tool was already called
        if not self.can_call_tool(self.next_action):
            count = self.tool_call_count.get(self.next_action, 0)
            error_msg = (
                f"ERROR: Tool {self.next_action} was already called ({count} time(s)). "
                "STOP calling tools immediately. "
                "You have already called this tool and received results. "
                "Check your scratchpad - look at previous Observations. "
                "You MUST provide Final Answer now using ALL context from previous tool Observations. "
                "Do NOT call any more tools. Write 'Thought: I see an error that this tool was already called. I should provide Final Answer now using the context I have from previous tool calls.' "
                "Then write 'Final Answer: [your answer here]'"
            )
            return False, error_msg
        
        return True, None
    
    def get_expected_next_tool(self) -> Optional[str]:
        """
        Get the tool name from next_action if set.
        
        Returns:
            Tool name from next_action, or None if not set
        """
        return self.next_action


def parse_structured_tool_response(output: str) -> dict:
    """
    Parse structured JSON response from tool output.
    Handles both pure JSON and JSON wrapped in text.
    
    Args:
        output: Tool output string (may be JSON or natural language)
    
    Returns:
        Parsed dictionary with status, context, etc., or None if parsing fails
    """
    if not output or not isinstance(output, str):
        return None
    
    # Try to extract JSON between markers first (new format)
    import re
    marker_match = re.search(r'===JSON_RESPONSE_START===\s*(.*?)\s*===JSON_RESPONSE_END===', output, re.DOTALL)
    if marker_match:
        try:
            return json.loads(marker_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to parse as direct JSON (fallback for old format)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text (look for JSON-like patterns)
    # Common patterns: {...} or lines starting with { and ending with }
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fallback: Try to parse natural language markers (backward compatibility)
    result = {"status": None, "context": output, "next_action": None}
    
    if "FINAL_ANSWER_REQUIRED" in output or "final answer required" in output.lower():
        result["status"] = "final_answer_required"
    elif "NEXT_ACTION_REQUIRED" in output or "next action required" in output.lower():
        result["status"] = "next_action_required"
        # Try to extract next action
        if "google_search_tool_v2" in output:
            result["next_action"] = "google_search_tool_v2"
        elif "youtube_rag_tool_general" in output:
            result["next_action"] = "youtube_rag_tool_general"
    elif "No relevant information" in output or "No web search results" in output:
        result["status"] = "no_results"
    else:
        result["status"] = "error"
    
    return result


class ValidatedTool(BaseTool):
    """Tool wrapper that validates state before execution"""
    # Required by BaseTool - these ARE Pydantic fields
    name: str = ""
    description: str = ""
    
    def __init__(self, original_tool: BaseTool, agent_state: Any, tool_name: str):
        # Extract required BaseTool fields BEFORE calling super().__init__()
        name = original_tool.name
        description = original_tool.description
        
        # Initialize BaseTool with required fields as kwargs
        super().__init__(name=name, description=description)
        
        # Set custom attributes (NOT Pydantic fields) using object.__setattr__()
        # This bypasses Pydantic validation for these fields
        object.__setattr__(self, 'original_tool', original_tool)
        object.__setattr__(self, 'agent_state', agent_state)
        object.__setattr__(self, 'tool_name', tool_name)
        
        # Copy args_schema if it exists (can be set normally after init)
        if hasattr(original_tool, 'args_schema'):
            self.args_schema = original_tool.args_schema
    
    def _run(self, *args, **kwargs) -> str:
        """Execute tool with validation"""
        if not self.agent_state.can_call_tool(self.tool_name):
            # Return PLAIN TEXT error, not JSON, to force Final Answer
            # Make it very explicit what the agent should do
            return (
                f"ERROR: Tool {self.tool_name} already called. "
                "STOP calling tools immediately. "
                "You have already called this tool and received results. "
                "Check your scratchpad - look at previous Observations. "
                "You MUST provide Final Answer now using ALL context from previous tool Observations. "
                "Do NOT call any more tools. Write 'Thought: I see an error that this tool was already called. I should provide Final Answer now using the context I have from previous tool calls.' "
                "Then write 'Final Answer: [your answer here]'"
            )
        # Execute original tool
        result = self.original_tool.invoke(*args, **kwargs)
        # Record call in agent state (this parses next_action from result)
        self.agent_state.record_tool_call(self.tool_name, result)
        
        # Validate next_action after recording (check if suggested tool was already called)
        is_valid, error_msg = self.agent_state.validate_next_action()
        if not is_valid and error_msg:
            # next_action suggests a tool that was already called
            # Modify the result to tell agent to provide Final Answer instead
            parsed = parse_structured_tool_response(result)
            if parsed:
                # Replace next_action with None and update status/context
                parsed["next_action"] = None
                parsed["status"] = "final_answer_required"
                parsed["context"] = (
                    parsed.get("context", "") + 
                    f"\n\n{error_msg}\n\n"
                    "You should provide Final Answer now using the context you have."
                )
                # Reformat as structured response
                return format_structured_tool_response(
                    status=parsed["status"],
                    context=parsed["context"],
                    next_action=parsed.get("next_action"),
                    quality_reason=parsed.get("quality_reason"),
                    citations=parsed.get("citations"),
                    metadata=parsed.get("metadata")
                )
            else:
                # If we can't parse, append error message
                return result + "\n\n" + error_msg
        
        return result


def create_youtube_rag_tool_v2(index, namespace: str, video_id: str):
    """
    Factory function to create youtube_rag_tool_v2 for video-scoped conversations.
    
    Args:
        index: Pinecone index instance
        namespace: User namespace for data isolation
        video_id: Video ID to filter search results
    
    Returns:
        LangChain tool for video RAG search
    """
    @tool
    def youtube_rag_tool_v2(query_text: str) -> str:
        """
        PRIORITY TOOL: Use this FIRST for any question about video content.
        
        Returns relevant context from YouTube videos for the agent to synthesize into an answer.
        This tool searches the specific video associated with this conversation only.
        
        This tool retrieves and returns raw context from the video transcript - the agent will use
        this context to generate the final answer.
        
        Use this when user asks about video content, topics, or specific details from the video.
        Input should be the user's question about the video.
        
        Format example:
        Action: youtube_rag_tool_v2
        Action Input: "explain the main topic"
        
        Do NOT use function call syntax like youtube_rag_tool_v2("question").
        """
        # Build Pinecone search query with video_id filter
        query_dict = {
            "top_k": 20,
            "inputs": {"text": query_text},
            "filter": {"video_id": video_id}  # Filter to this video only
        }
        
        # Search with reranking
        results = index.search(
            namespace=namespace,
            query=query_dict,
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 8,  # Optimized for performance while maintaining quality
                "rank_fields": ["content"]
            }
        )
        
        # Process results (reuse existing post-processing logic)
        docs = []
        if results and 'result' in results and 'hits' in results['result']:
            for hit in results['result']['hits']:
                content = hit.get('fields', {}).get('content', '')
                score = hit.get('_score', 0.0)
                docs.append(Document(
                    page_content=content,
                    metadata={'_score': score, 'video_id': video_id}
                ))
        
        # Post-process (diversity, context length)
        final_docs = process_retrieval_results(
            docs,
            min_score=MIN_SIMILARITY_SCORE,
            max_per_video=MAX_CHUNKS_PER_VIDEO,
            max_context_length=MAX_CONTEXT_LENGTH
        )
        
        # Determine if query is time-sensitive
        is_time_sensitive = any(keyword in query_text.lower() for keyword in TIME_SENSITIVE_KEYWORDS)
        
        if not final_docs:
            return format_structured_tool_response(
                status="no_results",
                context="No relevant information found in video transcript for this specific video.\n\nRECOVERY OPTIONS:\n1. If this is a time-sensitive or current events question, call google_search_tool_v2 to get up-to-date information from the web.\n2. If you want to search across all videos (not just this one), consider that this tool is video-specific. You may need to rephrase the query or use a different approach.\n3. If web search is not available, provide Final Answer explaining that no relevant information was found in this video's transcript.",
                next_action="google_search_tool_v2"
            )
        
        # Assess context quality (hybrid approach)
        is_quality_sufficient, quality_reason = assess_context_quality(final_docs)
        context = format_docs(final_docs)
        
        # Log quality assessment for debugging
        print(f"[QUALITY] Context quality assessment: {quality_reason}")
        
        # Return structured JSON response
        if is_time_sensitive:
            return format_structured_tool_response(
                status="next_action_required",
                context=f"TIME_SENSITIVE query. Historical context from video {video_id}.\n\nVideo context:\n{context}",
                next_action="google_search_tool_v2",
                quality_reason=quality_reason,
                metadata={"video_id": video_id, "is_time_sensitive": True}
            )
        elif not is_quality_sufficient:
            # Low quality context - recommend web search
            return format_structured_tool_response(
                status="next_action_required",
                context=f"Context quality insufficient. {quality_reason}\n\nVideo context (may be limited):\n{context}",
                next_action="google_search_tool_v2",
                quality_reason=quality_reason,
                metadata={"video_id": video_id, "is_time_sensitive": False}
            )
        else:
            # Good quality context - can provide final answer
            return format_structured_tool_response(
                status="final_answer_required",
                context=f"Sufficient context from video {video_id}.\n\nIMPORTANT: Your Final Answer should:\n- Synthesize comprehensively (not just summarize)\n- Use ALL relevant context provided\n- Include specific details, examples, and key points\n- Be thorough and detailed for complete understanding\n\nVideo context:\n{context}",
                quality_reason=quality_reason,
                metadata={"video_id": video_id, "is_time_sensitive": False}
            )
    
    return youtube_rag_tool_v2


def create_google_search_tool_v2():
    """
    Factory function to create google_search_tool_v2 for web search.
    
    Uses GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.
    
    Returns:
        LangChain tool for Google web search
    """
    @tool
    def google_search_tool_v2(query_text: str) -> str:
        """
        PRIORITY TOOL for time-sensitive queries: Performs a Google Custom Search and returns the top results.
        
        CRITICAL: For TIME-SENSITIVE queries (contains "latest", "recent", "new", "current", "2024", "2025", "now", "today"):
        - Call this tool FIRST to get current/recent information from the web
        - Then optionally use youtube_rag_tool_general for additional video context
        - This ensures you get up-to-date information first
        
        Use this tool for:
        - Questions about recent developments, latest updates, current events
        - Time-sensitive information that requires current data
        - When video content might be outdated
        
        After calling this tool, you will receive "NEXT_ACTION_REQUIRED" - you can either:
        - Optionally call youtube_rag_tool_general for additional video context (recommended for comprehensive answers)
        - OR provide Final Answer immediately if web results are sufficient
        
        Format example:
        Action: google_search_tool_v2
        Action Input: "recent developments in AI space"
        
        Do NOT use function call syntax like google_search_tool_v2("query").
        """
        # Use consistent environment variable names (matches top of file)
        search_api_key = GOOGLE_SEARCH_API_KEY
        search_engine_id = GOOGLE_SEARCH_ENGINE_ID
        
        if not search_api_key or not search_engine_id:
            return format_structured_tool_response(
                status="error",
                context="Web search is not configured (missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID).\n\nRECOVERY: You cannot use web search. Provide Final Answer using available video context from youtube_rag_tool_v2 or youtube_rag_tool_general, or explain that current web information is unavailable. Do not attempt to call google_search_tool_v2 again.",
                next_action=None
            )
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": search_api_key,
            "cx": search_engine_id,
            "q": query_text,
            "num": 5
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            
            if "items" not in results:
                return format_structured_tool_response(
                    status="no_results",
                    context="No web search results found for this query.\n\nRECOVERY OPTIONS:\n1. Try rephrasing the query with different keywords and call google_search_tool_v2 again (if you haven't called it multiple times already).\n2. Use youtube_rag_tool_general to search video transcripts for related information.\n3. If both tools return no results, provide Final Answer explaining that no relevant information was found and suggest the user rephrase their question.",
                    next_action="youtube_rag_tool_general"
                )
            
            docs = []
            for item in results["items"][:5]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                docs.append(Document(
                    page_content=f"{title}\n{snippet}",
                    metadata={"source": link}
                ))
            
            context = format_docs(docs)
            
            # Check if query is time-sensitive
            is_time_sensitive = any(keyword in query_text.lower() for keyword in TIME_SENSITIVE_KEYWORDS)
            
            # Return structured JSON response
            # For time-sensitive queries, web results are sufficient - don't suggest YouTube tool
            # For non-time-sensitive queries, optionally suggest YouTube tool for additional context
            if is_time_sensitive:
                # Time-sensitive query: web results are current and sufficient
                return format_structured_tool_response(
                    status="final_answer_required",
                    context=f"Web search results received. You have current information from the web.\n\nIMPORTANT: Your Final Answer should:\n- Synthesize comprehensively from web search results\n- Use ALL relevant context provided\n- Include specific details, examples, and key points\n- Be thorough and detailed for complete understanding\n\nWEB_SEARCH_RESULTS:\n{context}",
                    next_action=None,
                    metadata={"source": "google_search", "result_count": len(docs), "is_time_sensitive": True}
                )
            else:
                # Non-time-sensitive query: optionally suggest YouTube tool for additional context
                return format_structured_tool_response(
                    status="next_action_required",
                    context=f"Web search results received. You have current information from the web.\n\nOPTIONAL: If you want additional context from video summaries, you may call youtube_rag_tool_general now.\nOR: If web results are sufficient, provide Final Answer now by thinking 'I now know the final answer' and synthesizing from web results.\n\nIMPORTANT: Your Final Answer should synthesize comprehensively from web search results (and video context if you also call youtube_rag_tool_general).\n\nWEB_SEARCH_RESULTS:\n{context}",
                    next_action="youtube_rag_tool_general",  # Optional, agent can choose to skip
                    metadata={"source": "google_search", "result_count": len(docs), "is_time_sensitive": False}
                )
        
        except Exception as e:
            error_msg = str(e)
            # Determine if error is recoverable
            if "timeout" in error_msg.lower() or "temporarily" in error_msg.lower():
                recovery = "This appears to be a temporary error. You may try calling google_search_tool_v2 again, or use youtube_rag_tool_general for alternative information."
                next_action = "youtube_rag_tool_general"
            else:
                recovery = "This is a permanent error (e.g., API configuration issue). You should provide Final Answer using available video context from youtube_rag_tool_v2 or youtube_rag_tool_general, or explain that web search is unavailable."
                next_action = None
            
            return format_structured_tool_response(
                status="error",
                context=f"Web search failed: {error_msg}\n\nRECOVERY: {recovery}",
                next_action=next_action
            )
    
    return google_search_tool_v2


def create_youtube_rag_tool_general(index, namespace: str):
    """
    Factory function to create youtube_rag_tool_general for general (cross-video) conversations.
    
    Args:
        index: Pinecone index instance
        namespace: User namespace for data isolation
    
    Returns:
        LangChain tool for general RAG search with catalog shortlisting
    """
    @tool
    def youtube_rag_tool_general(query_text: str) -> str:
        """
        Use this tool for questions about video content from your processed YouTube videos.
        
        ⚠️ CRITICAL WARNING: DO NOT USE THIS TOOL FIRST FOR TIME-SENSITIVE QUERIES!
        - If the query contains ANY of these words: "latest", "recent", "new", "current", "2024", "2025", "now", "today", "update", "what is new"
        - You MUST call google_search_tool_v2 FIRST to get current information
        - This tool searches your processed videos which may be outdated (e.g., from 2023)
        - Only use this tool AFTER google_search_tool_v2 if you want additional historical video context
        - For GENERAL questions (NOT time-sensitive): You can use this tool first
        
        Returns relevant context from YouTube videos for the agent to synthesize into an answer.
        This tool searches across ALL videos processed by the user (including videos from previous
        conversations). It uses catalog-based shortlisting to find the most relevant videos first,
        then retrieves context from those videos.
        
        This tool retrieves and returns raw context from multiple videos - the agent will use this
        context to generate the final answer.
        
        Use this when user asks about video content, topics, or specific details from their videos.
        Input should be the user's question about the videos.
        
        Format example:
        Action: youtube_rag_tool_general
        Action Input: "explain machine learning concepts"
        
        Do NOT use function call syntax like youtube_rag_tool_general("question").
        """
        # Use General tab retrieval with catalog shortlisting
        docs, shortlisted_video_ids = retrieve_for_general_tab(
            chunks_index=index,
            namespace=namespace,
            query=query_text
        )
        
        # Determine if query is time-sensitive
        is_time_sensitive = any(keyword in query_text.lower() for keyword in TIME_SENSITIVE_KEYWORDS)
        
        if not docs:
            return format_structured_tool_response(
                status="no_results",
                context="No relevant information found in processed videos.\n\nRECOVERY OPTIONS:\n1. If this is a time-sensitive or current events question, call google_search_tool_v2 to get up-to-date information from the web.\n2. Try rephrasing the query with different keywords - the current query may not match any video content.\n3. If web search is not available or also returns no results, provide Final Answer explaining that no relevant information was found and suggest the user rephrase their question.",
                next_action="google_search_tool_v2"
            )
        
        # Assess context quality (hybrid approach)
        is_quality_sufficient, quality_reason = assess_context_quality(docs)
        context = format_docs(docs)
        
        # Get citations for response (create new session to avoid stale session issues)
        citations = format_citations_from_docs(docs, lambda: SessionLocal())
        citation_text = ""
        if citations:
            citation_text = "\n\nSources:\n"
            for vid_id, info in citations.items():
                citation_text += f"- {info['title']}: {info['url']}\n"
        
        # Log quality assessment for debugging
        print(f"[QUALITY] Context quality assessment: {quality_reason}")
        
        # Return structured JSON response with citations
        if is_time_sensitive:
            return format_structured_tool_response(
                status="next_action_required",
                context=f"TIME_SENSITIVE query. Historical context from {len(shortlisted_video_ids)} videos.\n\nVideo context:\n{context}{citation_text}",
                next_action="google_search_tool_v2",
                quality_reason=quality_reason,
                citations=citations,
                metadata={"video_count": len(shortlisted_video_ids), "is_time_sensitive": True}
            )
        elif not is_quality_sufficient:
            # Low quality context - recommend web search
            return format_structured_tool_response(
                status="next_action_required",
                context=f"Context quality insufficient. {quality_reason}\n\nVideo context (may be limited):\n{context}{citation_text}",
                next_action="google_search_tool_v2",
                quality_reason=quality_reason,
                citations=citations,
                metadata={"video_count": len(shortlisted_video_ids), "is_time_sensitive": False}
            )
        else:
            # Good quality context - can provide final answer
            return format_structured_tool_response(
                status="final_answer_required",
                context=f"Sufficient context from {len(shortlisted_video_ids)} videos.\n\nIMPORTANT: Your Final Answer should:\n- Synthesize comprehensively (not just summarize)\n- Use ALL relevant context from all videos provided\n- Include specific details, examples, and key points\n- Be thorough and detailed for complete understanding\n\nVideo context:\n{context}{citation_text}",
                quality_reason=quality_reason,
                citations=citations,
                metadata={"video_count": len(shortlisted_video_ids), "is_time_sensitive": False}
            )
    
    return youtube_rag_tool_general


# ============================================================================
# SCHEMA 2 ENDPOINTS - Video-Scoped Conversations
# ============================================================================

@app.post("/api/v2/chat/video", response_model=ConversationResponse)
async def chat_with_video_v2(
    user_input: ConversationInput,
    current_user: dict = Depends(require_scope("write")),
    db: Session = Depends(get_db)
):
    """
    Chat with a specific YouTube video using Schema 2 (video-scoped conversations).
    
    This endpoint:
    1. Creates or retrieves conversation for (user, video)
    2. Embeds video if not already embedded
    3. Runs RAG query filtered to this video
    4. Stores user message and assistant reply
    5. Updates conversation metadata
    
    Replaces legacy / endpoint (session-based) with video-first approach.
    Requires 'write' scope.
    """
    try:
        # Check if Pinecone client is initialized
        if pc is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
            )
        
        # STEP 1: Get current user
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # STEP 2: Resolve video_id and validate input
        import uuid as uuid_lib
        
        if user_input.conversation_id:
            # Phase 5.13: Resume existing conversation with validation
            try:
                # Debug logging (Phase 1.3): Log conversation retrieval
                conv_uuid = uuid_lib.UUID(user_input.conversation_id)
                if DEBUG_LOGGING:
                    log_data = {
                        "event": "conversation_retrieval",
                        "conversation_id": str(conv_uuid),
                        "user_id": str(user.id),
                        "username": user.username,
                        "endpoint": "/api/v2/chat/video",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.debug(f"[CHAT] Retrieving conversation: {json.dumps(log_data)}")
                
                # Phase 5.13: Use utility function for ownership verification
                is_owner, error_msg = verify_conversation_ownership(db, conv_uuid, user.id)
                
                if not is_owner:
                    # Phase 3.13: Enhanced error message for invalid conversation_id
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Conversation {user_input.conversation_id} not found or you don't have access. "
                               f"Please verify the conversation ID or start a new conversation."
                    )
                
                # Get conversation details
                retrieval_start = time.time()
                conversation = get_conversation_by_id(
                    db, 
                    conv_uuid, 
                    user_id=user.id
                )
                retrieval_time = (time.time() - retrieval_start) * 1000
                
                if DEBUG_LOGGING:
                    log_data["conversation_found"] = conversation is not None
                    log_data["retrieval_time_ms"] = round(retrieval_time, 2)
                    if conversation:
                        log_data["video_id"] = conversation.video_id
                        log_data["video_title"] = conversation.video_title[:100]
                    logger.debug(f"[CHAT] Conversation retrieval result: {json.dumps(log_data)}")
                
                if not conversation:
                    # This should not happen if ownership check passed, but defensive check
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Conversation {user_input.conversation_id} not found. "
                               f"Please verify the conversation ID or start a new conversation."
                    )
                video_id = conversation.video_id
                video_url = conversation.video_url
                video_title = conversation.video_title
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation_id format")
        elif user_input.video_url or user_input.video_id:
            # Start new conversation with video
            if user_input.video_url:
                video_id = get_youtube_video_id(str(user_input.video_url))
            else:
                video_id = user_input.video_id
            
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid video URL or ID")
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_title = "YouTube Video"  # Will be updated after embedding
            
            # Create or get conversation
            conversation = get_or_create_conversation(
                db=db,
                user_id=user.id,
                video_id=video_id,
                video_url=video_url,
                video_title=video_title
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide video_url, video_id, or conversation_id"
            )
        
        # Validate and sanitize query input
        user_input.query = user_input.query.strip()
        if not user_input.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if len(user_input.query) > MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
            )
        
        # STEP 3: Embed video (if not already embedded)
        index = get_shared_index()
        namespace = get_user_namespace(user.username)
        
        # Use extracted embedding function (also creates catalog entry)
        video_id, video_title = embed_video_if_needed(
            db=db,
            video_id=video_id,
            video_url=video_url,
            user_id=user.id,
            index=index,
            namespace=namespace
        )
        
        # Title will be updated after successful message insertion (see below)
        
        # STEP 4: Build agent context from conversation history
        recent_messages = get_recent_messages_for_context(
            db, conversation.conversation_id, limit=10
        )
        
        # Convert to LangChain message format
        history = []
        for msg in recent_messages:
            if msg.role == 'user':
                history.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                history.append(AIMessage(content=msg.content))
        
        # STEP 5: Create AgentState FIRST (before tool creation and wrapping)
        # Use shared AgentState class
        agent_state = AgentState()
        
        # STEP 6: Create agent tools using factory functions
        youtube_rag_tool_v2 = create_youtube_rag_tool_v2(index, namespace, video_id)
        google_search_tool_v2 = create_google_search_tool_v2()
        
        # STEP 7: Wrap tools with ValidatedTool for programmatic validation
        wrapped_youtube_tool = ValidatedTool(
            original_tool=youtube_rag_tool_v2,
            agent_state=agent_state,
            tool_name="youtube_rag_tool_v2"
        )
        wrapped_google_tool = ValidatedTool(
            original_tool=google_search_tool_v2,
            agent_state=agent_state,
            tool_name="google_search_tool_v2"
        )
        
        # Use wrapped tools
        tools = [wrapped_youtube_tool, wrapped_google_tool]
        print(f"[DEBUG] Created {len(tools)} wrapped tools: {[tool.name if hasattr(tool, 'name') else type(tool).__name__ for tool in tools]}")
        
        # Create enhanced callback to track tool execution with AgentState
        class EnhancedToolTrackingCallback(BaseCallbackHandler):
            """Enhanced callback that tracks tool calls using AgentState and enforces stopping"""
            def __init__(self, agent_state: AgentState):
                super().__init__()
                self.agent_state = agent_state
                self.tool_calls = []
                self.tool_results = []
                self.final_answer_detected = False
                self.last_llm_output = ""
                self.should_stop = False
            
            def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
                """Called when a tool starts execution - check for duplicate calls and next_action violations"""
                tool_name = serialized.get('name', 'unknown')
                
                # Check if tool can be called (proactive loop prevention)
                if not self.agent_state.can_call_tool(tool_name):
                    count = self.agent_state.tool_call_count.get(tool_name, 0)
                    print(f"[CALLBACK] WARNING: Attempting to call {tool_name} again (already called {count} times). This may indicate a loop.")
                
                # Monitor next_action violations: check if agent called a tool that doesn't match next_action
                expected_tool = self.agent_state.get_expected_next_tool()
                if expected_tool and tool_name != expected_tool:
                    # Agent ignored next_action suggestion
                    self.agent_state.next_action_violations += 1
                    print(f"[CALLBACK] WARNING: Agent called {tool_name} but next_action suggested {expected_tool}. This is a next_action violation.")
            
            def on_tool_end(self, output: str, **kwargs) -> None:
                """Called when a tool finishes execution"""
                tool_name = kwargs.get('name', 'unknown')
                self.tool_calls.append(tool_name)
                
                if isinstance(output, str):
                    self.tool_results.append(output)
                    
                    # NOTE: Don't call agent_state.record_tool_call() here
                    # ValidatedTool already records it. Just parse for monitoring/logging.
                    parsed = parse_structured_tool_response(output)
                    if parsed:
                        status = parsed.get("status")
                        if status == "final_answer_required":
                            print(f"[CALLBACK] {tool_name} returned final_answer_required - agent should provide final answer")
                        elif status == "next_action_required":
                            next_action = parsed.get("next_action")
                            print(f"[CALLBACK] {tool_name} returned next_action_required: {next_action}")
                    
                    # Check if we should stop (read state, don't modify)
                    if self.agent_state.should_stop():
                        self.should_stop = True
                        print(f"[CALLBACK] Stopping condition detected: status={self.agent_state.current_status}")
                    
                    # Check for duplicate tool calls (reactive detection - read only)
                    if not self.agent_state.can_call_tool(tool_name):
                        count = self.agent_state.tool_call_count.get(tool_name, 0)
                        print(f"[CALLBACK] WARNING: Tool {tool_name} called {count} times (exceeds limit). Loop detected!")
                    
                    # Log for monitoring (but don't duplicate state recording)
                    print(f"[CALLBACK] Tool {tool_name} completed")
            
            def on_llm_end(self, response: LLMResult, **kwargs) -> None:
                """Called when LLM finishes generating output"""
                if response.generations and len(response.generations) > 0:
                    generation = response.generations[0]
                    if len(generation) > 0:
                        output_text = generation[0].text
                        self.last_llm_output = output_text
                        # Detect final answer in output
                        if "Final Answer:" in output_text or "final answer:" in output_text.lower():
                            self.final_answer_detected = True
                            self.agent_state.final_answer_provided = True
                            print(f"[CALLBACK] Final answer detected in LLM output - should stop soon")
        
        tool_tracking_callback = EnhancedToolTrackingCallback(agent_state)
        
        # Create agent executor with enhanced prompt and callbacks
        # Use cached REACT_AGENT_PROMPT which includes tool instructions
        agent = create_react_agent(model, tools, REACT_AGENT_PROMPT)
        
        # Custom parsing error handler that detects final answers
        def custom_parse_error_handler(error: Exception) -> str:
            """Handle parsing errors gracefully, especially after final answers"""
            error_str = str(error)
            # If we detect final answer in error, return a stop signal
            if "Final Answer:" in error_str or "final answer:" in error_str.lower():
                print(f"[PARSER] Detected final answer in parsing error - stopping")
                agent_state.final_answer_provided = True
                return "FINAL_ANSWER_PROVIDED - STOP EXECUTION NOW"
            # Otherwise, return standard error message
            return f"Invalid format. Remember: After providing 'Final Answer:', you must stop. Error: {error_str}"
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=7,  # Increased to allow proper tool chaining and answer synthesis
            handle_parsing_errors=custom_parse_error_handler,
            callbacks=[tool_tracking_callback],  # Add callback for monitoring
            early_stopping_method="generate"  # Stop early if agent generates final answer
        )
        
        # Invoke agent with timeout protection and early stopping monitoring
        # Note: time and asyncio are already imported at module level
        start_time = time.time()
        timeout_seconds = 60.0  # 60 second timeout
        
        try:
            # Run agent with timeout
            # Note: LangChain 0.2.x doesn't support custom should_continue, so we monitor via callback
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    agent_executor.invoke,
                    {
                        "input": user_input.query,
                        "chat_history": history
                    }
                ),
                timeout=timeout_seconds
            )
            
            execution_time = time.time() - start_time
            print(f"[TIMING] Agent execution completed in {execution_time:.2f} seconds")
            
            # Check if early stopping was triggered
            if tool_tracking_callback.should_stop:
                print(f"[STOP] Early stopping was triggered: status={agent_state.current_status}")
            
            # Check for looping behavior (consolidated)
            if len(tool_tracking_callback.tool_calls) > 0:
                tool_count = {}
                for tool_name in tool_tracking_callback.tool_calls:
                    tool_count[tool_name] = tool_count.get(tool_name, 0) + 1
                
                # Check for excessive calls to same tool
                for tool_name, count in tool_count.items():
                    if count > 1:
                        print(f"[WARNING] Tool {tool_name} called {count} times (should be max 1)")
                        print(f"[WARNING] Tool call sequence: {' -> '.join(tool_tracking_callback.tool_calls)}")
                
                # Log tool usage summary
                print(f"[MONITOR] Tools used: {', '.join(set(tool_tracking_callback.tool_calls))}")
                print(f"[MONITOR] Total tool calls: {len(tool_tracking_callback.tool_calls)}")
            
            # Log next_action violation metrics
            if agent_state.next_action_violations > 0:
                print(f"[METRICS] next_action violations: {agent_state.next_action_violations}")
            else:
                print(f"[METRICS] next_action violations: 0 (agent followed all suggestions)")
            
            answer = result.get("output", "I couldn't generate an answer.")
            
            # Extract final answer if it contains the marker (prevent repetition)
            # Also handle structured responses from tools if they appear in the answer
            if "Final Answer:" in answer:
                # Extract only the final answer portion
                final_answer_idx = answer.find("Final Answer:")
                if final_answer_idx != -1:
                    answer = answer[final_answer_idx + len("Final Answer:"):].strip()
                    # Remove any trailing text that looks like continuation or repetition
                    # Split by common continuation patterns and take first part
                    for separator in ["\n\nThought:", "\n\nAction:", "\nThought:", "\nAction:"]:
                        if separator in answer:
                            answer = answer.split(separator)[0].strip()
                            break
                    print(f"[INFO] Extracted final answer, removed trailing text")
            
            # Validate answer is not empty after extraction
            if not answer or len(answer.strip()) < 10:
                print(f"[WARN] Answer too short after extraction, using full output")
                answer = result.get("output", "I couldn't generate an answer.")
        
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"[ERROR] Agent execution timed out after {execution_time:.2f} seconds")
            print(f"[ERROR] Query that timed out: {user_input.query[:100]}...")  # Log partial query for debugging
            raise HTTPException(
                status_code=504,
                detail=f"Agent execution timed out after {timeout_seconds} seconds. Please try a simpler query or contact support."
            )
        
        # STEP 7: Store conversation turn in database
        user_msg, assistant_msg = insert_conversation_turn(
            db=db,
            conversation_id=conversation.conversation_id,
            user_id=user.id,
            video_id=video_id,
            user_message=user_input.query,
            assistant_message=answer
        )
        
        # STEP 8: Update conversation title if it was generic (after successful message insertion)
        # This is deferred until after successful insert to maintain transaction consistency
        if conversation.video_title == "YouTube Video" and video_title != "YouTube Video":
            try:
                conversation.video_title = video_title
                db.commit()
                db.refresh(conversation)
                print(f"[INFO] Updated conversation title to: {video_title}")
            except Exception as e:
                print(f"[WARN] Failed to update conversation title: {e}")
                db.rollback()
                # Continue - title update is non-critical, main transaction already succeeded
        
        # STEP 9: Return response
        return ConversationResponse(
            conversation_id=str(conversation.conversation_id),
            video_id=video_id,
            video_title=conversation.video_title,
            user_message=user_input.query,
            assistant_message=answer,
            message_index=assistant_msg.message_index,
            created_at=assistant_msg.created_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/v2/chat/general")
async def chat_general_tab(
    user_input: ConversationInput,
    current_user: dict = Depends(require_scope("write")),
    db: Session = Depends(get_db)
):
    """
    General tab conversation endpoint (cross-video queries).
    
    Features:
    - Uses catalog-based video shortlisting for retrieval
    - No specific video filter (searches across all user's videos)
    - Stores conversation with video_id = "GENERAL"
    - Includes citations with video titles
    - Web search integration for time-sensitive queries
    
    Requires 'write' scope.
    """
    try:
        # Check if Pinecone client is initialized
        if pc is None:
            raise HTTPException(
                status_code=503,
                detail="Pinecone service is not available"
            )
        
        # STEP 1: Get current user
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # STEP 2: Get or create General conversation
        import uuid as uuid_lib
        
        if user_input.conversation_id:
            # Phase 5.14: Resume existing General conversation with validation
            try:
                # Debug logging (Phase 1.3): Log conversation retrieval
                conv_uuid = uuid_lib.UUID(user_input.conversation_id)
                if DEBUG_LOGGING:
                    log_data = {
                        "event": "conversation_retrieval",
                        "conversation_id": str(conv_uuid),
                        "user_id": str(user.id),
                        "username": user.username,
                        "endpoint": "/api/v2/chat/general",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.debug(f"[CHAT] Retrieving conversation: {json.dumps(log_data)}")
                
                # Phase 5.14: Use utility function for ownership verification
                is_owner, error_msg = verify_conversation_ownership(db, conv_uuid, user.id)
                
                if not is_owner:
                    # Phase 3.14: Enhanced error message for General conversation
                    raise HTTPException(
                        status_code=404, 
                        detail=f"General conversation {user_input.conversation_id} not found or you don't have access."
                    )
                
                # Get conversation details
                retrieval_start = time.time()
                conversation = get_conversation_by_id(
                    db,
                    conv_uuid,
                    user_id=user.id
                )
                retrieval_time = (time.time() - retrieval_start) * 1000
                
                if DEBUG_LOGGING:
                    log_data["conversation_found"] = conversation is not None
                    log_data["retrieval_time_ms"] = round(retrieval_time, 2)
                    if conversation:
                        log_data["video_id"] = conversation.video_id
                    logger.debug(f"[CHAT] Conversation retrieval result: {json.dumps(log_data)}")
                
                if not conversation or conversation.video_id != "GENERAL":
                    # Phase 3.14: Enhanced error message for General conversation
                    if not conversation:
                        error_msg = f"General conversation {user_input.conversation_id} not found or you don't have access."
                    else:
                        error_msg = f"Conversation {user_input.conversation_id} is not a General conversation (video_id: {conversation.video_id})."
                    raise HTTPException(status_code=404, detail=error_msg)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation_id format")
        else:
            # Create or get General conversation
            conversation = get_or_create_conversation(
                db=db,
                user_id=user.id,
                video_id="GENERAL",
                video_url="https://www.youtube.com/general",
                video_title="General Questions"
            )
        
        # Validate query
        user_input.query = user_input.query.strip()
        if not user_input.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if len(user_input.query) > MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
            )
        
        # STEP 3: Get user namespace and indexes
        index = get_shared_index()
        namespace = get_user_namespace(user.username)
        
        # STEP 4: Build agent context from conversation history
        recent_messages = get_recent_messages_for_context(
            db, conversation.conversation_id, limit=10
        )
        
        history = []
        for msg in recent_messages:
            if msg.role == 'user':
                history.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                history.append(AIMessage(content=msg.content))
        
        # STEP 5: Create AgentState FIRST (before tool creation and wrapping)
        # Use shared AgentState class
        agent_state = AgentState()
        
        # STEP 6: Create General tab tools using factory functions
        # Create tools - IMPORTANT: Order matters for time-sensitive queries
        # google_search_tool_v2 should be first to prioritize web search for "recent" queries
        google_search_tool_v2 = create_google_search_tool_v2()
        youtube_rag_tool_general = create_youtube_rag_tool_general(index, namespace)
        
        # STEP 7: Wrap tools with ValidatedTool for programmatic validation
        wrapped_google_tool = ValidatedTool(
            original_tool=google_search_tool_v2,
            agent_state=agent_state,
            tool_name="google_search_tool_v2"
        )
        wrapped_youtube_tool = ValidatedTool(
            original_tool=youtube_rag_tool_general,
            agent_state=agent_state,
            tool_name="youtube_rag_tool_general"
        )
        
        # Use wrapped tools (order: google_search first for time-sensitive queries)
        tools = [wrapped_google_tool, wrapped_youtube_tool]
        print(f"[DEBUG] Created {len(tools)} wrapped tools: {[tool.name if hasattr(tool, 'name') else type(tool).__name__ for tool in tools]}")
        
        # Create enhanced callback (same as video endpoint)
        class EnhancedToolTrackingCallback(BaseCallbackHandler):
            """Enhanced callback that tracks tool calls using AgentState and enforces stopping"""
            def __init__(self, agent_state: AgentState):
                super().__init__()
                self.agent_state = agent_state
                self.tool_calls = []
                self.tool_results = []
                self.final_answer_detected = False
                self.last_llm_output = ""
                self.should_stop = False
            
            def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
                """Called when a tool starts execution - check for duplicate calls and next_action violations"""
                tool_name = serialized.get('name', 'unknown')
                if not self.agent_state.can_call_tool(tool_name):
                    count = self.agent_state.tool_call_count.get(tool_name, 0)
                    print(f"[CALLBACK] WARNING: Attempting to call {tool_name} again (already called {count} times). This may indicate a loop.")
                
                # Monitor next_action violations: check if agent called a tool that doesn't match next_action
                expected_tool = self.agent_state.get_expected_next_tool()
                if expected_tool and tool_name != expected_tool:
                    # Agent ignored next_action suggestion
                    self.agent_state.next_action_violations += 1
                    print(f"[CALLBACK] WARNING: Agent called {tool_name} but next_action suggested {expected_tool}. This is a next_action violation.")
            
            def on_tool_end(self, output: str, **kwargs) -> None:
                """Called when a tool finishes execution"""
                tool_name = kwargs.get('name', 'unknown')
                self.tool_calls.append(tool_name)
                
                if isinstance(output, str):
                    self.tool_results.append(output)
                    
                    # NOTE: Don't call agent_state.record_tool_call() here
                    # ValidatedTool already records it. Just parse for monitoring/logging.
                    parsed = parse_structured_tool_response(output)
                    if parsed:
                        status = parsed.get("status")
                        if status == "final_answer_required":
                            print(f"[CALLBACK] {tool_name} returned final_answer_required - agent should provide final answer")
                        elif status == "next_action_required":
                            next_action = parsed.get("next_action")
                            print(f"[CALLBACK] {tool_name} returned next_action_required: {next_action}")
                    
                    # Check if we should stop (read state, don't modify)
                    if self.agent_state.should_stop():
                        self.should_stop = True
                        print(f"[CALLBACK] Stopping condition detected: status={self.agent_state.current_status}")
                    
                    # Check for duplicate tool calls (reactive detection - read only)
                    if not self.agent_state.can_call_tool(tool_name):
                        count = self.agent_state.tool_call_count.get(tool_name, 0)
                        print(f"[CALLBACK] WARNING: Tool {tool_name} called {count} times (exceeds limit). Loop detected!")
                    
                    # Log for monitoring (but don't duplicate state recording)
                    print(f"[CALLBACK] Tool {tool_name} completed")
            
            def on_llm_end(self, response: LLMResult, **kwargs) -> None:
                """Called when LLM finishes generating output"""
                if response.generations and len(response.generations) > 0:
                    generation = response.generations[0]
                    if len(generation) > 0:
                        output_text = generation[0].text
                        self.last_llm_output = output_text
                        if "Final Answer:" in output_text or "final answer:" in output_text.lower():
                            self.final_answer_detected = True
                            self.agent_state.final_answer_provided = True
                            print(f"[CALLBACK] Final answer detected in LLM output - should stop soon")
        
        tool_tracking_callback = EnhancedToolTrackingCallback(agent_state)
        
        # Custom parsing error handler (same as video endpoint)
        def custom_parse_error_handler(error: Exception) -> str:
            """Handle parsing errors gracefully, especially after final answers"""
            error_str = str(error)
            if "Final Answer:" in error_str or "final answer:" in error_str.lower():
                print(f"[PARSER] Detected final answer in parsing error - stopping")
                agent_state.final_answer_provided = True
                return "FINAL_ANSWER_PROVIDED - STOP EXECUTION NOW"
            return f"Invalid format. Remember: After providing 'Final Answer:', you must stop. Error: {error_str}"
        
        # Create agent executor
        agent = create_react_agent(model, tools, REACT_AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=7,  # Increased to allow proper tool chaining and answer synthesis
            handle_parsing_errors=custom_parse_error_handler,
            callbacks=[tool_tracking_callback],
            early_stopping_method="generate"  # Stop early if agent generates final answer
        )
        
        # Run agent with timeout
        # Note: time and asyncio are already imported at module level
        start_time = time.time()
        timeout_seconds = 60.0
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    agent_executor.invoke,
                    {
                        "input": user_input.query,
                        "chat_history": history
                    }
                ),
                timeout=timeout_seconds
            )
            
            execution_time = time.time() - start_time
            print(f"[TIMING] General tab execution completed in {execution_time:.2f} seconds")
            
            # Check if early stopping was triggered
            if tool_tracking_callback.should_stop:
                print(f"[STOP] Early stopping was triggered: status={agent_state.current_status}")
            
            # Check for looping behavior (consolidated)
            if len(tool_tracking_callback.tool_calls) > 0:
                tool_count = {}
                for tool_name in tool_tracking_callback.tool_calls:
                    tool_count[tool_name] = tool_count.get(tool_name, 0) + 1
                
                # Check for excessive calls to same tool
                for tool_name, count in tool_count.items():
                    if count > 1:
                        print(f"[WARNING] Tool {tool_name} called {count} times (should be max 1)")
                        print(f"[WARNING] Tool call sequence: {' -> '.join(tool_tracking_callback.tool_calls)}")
                
                # Log tool usage summary
                print(f"[MONITOR] Tools used: {', '.join(set(tool_tracking_callback.tool_calls))}")
                print(f"[MONITOR] Total tool calls: {len(tool_tracking_callback.tool_calls)}")
            
            # Log next_action violation metrics
            if agent_state.next_action_violations > 0:
                print(f"[METRICS] next_action violations: {agent_state.next_action_violations}")
            else:
                print(f"[METRICS] next_action violations: 0 (agent followed all suggestions)")
            
            answer = result.get("output", "I couldn't generate an answer.")
            
            # Extract final answer if it contains the marker (prevent repetition)
            # Also handle structured responses from tools if they appear in the answer
            if "Final Answer:" in answer:
                # Extract only the final answer portion
                final_answer_idx = answer.find("Final Answer:")
                if final_answer_idx != -1:
                    answer = answer[final_answer_idx + len("Final Answer:"):].strip()
                    # Remove any trailing text that looks like continuation or repetition
                    for separator in ["\n\nThought:", "\n\nAction:", "\nThought:", "\nAction:"]:
                        if separator in answer:
                            answer = answer.split(separator)[0].strip()
                            break
                    print(f"[INFO] Extracted final answer, removed trailing text")
            
            # Validate answer is not empty after extraction
            if not answer or len(answer.strip()) < 10:
                print(f"[WARN] Answer too short after extraction, using full output")
                answer = result.get("output", "I couldn't generate an answer.")
        
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"[ERROR] General tab execution timed out after {execution_time:.2f} seconds")
            print(f"[ERROR] Query that timed out: {user_input.query[:100]}...")  # Log partial query for debugging
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout_seconds} seconds"
            )
        
        # STEP 7: Store conversation turn
        user_msg, assistant_msg = insert_conversation_turn(
            db=db,
            conversation_id=conversation.conversation_id,
            user_id=user.id,
            video_id="GENERAL",
            user_message=user_input.query,
            assistant_message=answer
        )
        
        # STEP 8: Return response
        return ConversationResponse(
            conversation_id=str(conversation.conversation_id),
            video_id="GENERAL",
            video_title=conversation.video_title,
            user_message=user_input.query,
            assistant_message=answer,
            message_index=assistant_msg.message_index,
            created_at=assistant_msg.created_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] General tab chat failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/chat/conversations/{conversation_id}/messages", response_model=List[ConversationMessageResponse])
async def get_conversation_messages_endpoint(
    conversation_id: str = Path(..., description="Conversation UUID"),
    current_user: dict = Depends(require_scope("write")),
    db: Session = Depends(get_db)
):
    """
    Get all messages for a conversation in chronological order.
    
    Returns messages ordered by message_index ASC (0, 1, 2, 3...).
    Requires 'write' scope.
    
    - Returns 404 if conversation doesn't exist
    - Returns 403 if user doesn't own the conversation
    - Returns empty array [] if conversation exists but has no messages
    """
    try:
        import uuid as uuid_lib
        
        # Validate conversation_id format
        try:
            conv_uuid = uuid_lib.UUID(conversation_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid conversation_id format: {conversation_id}"
            )
        
        # Get current user
        user = db.query(User).filter(User.username == current_user["username"]).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # First check if conversation exists (without user_id filter)
        conversation_exists = db.query(VideoConversation).filter(
            VideoConversation.conversation_id == conv_uuid
        ).first()
        
        if not conversation_exists:
            # Phase 3.9: Enhanced 404 error message - conversation doesn't exist
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found. The conversation may have been deleted or never existed."
            )
        
        # Phase 5.12: Use utility function for ownership verification
        ownership_check_start = time.time()
        is_owner, error_msg = verify_conversation_ownership(db, conv_uuid, user.id)
        ownership_check_time = (time.time() - ownership_check_start) * 1000  # Convert to ms
        
        if not is_owner:
            # Enhanced debug logging for 403 errors (Phase 1.1)
            log_data = {
                "event": "ownership_check_failed",
                "conversation_id": str(conv_uuid),
                "current_user_id": str(user.id),
                "current_username": user.username,
                "conversation_owner_id": str(conversation_exists.user_id) if conversation_exists else None,
                "ownership_match": False,
                "check_time_ms": round(ownership_check_time, 2),
                "conversation_exists": conversation_exists is not None,
                "video_id": conversation_exists.video_id if conversation_exists else None,
                "video_title": conversation_exists.video_title[:100] if conversation_exists else None,  # Truncate for logging
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.warning(f"[AUTH] Ownership check failed: {json.dumps(log_data)}")
            
            # Phase 3.8: Enhanced 403 error message with diagnostic information
            owner_id_str = str(conversation_exists.user_id) if conversation_exists else "unknown"
            error_detail = (
                f"Access denied: Conversation {conversation_id} belongs to a different user. "
                f"Current user: {user.username} (ID: {str(user.id)}), "
                f"Conversation owner ID: {owner_id_str}. "
                f"If you believe this is an error, please contact support with this conversation ID."
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_detail
            )
        
        # Get conversation for further processing
        conversation = conversation_exists
        
        # Log successful ownership check (Phase 1.1)
        if DEBUG_LOGGING:
            log_data = {
                "event": "ownership_check_success",
                "conversation_id": str(conv_uuid),
                "current_user_id": str(user.id),
                "current_username": user.username,
                "conversation_owner_id": str(conversation.user_id),
                "ownership_match": True,
                "check_time_ms": round(ownership_check_time, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.debug(f"[AUTH] Ownership check success: {json.dumps(log_data)}")
        
        # Get all messages for this conversation (ordered by message_index ASC)
        # get_conversation_messages already orders by message_index ASC
        messages = get_conversation_messages(
            db=db,
            conversation_id=conv_uuid,
            limit=None,  # Get all messages
            offset=None
        )
        
        # Serialize messages for API response
        result = []
        for msg in messages:
            serialized = serialize_message_for_api(msg)
            result.append(serialized)
        
        # Return empty array if no messages (conversation exists but is empty)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to fetch conversation messages for {conversation_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch messages"
        )


