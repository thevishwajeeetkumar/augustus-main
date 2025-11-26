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
from langchain_core.tools import tool
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
from typing import Optional, Dict, Any, Tuple, List, Callable
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
    Build ReAct agent prompt with detailed, pointwise instructions.
    
    Keeps all detailed instructions as requested - pointwise, precise, and detailed.
    Organizes prompt for maintainability while preserving all essential guidance.
    
    Args:
        base_template: Optional base template from LangChain hub. If None, uses complete fallback.
    
    Returns:
        PromptTemplate configured with all detailed agent instructions.
    """
    # Common detailed instructions section (used in both enhanced and fallback)
    detailed_instructions = """

CRITICAL: Tool calling format rules:
- Action and Action Input must be on SEPARATE lines
- Do NOT use function call syntax like tool_name("input")
- CORRECT format:
  Action: youtube_rag_tool
  Action Input: "what is asyncio?"
- INCORRECT format (DO NOT USE):
  Action: youtube_rag_tool("what is asyncio?")
  Action Input: youtube_rag_tool("what is asyncio?")

CRITICAL: Query Intent Analysis and Tool Selection:
1. FIRST, analyze the query intent:
   - Is this a time-sensitive question? (e.g., "latest", "recent", "updates", "new features", "current", "2024", "2025")
   - Is this a general knowledge question about a topic? (e.g., "explain", "what is", "how does", "tell me about")

2. Tool Selection Strategy:
   - For GENERAL questions: Use youtube_rag_tool ONCE. Read the FIRST LINE of the Observation. If it says "FINAL_ANSWER_REQUIRED", STOP IMMEDIATELY. Your next Thought MUST be "I now know the final answer" then synthesize a COMPREHENSIVE final answer using ALL context. DO NOT call any tools again.
   - For TIME-SENSITIVE questions: Use youtube_rag_tool FIRST. Read the FIRST LINE of the Observation. If it says "NEXT_ACTION_REQUIRED" and "NEXT_STEP: Call google_search_tool NOW", call google_search_tool next. After google_search_tool returns "FINAL_ANSWER_REQUIRED", STOP IMMEDIATELY. Your next Thought MUST be "I now know the final answer" then synthesize a COMPREHENSIVE answer from both sources.

3. CRITICAL Tool Result Analysis (READ THE FIRST LINE OF OBSERVATION):
   - After EACH tool call, you MUST read the FIRST LINE of the Observation IMMEDIATELY:
     * If FIRST LINE contains "FINAL_ANSWER_REQUIRED" → STOP ALL TOOL CALLS. Your next Thought MUST be "I now know the final answer" and then provide Final Answer. DO NOT call any more tools. DO NOT call the same tool again.
     * If FIRST LINE contains "NEXT_ACTION_REQUIRED" → Read the "NEXT_STEP:" line and follow it exactly:
       - If it says "Call google_search_tool NOW" → Call google_search_tool immediately
       - If it says "Provide Final Answer NOW" → STOP ALL TOOL CALLS. Your next Thought MUST be "I now know the final answer" and then provide Final Answer.
     * If Observation says "No relevant information found" → Use google_search_tool or provide final answer
   - CRITICAL LOOP PREVENTION:
     * If you see "FINAL_ANSWER_REQUIRED" in ANY Observation, you MUST STOP and provide Final Answer
     * DO NOT call the same tool again if you already received "FINAL_ANSWER_REQUIRED"
     * DO NOT call youtube_rag_tool again after it returns "FINAL_ANSWER_REQUIRED"
     * Check your scratchpad: if you already called a tool and got "FINAL_ANSWER_REQUIRED", provide Final Answer immediately
   - CRITICAL: When providing Final Answer:
     * Your Thought MUST be: "I now know the final answer"
     * SYNTHESIZE comprehensively - don't just summarize
     * Use ALL relevant context from the Observation
     * Include specific details, examples, and key points from the context
     * Structure your answer clearly with proper organization
     * Be thorough and detailed - the user wants a complete understanding
   - CRITICAL: After calling google_search_tool, you will receive "FINAL_ANSWER_REQUIRED" - this means you have BOTH video context AND web search results. STOP ALL TOOL CALLS. Your next Thought MUST be "I now know the final answer" and then synthesize a COMPREHENSIVE final answer from both sources.
   - The Observation format is: [INSTRUCTION with plain text markers] followed by [context preview]
   - READ THE FIRST LINE FIRST - it tells you exactly what to do next
   - DO NOT call the same tool twice with the same input
   - DO NOT call youtube_rag_tool again after it returns "FINAL_ANSWER_REQUIRED"

4. Tool Call Limits:
   - youtube_rag_tool: Maximum 1 call per query (analyze result and decide next step)
   - google_search_tool: Maximum 1 call per query (only when needed based on youtube_rag_tool result or for time-sensitive queries)

5. Tool Call History Tracking:
   - Before calling ANY tool, check your previous actions in the scratchpad
   - The scratchpad format is: Thought → Action → Action Input → Observation → Thought (repeat)
   - If you see "Action: youtube_rag_tool" already executed → DO NOT call it again, check the Observation to see what to do next
   - If you see "Action: google_search_tool" already executed → Provide Final Answer immediately (you have all information: video context + web search results)
   - If you see both tools already called → You MUST provide Final Answer (no more tool calls needed)
   - Use this history to avoid redundant tool calls and prevent looping

Always use the CORRECT format with Action and Action Input on separate lines.
"""
    
    # Base template from hub or fallback
    if base_template:
        # Enhance hub template with detailed instructions
        enhanced_template = base_template + detailed_instructions
        return PromptTemplate.from_template(enhanced_template)
    else:
        # Complete fallback prompt with all details
        fallback_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

CRITICAL: When Observation contains "FINAL_ANSWER_REQUIRED", you MUST stop calling tools immediately:
Example:
Action: youtube_rag_tool_v2
Action Input: "What is discussed in this video?"
Observation: FINAL_ANSWER_REQUIRED - STOP ALL TOOL CALLS NOW. Your next Thought MUST be "I now know the final answer" then provide Final Answer.
Thought: I now know the final answer
Final Answer: [Your comprehensive answer here]

Thought: I now know the final answer
Final Answer: Provide a comprehensive, detailed answer that synthesizes all relevant context from the observations. Use specific details, examples, and key points. Structure your answer clearly and be thorough.
""" + detailed_instructions + """
Begin!

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
        time_sensitive_keywords = ["latest", "recent", "update", "new", "current", "2024", "2025", "now", "today"]
        is_time_sensitive = any(keyword in query_text.lower() for keyword in time_sensitive_keywords)
        
        if not final_docs:
            return "OBSERVATION: No relevant information found in video transcript. Consider asking a more specific question or use google_search_tool_v2 if available."
        
        context = format_docs(final_docs)
        
        # Return structured response with plain text markers for agent guidance
        if is_time_sensitive:
            return f"""OBSERVATION:
NEXT_ACTION_REQUIRED
TIME_SENSITIVE query. Historical context from video {video_id}.
NEXT_STEP: Call google_search_tool_v2 NOW

Video context:
{context}"""
        else:
            return f"""OBSERVATION:
FINAL_ANSWER_REQUIRED - STOP ALL TOOL CALLS NOW. Your next Thought MUST be "I now know the final answer" then provide Final Answer.
Sufficient context from video {video_id}. DO NOT call any more tools.

IMPORTANT: Your Final Answer should:
- Synthesize comprehensively (not just summarize)
- Use ALL relevant context provided
- Include specific details, examples, and key points
- Be thorough and detailed for complete understanding

Video context:
{context}"""
    
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
        Performs a Google Custom Search and returns the top results.
        
        Use only when youtube_rag_tool_v2 cannot answer the question or for current events
        that are not covered in the video content.
        
        IMPORTANT: For time-sensitive queries (latest, recent, new, current updates):
        - This tool should be called AFTER youtube_rag_tool_v2
        - Combine video context with web results in your final answer
        
        Format example:
        Action: google_search_tool_v2
        Action Input: "latest news about Python asyncio"
        
        Do NOT use function call syntax like google_search_tool_v2("query").
        """
        # Use consistent environment variable names (matches top of file)
        search_api_key = GOOGLE_SEARCH_API_KEY
        search_engine_id = GOOGLE_SEARCH_ENGINE_ID
        
        if not search_api_key or not search_engine_id:
            return "OBSERVATION: Web search is not configured. Please provide GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID."
        
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
                return "OBSERVATION: No web search results found."
            
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
            
            # Return structured response with plain text markers
            return f"""OBSERVATION:
FINAL_ANSWER_REQUIRED - STOP ALL TOOL CALLS NOW. Your next Thought MUST be "I now know the final answer" then provide Final Answer.
Web search results received. Synthesize video context + web results. DO NOT call any more tools.

IMPORTANT: Your Final Answer should:
- Synthesize comprehensively from BOTH video context AND web search results
- Use ALL relevant information from both sources
- Include specific details, examples, and key points
- Be thorough and detailed for complete understanding

WEB_SEARCH_RESULTS:
{context}"""
        
        except Exception as e:
            return f"OBSERVATION: Web search failed: {str(e)}"
    
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
        PRIORITY TOOL: Use this FIRST for any question about video content.
        
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
        time_sensitive_keywords = ["latest", "recent", "update", "new", "current", "2024", "2025", "now", "today"]
        is_time_sensitive = any(keyword in query_text.lower() for keyword in time_sensitive_keywords)
        
        if not docs:
            return "OBSERVATION: No relevant information found in processed videos. Consider using google_search_tool_v2 if this is a current events question."
        
        context = format_docs(docs)
        
        # Get citations for response (create new session to avoid stale session issues)
        citations = format_citations_from_docs(docs, lambda: SessionLocal())
        citation_text = ""
        if citations:
            citation_text = "\n\nSources:\n"
            for vid_id, info in citations.items():
                citation_text += f"- {info['title']}: {info['url']}\n"
        
        # Return structured response with citations
        if is_time_sensitive:
            return f"""OBSERVATION:
NEXT_ACTION_REQUIRED
TIME_SENSITIVE query. Historical context from {len(shortlisted_video_ids)} videos.
NEXT_STEP: Call google_search_tool_v2 NOW

Video context:
{context}
{citation_text}"""
        else:
            return f"""OBSERVATION:
FINAL_ANSWER_REQUIRED - STOP ALL TOOL CALLS NOW. Your next Thought MUST be "I now know the final answer" then provide Final Answer.
Sufficient context from {len(shortlisted_video_ids)} videos. DO NOT call any more tools.

IMPORTANT: Your Final Answer should:
- Synthesize comprehensively (not just summarize)
- Use ALL relevant context from all videos provided
- Include specific details, examples, and key points
- Be thorough and detailed for complete understanding

Video context:
{context}
{citation_text}"""
    
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
        
        # STEP 5: Create agent tools using factory functions
        youtube_rag_tool_v2 = create_youtube_rag_tool_v2(index, namespace, video_id)
        google_search_tool_v2 = create_google_search_tool_v2()
        
        # Verify tools are properly structured (debugging aid)
        tools = [youtube_rag_tool_v2, google_search_tool_v2]
        print(f"[DEBUG] Created {len(tools)} tools: {[tool.name if hasattr(tool, 'name') else type(tool).__name__ for tool in tools]}")
        
        # Create callback to track tool execution and detect looping
        class ToolTrackingCallback(BaseCallbackHandler):
            """Callback that tracks tool calls to detect looping"""
            def __init__(self):
                super().__init__()
                self.tool_calls = []
                self.tool_results = []
            
            def on_tool_end(self, output: str, **kwargs) -> None:
                """Called when a tool finishes execution"""
                tool_name = kwargs.get('name', 'unknown')
                self.tool_calls.append(tool_name)
                if isinstance(output, str):
                    self.tool_results.append(output)
                    # Log tool execution for monitoring
                    if "youtube_rag_tool_v2" in tool_name:
                        if "FINAL_ANSWER_REQUIRED" in output:
                            print(f"[CALLBACK] youtube_rag_tool_v2 returned context - agent should provide final answer")
                        elif "NEXT_ACTION_REQUIRED" in output:
                            print(f"[CALLBACK] youtube_rag_tool_v2 detected time-sensitive query - agent should call google_search_tool_v2")
                    elif "google_search_tool_v2" in tool_name:
                        print(f"[CALLBACK] google_search_tool_v2 executed - agent should provide final answer")
        
        tool_tracking_callback = ToolTrackingCallback()
        
        # Create agent executor with enhanced prompt and callbacks
        # Use cached REACT_AGENT_PROMPT which includes tool instructions
        agent = create_react_agent(model, tools, REACT_AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=5,  # Increased to allow proper answer synthesis from large contexts
            handle_parsing_errors=True,
            callbacks=[tool_tracking_callback]  # Add callback for monitoring
        )
        
        # Invoke agent with timeout protection
        import asyncio
        import time
        
        start_time = time.time()
        timeout_seconds = 60.0  # 60 second timeout
        
        try:
            # Run agent with timeout
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
            
            answer = result.get("output", "I couldn't generate an answer.")
            
            # Detect looping behavior by analyzing tool calls
            if len(tool_tracking_callback.tool_calls) > 0:
                # Count occurrences of each tool
                tool_count = {}
                for tool_name in tool_tracking_callback.tool_calls:
                    tool_count[tool_name] = tool_count.get(tool_name, 0) + 1
                
                # Check for excessive calls to same tool
                for tool_name, count in tool_count.items():
                    if count > 2:
                        print(f"[WARNING] Detected potential looping: {tool_name} called {count} times")
                        print(f"[WARNING] Tool call sequence: {' -> '.join(tool_tracking_callback.tool_calls)}")
                
                # Log tool usage summary
                print(f"[MONITOR] Tools used: {', '.join(set(tool_tracking_callback.tool_calls))}")
                print(f"[MONITOR] Total tool calls: {len(tool_tracking_callback.tool_calls)}")
        
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
        
        # STEP 5: Create General tab tools using factory functions
        youtube_rag_tool_general = create_youtube_rag_tool_general(index, namespace)
        google_search_tool_v2 = create_google_search_tool_v2()
        
        # Verify tools are properly structured (debugging aid)
        tools = [youtube_rag_tool_general, google_search_tool_v2]
        print(f"[DEBUG] Created {len(tools)} tools: {[tool.name if hasattr(tool, 'name') else type(tool).__name__ for tool in tools]}")
        
        # Create agent executor
        agent = create_react_agent(model, tools, REACT_AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,  # Increased to allow proper answer synthesis from large contexts
            handle_parsing_errors=True
        )
        
        # Run agent with timeout
        import asyncio
        import time
        
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


