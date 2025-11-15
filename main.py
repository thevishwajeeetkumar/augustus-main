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
import os,hashlib
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
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db, User, UserSession, Message, SessionLocal
from langchain_core.messages import HumanMessage, AIMessage
try:
    import bcrypt
except ImportError:
    print("Warning: bcrypt not installed. Install with: pip install bcrypt")
    bcrypt = None

from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

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

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    if jwt is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

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
    """Get current user from OAuth2 token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = verify_token(token)
    if username is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
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


class input(BaseModel):
    query:str
    url:Optional[HttpUrl] = None
    session_id: Optional[str] = None  # For session continuity

class AgentResponse(BaseModel):
    answer: str
    session_id: str
    video_context: Optional[str] = None

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

class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str
    created_at: str
    is_active: bool = True
    scopes: list[str] = ["read", "write"]  # OAuth2 
    









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

model = ChatOpenAI(model="gpt-4.1-mini")  # Uses OPENAI_API_KEY from environment

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
        # We have a valid template - enhance it
        enhanced_template = original_template + """

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
   - For GENERAL questions: Use youtube_rag_tool ONCE. Analyze the result. If it says "✅ FINAL ANSWER REQUIRED ✅", IMMEDIATELY synthesize the final answer. DO NOT call it again.
   - For TIME-SENSITIVE questions: Use youtube_rag_tool FIRST. Analyze the result. If it says "🚨 NEXT ACTION REQUIRED 🚨" and "→ NEXT STEP: Call google_search_tool NOW", call google_search_tool next, then synthesize both sources.

3. CRITICAL Tool Result Analysis (READ THE FIRST 3 LINES OF OBSERVATION):
   - After EACH tool call, you MUST read the FIRST 3 lines of the Observation:
     * If Observation contains "✅ FINAL ANSWER REQUIRED ✅" → IMMEDIATELY provide Final Answer (DO NOT call any more tools)
     * If Observation contains "🚨 NEXT ACTION REQUIRED 🚨" → Read the "→ NEXT STEP:" line and follow it exactly:
       - If it says "Call google_search_tool NOW" → Call google_search_tool immediately
       - If it says "Provide Final Answer NOW" → Provide Final Answer immediately
     * If Observation says "No relevant information found" → Use google_search_tool or provide final answer
   - CRITICAL: After calling google_search_tool, you will receive "✅ FINAL ANSWER REQUIRED ✅" - this means you have BOTH video context AND web search results. IMMEDIATELY synthesize the final answer from both sources. DO NOT call youtube_rag_tool again.
   - CRITICAL: Check your previous actions in the scratchpad - if you already called youtube_rag_tool, DO NOT call it again. If you already called google_search_tool, provide Final Answer.
   - The Observation format is: [INSTRUCTION with emoji markers] followed by [context preview]
   - READ THE FIRST 3 LINES FIRST - they tell you exactly what to do next
   - DO NOT call the same tool twice with the same input
   - DO NOT call youtube_rag_tool again after it returns context - follow the "→ NEXT STEP:" instruction

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
        REACT_AGENT_PROMPT = PromptTemplate.from_template(enhanced_template)
        print("[OK] Agent prompt loaded from LangChain hub and enhanced with format instructions")
    else:
        # No valid template extracted - use fallback prompt
        print("[WARN] Could not extract valid template from hub, using fallback prompt")
        REACT_AGENT_PROMPT = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL: Tool calling format rules:
- Action and Action Input must be on SEPARATE lines
- Do NOT use function call syntax like tool_name("input")
- CORRECT format:
  Action: youtube_rag_tool
  Action Input: "what is asyncio?"
- INCORRECT format (DO NOT USE):
  Action: youtube_rag_tool("what is asyncio?")

CRITICAL: Query Intent Analysis and Tool Selection:
1. FIRST, analyze the query intent:
   - Is this a time-sensitive question? (e.g., "latest", "recent", "updates", "new features", "current", "2024", "2025")
   - Is this a general knowledge question about a topic? (e.g., "explain", "what is", "how does", "tell me about")

2. Tool Selection Strategy:
   - For GENERAL questions: Use youtube_rag_tool ONCE. Analyze the result. If it says "✅ FINAL ANSWER REQUIRED ✅", IMMEDIATELY synthesize the final answer. DO NOT call it again.
   - For TIME-SENSITIVE questions: Use youtube_rag_tool FIRST. Analyze the result. If it says "🚨 NEXT ACTION REQUIRED 🚨" and "→ NEXT STEP: Call google_search_tool NOW", call google_search_tool next, then synthesize both sources.

3. CRITICAL Tool Result Analysis (READ THE FIRST 3 LINES OF OBSERVATION):
   - After EACH tool call, you MUST read the FIRST 3 lines of the Observation:
     * If Observation contains "✅ FINAL ANSWER REQUIRED ✅" → IMMEDIATELY provide Final Answer (DO NOT call any more tools)
     * If Observation contains "🚨 NEXT ACTION REQUIRED 🚨" → Read the "→ NEXT STEP:" line and follow it exactly:
       - If it says "Call google_search_tool NOW" → Call google_search_tool immediately
       - If it says "Provide Final Answer NOW" → Provide Final Answer immediately
     * If Observation says "No relevant information found" → Use google_search_tool or provide final answer
   - CRITICAL: After calling google_search_tool, you will receive "✅ FINAL ANSWER REQUIRED ✅" - this means you have BOTH video context AND web search results. IMMEDIATELY synthesize the final answer from both sources. DO NOT call youtube_rag_tool again.
   - CRITICAL: Check your previous actions in the scratchpad - if you already called youtube_rag_tool, DO NOT call it again. If you already called google_search_tool, provide Final Answer.
   - The Observation format is: [INSTRUCTION with emoji markers] followed by [context preview]
   - READ THE FIRST 3 LINES FIRST - they tell you exactly what to do next
   - DO NOT call the same tool twice with the same input
   - DO NOT call youtube_rag_tool again after it returns context - follow the "→ NEXT STEP:" instruction

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

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
        print("[OK] Fallback prompt loaded")
except Exception as e:
    print(f"[WARN] Could not load agent prompt from hub: {e}")
    print("   Using fallback prompt")
    # Fallback prompt if hub is unavailable
    # PromptTemplate already imported at top of file
    REACT_AGENT_PROMPT = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL: Tool calling format rules:
- Action and Action Input must be on SEPARATE lines
- Do NOT use function call syntax like tool_name("input")
- CORRECT format:
  Action: youtube_rag_tool
  Action Input: "what is asyncio?"
- INCORRECT format (DO NOT USE):
  Action: youtube_rag_tool("what is asyncio?")

CRITICAL: Query Intent Analysis and Tool Selection:
1. FIRST, analyze the query intent:
   - Is this a time-sensitive question? (e.g., "latest", "recent", "updates", "new features", "current", "2024", "2025")
   - Is this a general knowledge question about a topic? (e.g., "explain", "what is", "how does", "tell me about")

2. Tool Selection Strategy:
   - For GENERAL questions: Use youtube_rag_tool ONCE. Analyze the result. If it says "✅ FINAL ANSWER REQUIRED ✅", IMMEDIATELY synthesize the final answer. DO NOT call it again.
   - For TIME-SENSITIVE questions: Use youtube_rag_tool FIRST. Analyze the result. If it says "🚨 NEXT ACTION REQUIRED 🚨" and "→ NEXT STEP: Call google_search_tool NOW", call google_search_tool next, then synthesize both sources.

3. CRITICAL Tool Result Analysis (READ THE FIRST 3 LINES OF OBSERVATION):
   - After EACH tool call, you MUST read the FIRST 3 lines of the Observation:
     * If Observation contains "✅ FINAL ANSWER REQUIRED ✅" → IMMEDIATELY provide Final Answer (DO NOT call any more tools)
     * If Observation contains "🚨 NEXT ACTION REQUIRED 🚨" → Read the "→ NEXT STEP:" line and follow it exactly:
       - If it says "Call google_search_tool NOW" → Call google_search_tool immediately
       - If it says "Provide Final Answer NOW" → Provide Final Answer immediately
     * If Observation says "No relevant information found" → Use google_search_tool or provide final answer
   - CRITICAL: After calling google_search_tool, you will receive "✅ FINAL ANSWER REQUIRED ✅" - this means you have BOTH video context AND web search results. IMMEDIATELY synthesize the final answer from both sources. DO NOT call youtube_rag_tool again.
   - CRITICAL: Check your previous actions in the scratchpad - if you already called youtube_rag_tool, DO NOT call it again. If you already called google_search_tool, provide Final Answer.
   - The Observation format is: [INSTRUCTION with emoji markers] followed by [context preview]
   - READ THE FIRST 3 LINES FIRST - they tell you exactly what to do next
   - DO NOT call the same tool twice with the same input
   - DO NOT call youtube_rag_tool again after it returns context - follow the "→ NEXT STEP:" instruction

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

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

#Pinecone Database setting-----------------------------------------------------------------------------
# Initialize Pinecone client - using shared index with namespaces
SHARED_INDEX_NAME = "augustus-videos-integrated"

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


prompt=PromptTemplate(
        template="<role>You are an helpful assistant, you only know about the context provided to you.</role>\n <task>Answer the {question} using context</task>\n <instructions>\n1)Query the  vector store to complete the task.\n2)Don't use the internet, or your internal knowledge to answer the question, use only the vector store.\n3)Expand on your answers in simple yet explanatory way.</instructions><context>Here is the necessary info-\n{context}</context>",
        input_variables=['question','context']
)


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
        time.sleep(0.1)  # Rate limiting


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
    access_token = create_access_token(
        data={
            "sub": user_data.username,
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
    access_token = create_access_token(
        data={
            "sub": user.username,
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

#ingesting------------------------------------------------------------------------------------------------
@app.post("/", response_model=AgentResponse)
def answer_from_video(user_input: input, current_user: dict = Depends(require_scope("write")), db: Session = Depends(get_db)) -> dict:
    """
    Intelligent Q&A with agent-based tool selection and persistent context.
    
    Features:
    - User-specific Pinecone indexes for data isolation
    - Session-based conversation history (sliding window)
    - Agent decides between YouTube RAG and web search
    - Supports follow-up questions across requests
    
    Requires 'write' scope.
    """
    # Check if Pinecone client is initialized
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )
    
    # ========================================
    # STEP 1: Session Management
    # ========================================
    
    # Get user from database
    user = db.query(User).filter(User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get or create session
    session_id_input = user_input.session_id
    session = None
    
    if session_id_input:
        # Try to find the specific session requested
        try:
            session = db.query(UserSession).filter(
                UserSession.session_id == session_id_input,
                UserSession.user_id == user.id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            ).first()
            
            if session:
                print(f"[INFO] Resuming session {session.session_id}")
        except Exception as e:
            print(f"[WARN] Invalid session_id format: {e}")
            session = None
    
    # If no session found, try to auto-resume user's most recent session
    if not session:
        # Try to find user's most recent active session (auto-resume)
        session = db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).order_by(UserSession.last_activity.desc()).first()
        
        if session:
            print(f"[INFO] Auto-resumed most recent session {session.session_id}")
        else:
            # No active sessions - create new one
            session = UserSession(user_id=user.id)
            db.add(session)
            db.commit()
            db.refresh(session)
            print(f"[INFO] Created new session {session.session_id} for user {user.username}")
    
    # ========================================
    # STEP 2: Video Embedding (if URL provided)
    # ========================================
    
    query = user_input.query
    url = str(user_input.url) if user_input.url else None
    current_video_id = None
    
    if url:
        try:
            # Extract video_id from URL first (no API call needed)
            video_id = get_youtube_video_id(url)
            if not video_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid YouTube URL. Could not extract video ID."
                )
            
            current_video_id = video_id
            
            # Get shared index and namespace
            index = get_shared_index()
            namespace = get_user_namespace(user.username)
            
            # Check if video already embedded in Pinecone (before fetching transcript)
            if check_video_exists_in_pinecone(index, namespace, video_id):
                print(f"[INFO] Video {video_id} already embedded, skipping transcript fetch")
                # Update session with current video and continue to Q&A
                session.current_video_id = video_id
                session.current_video_url = url
                session.last_activity = datetime.utcnow()
                db.commit()
            else:
                # Video not found in Pinecone - fetch transcript and embed
                print(f"[INFO] Video {video_id} not found in Pinecone, fetching transcript")
                
                # Fetch transcript
                l = fetchTranscript(url)
                transcript = l[0]
                # video_id already extracted above, no need to get from l[1]
                
                # Embed in Pinecone
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
                chunks = parent_splitter.create_documents([transcript])
                
                # Prepare records (text only - Pinecone embeds automatically)
                # Format matches AGENTS.md line 404-411
                records = []
                for i, chunk in enumerate(chunks):
                    vector_id = f"{video_id}:{hashlib.sha1(chunk.page_content.encode('utf-8')).hexdigest()[:16]}"
                    records.append({
                        "_id": vector_id,
                        "content": chunk.page_content,  # CRITICAL: Must match field_map (text=content)
                        "video_id": video_id,            # Metadata (flat structure - AGENTS.md line 630-635)
                        "doc_id": f"{video_id}:p{i}"     # Metadata (flat structure)
                    })
                    # Note: No nested objects allowed (AGENTS.md line 619-636)
                    # All metadata must be flat: strings, ints, floats, bools, string lists only
                
                # Defensive check: verify records don't already exist (should rarely trigger since we checked above)
                # This handles potential race conditions or check failures
                already = False
                if records:
                    probe_ids = [r["_id"] for r in records[:8]]
                    try:
                        result = index.fetch(namespace=namespace, ids=probe_ids)
                        # CRITICAL: Use result.records (not result.vectors) for integrated embeddings
                        # AGENTS.md line 449: if result.records:
                        already = len(result.records) > 0
                    except Exception:
                        already = False
                
                # Batch upsert with retry
                # AGENTS.md line 416: index.upsert_records(namespace, records)
                if not already:
                    batch_upsert_records(index, namespace, records, batch_size=96)  # Max 96 per batch
                    print(f"[OK] Embedded video {video_id} for user {user.username}")
                else:
                    print(f"[INFO] Video {video_id} already embedded (defensive check triggered)")
                
                # Update session with current video
                session.current_video_id = video_id
                session.current_video_url = url
                session.last_activity = datetime.utcnow()
                db.commit()
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    # ========================================
    # STEP 3: Build Agent Context from DB
    # ========================================
    
    # Get recent messages for context
    recent_messages = db.query(Message).filter(
        Message.session_id == session.session_id
    ).order_by(Message.message_index.desc()).limit(10).all()
    recent_messages.reverse()  # Chronological order
    
    # Convert to LangChain format
    history = [
        HumanMessage(content=m.content) if m.message_type == 'human'
        else AIMessage(content=m.content)
        for m in recent_messages
    ]
    
    # ========================================
    # STEP 4: Create Context-Aware Tools
    # ========================================
    
    # Capture context in closure
    session_id_str = str(session.session_id)
    username_str = user.username
    current_vid = current_video_id or session.current_video_id
    
    # Get current date/time for time-sensitive queries
    current_time = datetime.utcnow()
    current_year = current_time.year
    current_date_str = current_time.strftime("%Y-%m-%d")
    
    @tool
    def youtube_rag_tool(question: str) -> str:
        """
        PRIORITY TOOL: Use this FIRST for any question about video content.
        
        Returns relevant context from YouTube videos for the agent to synthesize into an answer.
        This tool searches across ALL videos processed by the user (including videos from previous sessions).
        If the current session has a video_id, it will prioritize that video; otherwise, it searches
        across all user videos.
        
        This tool retrieves and returns raw context from the videos - the agent will use this
        context to generate the final answer.
        
        Use this when user asks about video content, topics, or specific details from videos.
        Input should be the user's question about the video.
        
        Format example:
        Action: youtube_rag_tool
        Action Input: "explain the main topic"
        
        Do NOT use function call syntax like youtube_rag_tool("question").
        """
        tool_db = SessionLocal()
        try:
            # Get session context
            sess = tool_db.query(UserSession).filter(
                UserSession.session_id == session_id_str
            ).first()
            
            if not sess:
                return "OBSERVATION: Session not found. Cannot retrieve video information."
            
            # Get shared index and namespace
            index = get_shared_index()
            namespace = get_user_namespace(username_str)
            
            # Build query with text input (Pinecone embeds automatically)
            # AGENTS.md line 510-515: query structure for integrated embeddings
            query_dict = {
                "top_k": 12 * 2,  # More candidates for reranking (AGENTS.md line 511)
                "inputs": {
                    "text": question  # Text input (must match field_map: text=content)
                }
            }
            
            # Add filter if current_video_id exists
            # AGENTS.md line 825-841: Dynamic filter pattern - only add if exists
            if sess.current_video_id:
                query_dict["filter"] = {"video_id": sess.current_video_id}
                print(f"[youtube_rag_tool] Searching with video_id filter: {sess.current_video_id}")
            else:
                # CRITICAL: Don't set filter to None - omit the key entirely (AGENTS.md line 826)
                print(f"[youtube_rag_tool] No current_video_id, searching across all user videos")
            
            # Search with reranking (AGENTS.md line 213-217, 516-520)
            # AGENTS.md line 662-674: "always rerank in production"
            try:
                results = exponential_backoff_retry(
                    lambda: index.search(
                        namespace=namespace,
                        query=query_dict,
                        rerank={
                            "model": "bge-reranker-v2-m3",  # AGENTS.md line 517, 214
                            "top_n": 12,                    # Final number after reranking
                            "rank_fields": ["content"]       # Field to rerank on (AGENTS.md line 519, 216)
                        }
                    )
                )
                
                # Convert results to LangChain Document format
                # CRITICAL: With reranking, use dict-style access (AGENTS.md line 221-230)
                # AGENTS.md line 221: reranked_results['result']['hits']
                # AGENTS.md line 225: "IMPORTANT: With reranking, use dict-style access"
                raw_docs = []
                for hit in results['result']['hits']:  # Dict access pattern
                    doc = Document(
                        page_content=hit['fields']['content'],  # Dict access (AGENTS.md line 222, 229)
                        metadata={
                            'video_id': hit['fields'].get('video_id'),  # Use .get() for optional (AGENTS.md line 230)
                            'doc_id': hit['fields'].get('doc_id'),
                            '_id': hit['_id'],              # Dict access (AGENTS.md line 227)
                            '_score': hit['_score']         # Dict access (AGENTS.md line 228)
                        }
                    )
                    raw_docs.append(doc)
                
                # Apply post-processing: score thresholding, diversity, context management
                docs = process_retrieval_results(
                    raw_docs,
                    min_score=MIN_SIMILARITY_SCORE,
                    max_per_video=MAX_CHUNKS_PER_VIDEO,
                    max_context_length=MAX_CONTEXT_LENGTH
                )
            except Exception as e:
                print(f"[ERROR] Search failed: {e}")
                import traceback
                traceback.print_exc()
                docs = []
            
            if not docs:
                if sess.current_video_id:
                    return "OBSERVATION: No relevant information found in the current video for this question. The retrieved documents did not contain information related to the query. You should either: 1) Try rephrasing the question, 2) Use google_search_tool if the question is about current events, or 3) Provide a final answer explaining that the video does not contain relevant information."
                else:
                    return "OBSERVATION: No relevant information found in any processed videos for this question. The user's video index does not contain information related to the query. You should either: 1) Use google_search_tool if available to answer the question, or 2) Provide a final answer explaining that no relevant video content was found."
            
            # Format and return raw context (NOT a pre-generated answer)
            # The agent will synthesize this context into a final answer
            context = format_docs(docs, max_length=MAX_CONTEXT_LENGTH)
            
            # Add context about which video(s) the information came from
            video_ids_found = set()
            for doc in docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    vid_id = doc.metadata.get('video_id')
                    if vid_id:
                        video_ids_found.add(vid_id)
            
            # Determine if query is time-sensitive (check the original question)
            time_sensitive_keywords = ["latest", "recent", "update", "new", "current", "2024", "2025", "now", "today"]
            is_time_sensitive_query = any(keyword in question.lower() for keyword in time_sensitive_keywords)
            
            # Structure response with CRITICAL INSTRUCTION FIRST, then context
            # Use full context - no truncation (MAX_CONTEXT_LENGTH already limits retrieval)
            context_preview = context
            
            if video_ids_found:
                if len(video_ids_found) == 1:
                    vid_id = list(video_ids_found)[0]
                    if is_time_sensitive_query:
                        return f"""OBSERVATION: 
🚨 NEXT ACTION REQUIRED 🚨
This is a TIME-SENSITIVE query. Historical context found from video {vid_id}.
→ NEXT STEP: Call google_search_tool NOW (do NOT call youtube_rag_tool again)

Video context preview:
{context_preview}"""
                    else:
                        return f"""OBSERVATION: 
✅ FINAL ANSWER REQUIRED ✅
Sufficient context found from video {vid_id} to answer this GENERAL question.
→ NEXT STEP: Provide Final Answer NOW (do NOT call any more tools)

Video context:
{context_preview}"""
                else:
                    if is_time_sensitive_query:
                        return f"""OBSERVATION: 
🚨 NEXT ACTION REQUIRED 🚨
This is a TIME-SENSITIVE query. Historical context found from {len(video_ids_found)} videos.
→ NEXT STEP: Call google_search_tool NOW (do NOT call youtube_rag_tool again)

Video context preview:
{context_preview}"""
                    else:
                        return f"""OBSERVATION: 
✅ FINAL ANSWER REQUIRED ✅
Sufficient context found from {len(video_ids_found)} videos to answer this GENERAL question.
→ NEXT STEP: Provide Final Answer NOW (do NOT call any more tools)

Video context:
{context_preview}"""
            else:
                if is_time_sensitive_query:
                    return f"""OBSERVATION: 
🚨 NEXT ACTION REQUIRED 🚨
This is a TIME-SENSITIVE query. Historical context found.
→ NEXT STEP: Call google_search_tool NOW (do NOT call youtube_rag_tool again)

Video context preview:
{context_preview}"""
                else:
                    return f"""OBSERVATION: 
✅ FINAL ANSWER REQUIRED ✅
Sufficient context found to answer this GENERAL question.
→ NEXT STEP: Provide Final Answer NOW (do NOT call any more tools)

Video context:
{context_preview}"""
            
        except Exception as e:
            return f"Error retrieving video information: {str(e)}"
        finally:
            tool_db.close()
    
    @tool
    def google_search_tool(query_text: str) -> str:
        """Performs a Google Custom Search and returns the top results.
        
Use only when youtube_rag_tool cannot answer the question or for current events
that are not covered in the video content.

IMPORTANT: For time-sensitive queries (latest, recent, new, current updates):
- This tool automatically updates the search query to use the current year
- If the query contains an outdated year, it will be replaced with the current year
- This ensures you get the most recent information available

Format example:
Action: google_search_tool
Action Input: "latest news about Python asyncio"

Do NOT use function call syntax like google_search_tool("query")."""
        if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
            # return "Google search is not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID."
            raise ValueError("Google search is not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in the environment.")
        
        # Update query for time-sensitive searches: replace outdated years with current year
        import re
        updated_query = query_text
        
        # Detect if query is time-sensitive (contains latest, recent, new, current, updates)
        time_sensitive_keywords = ["latest", "recent", "new", "current", "update", "updates"]
        is_time_sensitive = any(keyword in query_text.lower() for keyword in time_sensitive_keywords)
        
        if is_time_sensitive:
            # Replace any 4-digit years (1900-2099) that are not the current year with current year
            year_pattern = r'\b(19|20)\d{2}\b'
            updated_query = query_text
            found_outdated_year = False
            
            # Find all years in the query
            for year_match in re.finditer(year_pattern, query_text):
                year = int(year_match.group())
                if year < current_year:
                    # Replace this specific outdated year with current year
                    updated_query = updated_query.replace(year_match.group(), str(current_year), 1)
                    found_outdated_year = True
                    print(f"[google_search_tool] Updated query year from {year} to {current_year}: '{query_text}' -> '{updated_query}'")
                    break  # Replace first outdated year found
            
            # If no outdated year found, add current year if query is about "latest" or "recent"
            if not found_outdated_year:
                if ("latest" in query_text.lower() or "recent" in query_text.lower() or "new" in query_text.lower()):
                    if str(current_year) not in query_text:
                        updated_query = f"{query_text} {current_year}"
                        print(f"[google_search_tool] Added current year {current_year} to query: '{query_text}' -> '{updated_query}'")
        
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": updated_query,
        }

        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Google search request failed: {e}"

        try:
            payload = response.json()
        except ValueError:
            return "Google search returned an invalid response."

        items = payload.get("items", [])
        if not items:
            return "Google search returned no results."

        summaries = []
        for item in items[:5]:
            title = item.get("title", "Untitled result")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            summaries.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}")

        # Join summaries outside f-string to avoid backslash in expression
        search_results = "\n\n".join(summaries)
        
        return f"""OBSERVATION: 
✅ FINAL ANSWER REQUIRED ✅
You have received web search results for current/recent information.
→ NEXT STEP: Provide Final Answer NOW by synthesizing:
  1. The video context from youtube_rag_tool (already retrieved)
  2. The web search results below
DO NOT call any more tools - you have all the information needed.

=== WEB SEARCH RESULTS ===
{search_results}"""
    
    # ========================================
    # STEP 5: Create and Execute Agent
    # ========================================
    
    # Custom error handler for tool parsing errors
    def handle_tool_parsing_error(error: str, tools: list) -> str:
        """
        Detects malformed tool calls and reformats them to proper ReAct format.
        Handles patterns like tool_name("input") or tool_name('input')
        """
        import re
        
        # Pattern to match function call syntax: tool_name("input") or tool_name('input')
        pattern = r'(\w+)\s*\(["\']([^"\']+)["\']\)'
        match = re.search(pattern, error)
        
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            
            # Check if tool_name is a valid tool
            tool_names = [tool.name for tool in tools]
            if tool_name in tool_names:
                # Return corrected format
                return f"I need to use the {tool_name} tool. Action: {tool_name}\nAction Input: {tool_input}"
        
        # If no pattern matched, return original error message
        return f"Invalid tool format. Please use:\nAction: tool_name\nAction Input: your_input\n\nError: {error}"
    
    # Pre-execution validation: Check tool availability and video context
    tools_list = [youtube_rag_tool]  # Always include RAG tool
    
    # Only add Google search tool if configured
    if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        tools_list.append(google_search_tool)
    else:
        print("[INFO] Google search tool not available (not configured)")
    
    # Validate video context exists
    if not current_vid and not session.current_video_id:
        print("[WARN] No video context available for this session")
    
    # Create agent with request-specific tools (use cached prompt)
    agent = create_react_agent(
        llm=model,
        tools=tools_list,
        prompt=REACT_AGENT_PROMPT  # Use cached prompt from module level
    )
    
    # Create error handler wrapper
    def parsing_error_handler(error: str) -> str:
        return handle_tool_parsing_error(error, tools_list)
    
    # Custom callback to track tool execution and detect looping
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
                # Log tool execution
                if "youtube_rag_tool" in tool_name:
                    if "Found relevant context" in output:
                        print(f"[CALLBACK] youtube_rag_tool returned context - agent should analyze and decide next step")
                    elif "No relevant information" in output:
                        print(f"[CALLBACK] youtube_rag_tool found no context - agent should try google_search_tool or provide final answer")
    
    tool_tracking_callback = ToolTrackingCallback()
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        handle_parsing_errors=parsing_error_handler,
        verbose=True,  # Disable verbose to avoid callback errors
        max_iterations=5,  # Allow: 1 for youtube_rag, 1 for google_search (if needed), 1 for final answer + buffer for looping
        return_intermediate_steps=True,  # Enable to detect tool failures
        callbacks=[tool_tracking_callback]  # Track tool calls
    )
    
    # Build agent input with context and history
    context_parts = []
    
    # Add instructions based on video context availability
    if current_vid or session.current_video_id:
        context_parts.append("=" * 60)
        context_parts.append("IMPORTANT: VIDEO CONTEXT IS AVAILABLE")
        context_parts.append("=" * 60)
        context_parts.append("ALWAYS use youtube_rag_tool FIRST for any question about video content.")
        context_parts.append("Do NOT use your general knowledge if video context is available.")
        context_parts.append("Only use google_search_tool if youtube_rag_tool cannot answer or for current events.")
        context_parts.append("=" * 60)
        context_parts.append("")
    else:
        # No video context in current session - but videos may exist from previous sessions
        context_parts.append("=" * 60)
        context_parts.append("NOTE: NO CURRENT VIDEO IN THIS SESSION")
        context_parts.append("=" * 60)
        context_parts.append("No video has been processed in this session yet.")
        context_parts.append("However, youtube_rag_tool will search across ALL videos processed by this user")
        context_parts.append("(including videos from previous sessions).")
        context_parts.append("You should still try youtube_rag_tool first - it may find relevant content from previous sessions.")
        context_parts.append("If youtube_rag_tool returns 'No relevant information found', then:")
        context_parts.append("1) Use google_search_tool if available to answer the question, OR")
        context_parts.append("2) Provide a final answer explaining that no relevant video content was found.")
        context_parts.append("=" * 60)
        context_parts.append("")
    
    # Add conversation history if exists
    if history:
        context_parts.append("Previous conversation:")
        for msg in history[-6:]:  # Last 3 exchanges
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            # Don't truncate - agent needs full context to understand references
            content = msg.content  # Full content for proper context
            context_parts.append(f"{msg_type}: {content}")
        context_parts.append("")  # Empty line
    
    # Add video context
    if current_vid:
        context_parts.append(f"[Just processed YouTube video ID: {current_vid}]")
    elif session.current_video_id:
        context_parts.append(f"[Previously discussed video: {session.current_video_id}]")
    
    # Get current date/time for time-sensitive queries (same as in tool closure)
    current_time = datetime.utcnow()
    current_year = current_time.year
    current_date_str = current_time.strftime("%Y-%m-%d")
    
    # Add intent analysis hint to help agent understand query type
    time_sensitive_keywords = ["latest", "recent", "update", "new", "current", "2024", "2025", "now", "today"]
    is_time_sensitive = any(keyword in query.lower() for keyword in time_sensitive_keywords)
    
    if is_time_sensitive:
        context_parts.append("")
        context_parts.append("QUERY TYPE: TIME-SENSITIVE")
        context_parts.append("This question asks about recent/latest information.")
        context_parts.append(f"CURRENT DATE: {current_date_str} (Year: {current_year})")
        context_parts.append("CRITICAL: When calling google_search_tool, use the CURRENT YEAR in your search query.")
        context_parts.append(f"Example: If user asks about 'Python updates 2023', search for 'Python updates {current_year}' or 'latest Python updates {current_year}'")
        context_parts.append("Strategy: 1) Use youtube_rag_tool first, 2) Then use google_search_tool for current info, 3) Combine both in final answer.")
    else:
        context_parts.append("")
        context_parts.append("QUERY TYPE: GENERAL KNOWLEDGE")
        context_parts.append("This is a general question about a topic.")
        context_parts.append("Strategy: Use youtube_rag_tool ONCE. If it returns context, IMMEDIATELY provide final answer. DO NOT call it again.")
    
    context_parts.append(f"\nCurrent question: {query}")
    agent_input = "\n".join(context_parts)
    
    # Execute agent - agent should analyze tool results and decide next steps
    try:
        result = agent_executor.invoke({"input": agent_input})
        answer = result["output"]
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Check for looping behavior (agent calling same tool multiple times)
        youtube_rag_calls = sum(1 for step in intermediate_steps if len(step) >= 2 and "youtube_rag_tool" in str(step[0]))
        if youtube_rag_calls > 1:
            print(f"[WARN] Agent called youtube_rag_tool {youtube_rag_calls} times - this indicates looping behavior")
            # Extract context from first call and check instruction
            for i, step in enumerate(intermediate_steps):
                if len(step) >= 2:
                    tool_result = step[1]
                    if isinstance(tool_result, str) and "OBSERVATION:" in tool_result:
                        # Check for new structured format with emojis
                        if "✅ FINAL ANSWER REQUIRED ✅" in tool_result:
                            # General question - extract context and synthesize
                            if "Video context:" in tool_result:
                                found_context = tool_result.split("Video context:", 1)[1].strip()
                                print(f"[FIX] Agent looped but should have stopped. Synthesizing from first tool result.")
                                synthesis_chain = RunnableParallel({
                                    'question': RunnablePassthrough(),
                                    'context': RunnableLambda(lambda x: found_context)
                                }) | prompt | model | parser
                                answer = synthesis_chain.invoke(query)
                                intermediate_steps = intermediate_steps[:i+1]
                                break
                        elif "🚨 NEXT ACTION REQUIRED 🚨" in tool_result:
                            # Time-sensitive - should call google_search_tool
                            if "Video context preview:" in tool_result:
                                found_context = tool_result.split("Video context preview:", 1)[1].strip()
                                print(f"[FIX] Agent looped but should have called google_search_tool. Checking if google_search was called...")
                                # Check if google_search_tool was called after this
                                google_called = any(
                                    "google_search_tool" in str(step[0]) 
                                    for step in intermediate_steps[i+1:] 
                                    if len(step) >= 2
                                )
                                if not google_called:
                                    print(f"[FIX] google_search_tool was not called. This is a time-sensitive query - agent should have called it.")
                                    # For now, synthesize from video context only (fallback)
                                    synthesis_chain = RunnableParallel({
                                        'question': RunnablePassthrough(),
                                        'context': RunnableLambda(lambda x: found_context)
                                    }) | prompt | model | parser
                                    answer = synthesis_chain.invoke(query)
                                    intermediate_steps = intermediate_steps[:i+1]
                                    break
        
        # Additional check: If agent hit iteration limit, synthesize from available context
        if "Agent stopped" in answer or not answer:
            print(f"[FALLBACK] Agent hit iteration limit or didn't provide answer. Synthesizing from tool results...")
            # Collect all context from tool results
            all_contexts = []
            google_search_result = None
            
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_result = step[1]
                    if isinstance(tool_result, str):
                        # Extract video context from new format
                        if "Video context" in tool_result or "Video context preview" in tool_result:
                            # Extract context after "Video context" or "Video context preview"
                            if "Video context preview:" in tool_result:
                                ctx = tool_result.split("Video context preview:", 1)[1].strip()
                            elif "Video context:" in tool_result:
                                ctx = tool_result.split("Video context:", 1)[1].strip()
                            else:
                                ctx = None
                            if ctx:
                                all_contexts.append(ctx)
                        # Extract google search results
                        if "Title:" in tool_result and "Snippet:" in tool_result:
                            google_search_result = tool_result
            
            # Synthesize answer from collected contexts
            if all_contexts or google_search_result:
                combined_context = "\n\n".join(all_contexts)
                if google_search_result:
                    combined_context += f"\n\n=== Web Search Results ===\n{google_search_result}"
                
                print(f"[FALLBACK] Synthesizing answer from {len(all_contexts)} video context(s) and web search results")
                synthesis_chain = RunnableParallel({
                    'question': RunnablePassthrough(),
                    'context': RunnableLambda(lambda x: combined_context)
                }) | prompt | model | parser
                answer = synthesis_chain.invoke(query)
                print(f"[FALLBACK] Successfully synthesized answer from tool results")
        
        # Monitor agent execution: Track tool usage patterns
        tools_used = []
        tool_results = []
        tool_calls_with_inputs = []  # Track tool calls with their inputs
        
        for step in intermediate_steps:
            if len(step) >= 2:
                tool_action = step[0]
                tool_result = step[1]
                
                # Extract tool name and input from action
                tool_name = str(tool_action) if tool_action else "unknown"
                tool_input = ""
                if hasattr(tool_action, 'tool_input'):
                    tool_input = str(tool_action.tool_input)
                elif isinstance(tool_action, dict):
                    tool_input = str(tool_action.get('tool_input', ''))
                
                tools_used.append(tool_name)
                tool_results.append(tool_result)
                tool_calls_with_inputs.append((tool_name, tool_input))
        
        # Detect repeated tool calls with same input (indicates looping)
        if len(tool_calls_with_inputs) > 1:
            seen_calls = {}
            for tool_name, tool_input in tool_calls_with_inputs:
                key = f"{tool_name}:{tool_input}"
                if key in seen_calls:
                    seen_calls[key] += 1
                else:
                    seen_calls[key] = 1
            
            repeated_calls = {k: v for k, v in seen_calls.items() if v > 1}
            if repeated_calls:
                print(f"[MONITOR] WARNING: Repeated tool calls detected: {repeated_calls}")
                print(f"[MONITOR] Agent may be looping - consider this in failure detection")
        
        # Log tool usage
        if tools_used:
            unique_tools = list(set(tools_used))
            print(f"[MONITOR] Tools executed: {', '.join(unique_tools)}")
            print(f"[MONITOR] Total tool calls: {len(tools_used)}")
            
            # Log tool results summary
            for i, (tool_name, tool_result) in enumerate(zip(tools_used, tool_results)):
                if isinstance(tool_result, str):
                    result_preview = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    print(f"[MONITOR] Tool call {i+1} ({tool_name}): Result preview: {result_preview}")
                    if "No video context" in tool_result or "OBSERVATION: No video context" in tool_result:
                        print(f"[MONITOR] Tool call {i+1} ({tool_name}): Returned 'No video context' - agent should stop retrying")
                    elif "Relevant context" in tool_result:
                        print(f"[MONITOR] Tool call {i+1} ({tool_name}): Returned relevant context - agent should synthesize and provide final answer")
        else:
            print("[MONITOR] No tools were executed - agent may have used general knowledge")
            if current_vid or session.current_video_id:
                print("[MONITOR] WARNING: Video context exists but no tools were used!")
        
        # Check for various failure scenarios
        should_fallback = False
        failure_reason = None
        
        # Check 1: Agent stopped due to iteration limit or parsing errors
        if "Agent stopped" in answer or "Invalid Format" in answer or "not a valid tool" in answer:
            should_fallback = True
            failure_reason = "Agent stopped or parsing error"
        
        # Check 2: Agent looping on tool responses (both success and failure cases)
        no_video_context_count = 0
        context_found_count = 0
        if intermediate_steps:
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_result = step[1]
                    if isinstance(tool_result, str):
                        if "No video context" in tool_result or "OBSERVATION: No video context" in tool_result:
                            no_video_context_count += 1
                        elif "OBSERVATION: Found relevant context" in tool_result or "Relevant context" in tool_result:
                            context_found_count += 1
            
            # If agent called youtube_rag_tool multiple times with "No video context" response
            if no_video_context_count > 1:
                should_fallback = True
                failure_reason = f"Agent looping on 'No video context' ({no_video_context_count} times)"
                print(f"[FALLBACK] Detected looping: Agent called youtube_rag_tool {no_video_context_count} times despite 'No video context' response")
            
            # If agent got context but still called the tool again (looping on success)
            if context_found_count > 0 and len([s for s in intermediate_steps if len(s) >= 2 and "youtube_rag_tool" in str(s[0])]) > context_found_count:
                should_fallback = True
                failure_reason = f"Agent looping after receiving context ({context_found_count} context(s) found but {len([s for s in intermediate_steps if len(s) >= 2 and 'youtube_rag_tool' in str(s[0])])} tool calls)"
                print(f"[FALLBACK] Detected looping: Agent received context but continued calling youtube_rag_tool")
        
        # Check 3: Tool execution failures (tools were called but returned errors)
        if intermediate_steps:
            tool_failures = 0
            successful_tools = 0
            for step in intermediate_steps:
                if len(step) >= 2:
                    tool_result = step[1]
                    # Check if tool result indicates failure
                    if isinstance(tool_result, str):
                        if "Error" in tool_result or "failed" in tool_result.lower() or "not configured" in tool_result.lower():
                            tool_failures += 1
                        elif "No video context" not in tool_result and "OBSERVATION: No video context" not in tool_result:
                            # Don't count "No video context" as a failure if it's the first call
                            successful_tools += 1
            
            # If all tools failed or no tools succeeded (excluding first "No video context" call)
            if tool_failures > 0 and successful_tools == 0 and no_video_context_count <= 1:
                should_fallback = True
                failure_reason = f"All tools failed ({tool_failures} failures)"
        else:
            # Check 4: No tools were executed at all (agent answered without using tools)
            if current_vid or session.current_video_id:
                # If video context exists but no tools were used, agent likely used knowledge
                if not any("youtube_rag_tool" in str(step) for step in intermediate_steps):
                    should_fallback = True
                    failure_reason = "No tools executed despite video context"
        
        # Execute fallback RAG if needed
        if should_fallback and (current_vid or session.current_video_id):
            try:
                print(f"[FALLBACK] Agent failed ({failure_reason}), using direct RAG for query: {query}")
                index = get_shared_index()
                namespace = get_user_namespace(user.username)
                
                # Build query dict (AGENTS.md line 855-862: Dynamic filter pattern)
                query_dict = {
                    "top_k": 10 * 2,  # More candidates for reranking (AGENTS.md line 511)
                    "inputs": {"text": query}  # Text input (AGENTS.md line 512-513)
                }
                
                # Only add filter if it exists (AGENTS.md line 859-860)
                if current_vid or session.current_video_id:
                    query_dict["filter"] = {"video_id": current_vid or session.current_video_id}
                # Don't set filter to None - omit key entirely if no filter
                
                # Search with reranking (AGENTS.md line 508-522)
                results = exponential_backoff_retry(
                    lambda: index.search(
                        namespace=namespace,
                        query=query_dict,
                        rerank={
                            "model": "bge-reranker-v2-m3",  # AGENTS.md line 517
                            "top_n": 10,                    # Final results after reranking
                            "rank_fields": ["content"]      # AGENTS.md line 519
                        }
                    )
                )
                
                # Parse results with dict-style access (AGENTS.md line 221-230)
                raw_docs = [Document(
                    page_content=hit['fields']['content'],  # Dict access (AGENTS.md line 222)
                    metadata={
                        'video_id': hit['fields'].get('video_id'),  # Use .get() for optional (AGENTS.md line 230)
                        'doc_id': hit['fields'].get('doc_id'),
                        '_id': hit['_id'],
                        '_score': hit['_score']
                    })
                    for hit in results['result']['hits']]  # Dict access (AGENTS.md line 221)
                
                # Apply post-processing: score thresholding, diversity, context management
                docs = process_retrieval_results(
                    raw_docs,
                    min_score=MIN_SIMILARITY_SCORE,
                    max_per_video=MAX_CHUNKS_PER_VIDEO,
                    max_context_length=MAX_CONTEXT_LENGTH
                )
                
                if docs:
                    context = format_docs(docs, max_length=MAX_CONTEXT_LENGTH)
                    # Build fallback chain
                    fallback_chain = RunnableParallel({
                        'question': RunnablePassthrough(),
                        'context': RunnableLambda(lambda x: context)
                    }) | prompt | model | parser
                    answer = fallback_chain.invoke(query)
                    print(f"[FALLBACK] Successfully retrieved {len(docs)} documents and generated answer")
            except Exception as e:
                print(f"[FALLBACK] Error in fallback RAG: {e}")
                pass  # Use agent's answer even if fallback fails
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}"
        )
    
    # ========================================
    # STEP 6: Store Conversation in DB
    # ========================================
    
    # Get next message index
    last_msg = db.query(Message).filter(
        Message.session_id == session.session_id
    ).order_by(Message.message_index.desc()).first()
    
    next_index = (last_msg.message_index + 1) if last_msg else 0
    
    # Store user message
    user_msg = Message(
        session_id=session.session_id,
        message_index=next_index,
        message_type='human',
        content=query,
        video_id=current_vid
    )
    db.add(user_msg)
    
    # Store AI response
    ai_msg = Message(
        session_id=session.session_id,
        message_index=next_index + 1,
        message_type='ai',
        content=answer,
        video_id=current_vid
    )
    db.add(ai_msg)
    
    # Commit all changes (trigger will auto-cleanup old messages)
    db.commit()
    
    print(f"[INFO] Stored conversation (messages {next_index}-{next_index+1})")
    
    # ========================================
    # STEP 7: Return Response
    # ========================================
    
    return {
        "answer": answer,
        "session_id": str(session.session_id),
        "video_context": current_vid or session.current_video_id
    }



