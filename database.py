"""
Database configuration and models for the FastAPI authentication system

Setup Instructions:
1. Install PostgreSQL locally:
   - macOS: brew install postgresql@15
   - Ubuntu: sudo apt-get install postgresql postgresql-contrib
   - Windows: Download from https://www.postgresql.org/download/

2. Start PostgreSQL service:
   - macOS: brew services start postgresql@15
   - Ubuntu: sudo systemctl start postgresql
   - Windows: Use PostgreSQL service manager

3. Create the database:
   - psql postgres
   - CREATE DATABASE fastapi_auth;
   - CREATE USER postgres WITH PASSWORD 'your_password';
   - GRANT ALL PRIVILEGES ON DATABASE fastapi_auth TO postgres;

4. Set environment variable in .env file:
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/fastapi_auth

5. Run migrations:
   alembic upgrade head
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
from sqlalchemy.engine import make_url
from sqlalchemy import JSON

# Database URL from environment variable
# MUST be set - no default to prevent deployment errors
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "[ERROR] DATABASE_URL environment variable is required!\n"
        "   Set it in .env file for your database connection.\n"
        "   Example: postgresql://user:pass@host:5432/dbname?sslmode=require"
    )

parsed_url = make_url(DATABASE_URL)
DB_BACKEND = parsed_url.get_backend_name()
IS_SQLITE = DB_BACKEND == "sqlite"

if IS_SQLITE:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    UUID_TYPE = String(36)
    EXTRA_DATA_TYPE = JSON
    SCOPES_TYPE = JSON

    def uuid_default() -> str:
        return str(uuid.uuid4())
else:
    from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

    # Production-ready engine configuration
    # Optimized for NeonDB and cloud databases with auto-suspend
    IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

    engine = create_engine(
        DATABASE_URL,
        echo=not IS_PRODUCTION,  # Only log SQL in development
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Max overflow connections
        pool_pre_ping=True,  # Test connections before use
        pool_recycle=300,  # Recycle connections after 5 minutes
        connect_args={
            "connect_timeout": 10,
            "options": "-c timezone=utc",
        },
    )

    UUID_TYPE = UUID(as_uuid=True)
    EXTRA_DATA_TYPE = JSONB
    SCOPES_TYPE = ARRAY(String)

    def uuid_default():
        return uuid.uuid4()

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

class User(Base):
    """User model for authentication system"""
    __tablename__ = "users"
    
    id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    scopes = Column(SCOPES_TYPE, default=lambda: ["read", "write"], nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class VideoConversation(Base):
    """
    Video conversation model (Schema 2 - parent table).
    One row per (user, video) conversation for fast tab rendering.
    """
    __tablename__ = "video_conversations"
    
    # Primary key
    conversation_id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    
    # Foreign key to users
    user_id = Column(UUID_TYPE, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Video identification
    video_id = Column(String(11), nullable=False, index=True)
    video_url = Column(Text, nullable=False)
    video_title = Column(String(500), nullable=False)
    
    # Timestamps (timezone=True for PostgreSQL TIMESTAMPTZ)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    last_message_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    # Denormalized previews and counters
    message_count = Column(Integer, default=0, nullable=False)
    last_user_message = Column(Text, nullable=True)
    last_assistant_message = Column(Text, nullable=True)
    
    # Retention control
    is_pinned = Column(Boolean, default=False, nullable=False, index=True)
    
    # Optional: General tab routing cache
    synopsis_preview = Column(Text, nullable=True)
    
    # Constraints
    __table_args__ = (
        sa.UniqueConstraint('user_id', 'video_id', name='uq_user_video'),
    )
    
    def __repr__(self):
        return f"<VideoConversation(conversation_id='{self.conversation_id}', user_id='{self.user_id}', video_id='{self.video_id}')>"


class ConversationMessage(Base):
    """
    Conversation message model (Schema 2 - child table).
    Stores all messages for a conversation in chronological order.
    """
    __tablename__ = "conversation_messages"
    
    # Primary key
    id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    
    # Foreign key to parent conversation
    conversation_id = Column(
        UUID_TYPE, 
        ForeignKey('video_conversations.conversation_id', ondelete='CASCADE'), 
        nullable=False
    )
    
    # Denormalized fields (for direct queries without joins)
    user_id = Column(UUID_TYPE, nullable=False, index=True)
    video_id = Column(String(11), nullable=False, index=True)
    
    # Message ordering
    message_index = Column(Integer, nullable=False)
    
    # Message content
    role = Column(String(10), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    tokens_estimate = Column(Integer, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        sa.UniqueConstraint('conversation_id', 'message_index', name='uq_conversation_message_index'),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name='check_role_valid'
        ),
    )
    
    def __repr__(self):
        return f"<ConversationMessage(id='{self.id}', conversation_id='{self.conversation_id}', role='{self.role}', index={self.message_index})>"


class VideosCatalog(Base):
    """
    Videos catalog model (authoritative per-video metadata).
    Single source of truth for video synopsis and topics used in General tab routing.
    """
    __tablename__ = "videos_catalog"
    
    # Primary key
    video_id = Column(String(11), primary_key=True)
    
    # Video metadata
    video_title = Column(String(500), nullable=False)
    channel_title = Column(String(300), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    video_url = Column(Text, nullable=False)
    
    # Routing metadata
    synopsis = Column(Text, nullable=False)
    topics = Column(ARRAY(String) if not IS_SQLITE else JSON, nullable=False)
    
    # Embedding tracking (optional)
    embedding_version = Column(String(50), nullable=True)
    last_embedded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Refresh tracking
    last_refreshed_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<VideosCatalog(video_id='{self.video_id}', title='{self.video_title[:50]}...')>"


# Dependency to get database session
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Drop all tables (for testing)
def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

