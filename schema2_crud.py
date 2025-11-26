"""
Schema 2 CRUD Operations for Augustus Chat

This module provides all database operations for video-scoped conversations
(Schema 2), implementing the write and read paths specified in DEVELOPMENT_PHASES.md
Phase 1.

Key Features:
- Video-scoped conversation management (per user+video)
- Transaction-safe message insertion with parent updates
- Efficient tab rendering with denormalized previews
- Support for both per-video and General tab patterns

Dependencies: database.py models (VideoConversation, ConversationMessage, VideosCatalog)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from sqlalchemy.exc import IntegrityError
import uuid
import logging
import json
import time
import os

from database import (
    VideoConversation,
    ConversationMessage,
    VideosCatalog,
    UUID_TYPE,
    uuid_default,
)

# Debug logging configuration
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
logger = logging.getLogger(__name__)


# ============================================================================
# WRITE PATH OPERATIONS
# ============================================================================


def get_or_create_conversation(
    db: Session,
    user_id: uuid.UUID,
    video_id: str,
    video_url: str,
    video_title: str
) -> VideoConversation:
    """
    Get existing conversation or create new one for (user, video) pair.
    
    This is the entry point for all per-video tab interactions.
    Uses UNIQUE constraint (user_id, video_id) to ensure one conversation per video.
    
    Args:
        db: Database session
        user_id: User UUID
        video_id: YouTube video ID (11 chars)
        video_url: Full YouTube URL
        video_title: Video title for preview
    
    Returns:
        VideoConversation object (existing or newly created)
    
    Raises:
        ValueError: If video_id is invalid format
    """
    if not video_id or len(video_id) > 11:
        raise ValueError(f"Invalid video_id: {video_id}")
    
    # Try to get existing conversation
    conversation = db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id,
        VideoConversation.video_id == video_id
    ).first()
    
    if conversation:
        return conversation
    
    # Create new conversation
    conversation = VideoConversation(
        conversation_id=uuid_default(),
        user_id=user_id,
        video_id=video_id,
        video_url=video_url,
        video_title=video_title[:500],  # Enforce length limit
        created_at=datetime.utcnow(),
        last_message_at=datetime.utcnow(),
        message_count=0,
        is_pinned=False
    )
    
    try:
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    except IntegrityError as e:
        # Handle race condition: another request created the same conversation
        db.rollback()
        # Retry fetch
        conversation = db.query(VideoConversation).filter(
            VideoConversation.user_id == user_id,
            VideoConversation.video_id == video_id
        ).first()
        if conversation:
            return conversation
        # If still fails, re-raise
        raise e


def insert_message_with_parent_update(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    video_id: str,
    role: str,
    content: str,
    tokens_estimate: Optional[int] = None
) -> ConversationMessage:
    """
    Insert a message and update parent conversation in a single transaction.
    
    This is the core write operation for Schema 2. It:
    1. Computes next message_index (auto-increment per conversation)
    2. Inserts message into conversation_messages
    3. Updates parent video_conversations (counters, previews, timestamp)
    
    All operations are atomic - if any step fails, the entire transaction rolls back.
    
    Args:
        db: Database session (must be in transaction)
        conversation_id: Parent conversation UUID
        user_id: User UUID (denormalized for direct queries)
        video_id: Video ID (denormalized for direct queries)
        role: Message role ('user' | 'assistant' | 'system')
        content: Message content (full text)
        tokens_estimate: Optional token count estimate
    
    Returns:
        ConversationMessage object (newly created)
    
    Raises:
        ValueError: If role is invalid or conversation not found
        IntegrityError: If database constraints violated
    """
    if role not in ('user', 'assistant', 'system'):
        raise ValueError(f"Invalid role: {role}")
    
    # Get parent conversation (with row lock to prevent race conditions)
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    ).with_for_update().first()
    
    if not conversation:
        raise ValueError(f"Conversation not found: {conversation_id}")
    
    # Compute next message_index (0-based)
    last_msg = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).order_by(desc(ConversationMessage.message_index)).first()
    
    next_index = (last_msg.message_index + 1) if last_msg else 0
    
    # Create message
    message = ConversationMessage(
        id=uuid_default(),
        conversation_id=conversation_id,
        user_id=user_id,
        video_id=video_id,
        message_index=next_index,
        role=role,
        content=content,
        content_length=len(content),
        tokens_estimate=tokens_estimate,
        created_at=datetime.utcnow()
    )
    
    # Update parent conversation
    conversation.message_count += 1
    conversation.last_message_at = datetime.utcnow()
    
    # Update preview fields (truncate to 200-300 chars per chatSchema.md)
    preview_length = 250  # Middle of 200-300 range
    if role == 'user':
        conversation.last_user_message = content[:preview_length]
    elif role == 'assistant':
        conversation.last_assistant_message = content[:preview_length]
    
    # Commit transaction (both message insert and parent update)
    db.add(message)
    db.commit()
    db.refresh(message)
    
    return message


def insert_conversation_turn(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    video_id: str,
    user_message: str,
    assistant_message: str,
    user_tokens: Optional[int] = None,
    assistant_tokens: Optional[int] = None
) -> Tuple[ConversationMessage, ConversationMessage]:
    """
    Insert a complete turn (user message + assistant reply) in a SINGLE atomic transaction.
    
    This ensures both messages are inserted together or neither is inserted,
    preventing orphaned user messages without assistant replies.
    
    Args:
        db: Database session
        conversation_id: Parent conversation UUID
        user_id: User UUID
        video_id: Video ID
        user_message: User's query/message
        assistant_message: AI assistant's response
        user_tokens: Optional token count for user message
        assistant_tokens: Optional token count for assistant message
    
    Returns:
        Tuple of (user_message_obj, assistant_message_obj)
    
    Raises:
        ValueError: If conversation not found or validation fails
    """
    try:
        # Get parent conversation with row lock (prevents race conditions)
        conversation = db.query(VideoConversation).filter(
            VideoConversation.conversation_id == conversation_id
        ).with_for_update().first()
        
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Compute next message_index (0-based)
        last_msg = db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).order_by(desc(ConversationMessage.message_index)).first()
        
        next_index = (last_msg.message_index + 1) if last_msg else 0
        
        # Create user message
        user_msg = ConversationMessage(
            id=uuid_default(),
            conversation_id=conversation_id,
            user_id=user_id,
            video_id=video_id,
            message_index=next_index,
            role='user',
            content=user_message,
            content_length=len(user_message),
            tokens_estimate=user_tokens,
            created_at=datetime.utcnow()
        )
        
        # Create assistant message (next index)
        assistant_msg = ConversationMessage(
            id=uuid_default(),
            conversation_id=conversation_id,
            user_id=user_id,
            video_id=video_id,
            message_index=next_index + 1,
            role='assistant',
            content=assistant_message,
            content_length=len(assistant_message),
            tokens_estimate=assistant_tokens,
            created_at=datetime.utcnow()
        )
        
        # Update parent conversation (both messages count)
        conversation.message_count += 2
        conversation.last_message_at = datetime.utcnow()
        
        # Update preview fields (truncate to 200-300 chars per chatSchema.md)
        preview_length = 250  # Middle of 200-300 range
        conversation.last_user_message = user_message[:preview_length]
        conversation.last_assistant_message = assistant_message[:preview_length]
        
        # Add both messages to session
        db.add(user_msg)
        db.add(assistant_msg)
        
        # Single commit for entire turn (atomic operation)
        db.commit()
        
        # Refresh to get database-generated fields
        db.refresh(user_msg)
        db.refresh(assistant_msg)
        
        return user_msg, assistant_msg
    
    except Exception as e:
        db.rollback()  # Now this actually works - rolls back entire turn
        raise e


# ============================================================================
# READ PATH OPERATIONS
# ============================================================================


def get_user_conversations_for_tabs(
    db: Session,
    user_id: uuid.UUID,
    limit: int = 5
) -> List[VideoConversation]:
    """
    Get conversations for tab rendering (pinned + most recent unpinned).
    
    Returns conversations ordered by (is_pinned DESC, last_message_at DESC).
    This uses the composite index created in Phase 0 for optimal performance.
    
    Per plans.md: Show pinned + 5 most recent unpinned.
    
    Args:
        db: Database session
        user_id: User UUID
        limit: Maximum number of unpinned conversations to return (default: 5)
    
    Returns:
        List of VideoConversation objects with preview data
    """
    # Get all pinned conversations
    pinned = db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id,
        VideoConversation.is_pinned == True
    ).order_by(desc(VideoConversation.last_message_at)).all()
    
    # Get most recent unpinned conversations
    unpinned = db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id,
        VideoConversation.is_pinned == False
    ).order_by(desc(VideoConversation.last_message_at)).limit(limit).all()
    
    # Combine: pinned first, then unpinned by recency
    return pinned + unpinned


def get_conversation_by_id(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None
) -> Optional[VideoConversation]:
    """
    Get a single conversation by ID.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        user_id: Optional user UUID for authorization check
    
    Returns:
        VideoConversation object or None if not found
    """
    # Debug logging (Phase 1.4): Log query parameters and execution
    query_start = time.time()
    
    if DEBUG_LOGGING:
        log_data = {
            "event": "get_conversation_by_id",
            "conversation_id": str(conversation_id),
            "user_id_filter": str(user_id) if user_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"[DB] Query conversation: {json.dumps(log_data)}")
    
    query = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    )
    
    if user_id:
        query = query.filter(VideoConversation.user_id == user_id)
    
    result = query.first()
    query_time = (time.time() - query_start) * 1000  # Convert to ms
    
    # Debug logging (Phase 1.4): Log query result and performance
    if DEBUG_LOGGING or query_time > 100:  # Log if slow query (>100ms)
        log_data = {
            "event": "get_conversation_by_id_result",
            "conversation_id": str(conversation_id),
            "user_id_filter": str(user_id) if user_id else None,
            "conversation_found": result is not None,
            "query_time_ms": round(query_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result:
            log_data["conversation_user_id"] = str(result.user_id)
            log_data["video_id"] = result.video_id
            log_data["video_title"] = result.video_title[:100] if result.video_title else None  # Truncate for logging
            log_data["message_count"] = result.message_count
        
        if query_time > 100:
            logger.warning(f"[DB] Slow query detected: {json.dumps(log_data)}")
        else:
            logger.debug(f"[DB] Query result: {json.dumps(log_data)}")
    
    return result


def get_conversation_messages(
    db: Session,
    conversation_id: uuid.UUID,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[ConversationMessage]:
    """
    Get messages for a conversation in chronological order.
    
    Returns messages ordered by message_index ASC (chronological).
    Uses composite index (conversation_id, message_index) for performance.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        limit: Optional limit for pagination
        offset: Optional offset for pagination (by message_index)
    
    Returns:
        List of ConversationMessage objects in chronological order
    """
    query = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).order_by(ConversationMessage.message_index)
    
    if offset is not None:
        query = query.offset(offset)
    
    if limit is not None:
        query = query.limit(limit)
    
    return query.all()


def get_recent_messages_for_context(
    db: Session,
    conversation_id: uuid.UUID,
    limit: int = 10
) -> List[ConversationMessage]:
    """
    Get most recent messages for agent context (LLM history).
    
    Returns messages in chronological order (oldest to newest) for LLM.
    This is used to build conversation history for the agent.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        limit: Number of recent messages to retrieve (default: 10)
    
    Returns:
        List of ConversationMessage objects in chronological order
    """
    # Fetch latest messages in descending order
    messages = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).order_by(desc(ConversationMessage.message_index)).limit(limit).all()
    
    # Reverse to get chronological order (oldest to newest)
    messages.reverse()
    
    return messages


# ============================================================================
# PINNING OPERATIONS
# ============================================================================


def pin_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> VideoConversation:
    """
    Pin a conversation (exempt from retention policy).
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        user_id: User UUID (for authorization)
    
    Returns:
        Updated VideoConversation object
    
    Raises:
        ValueError: If conversation not found or unauthorized
    """
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id,
        VideoConversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise ValueError(f"Conversation not found or unauthorized: {conversation_id}")
    
    conversation.is_pinned = True
    db.commit()
    db.refresh(conversation)
    
    return conversation


def unpin_conversation(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> VideoConversation:
    """
    Unpin a conversation (subject to retention policy).
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        user_id: User UUID (for authorization)
    
    Returns:
        Updated VideoConversation object
    
    Raises:
        ValueError: If conversation not found or unauthorized
    """
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id,
        VideoConversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise ValueError(f"Conversation not found or unauthorized: {conversation_id}")
    
    conversation.is_pinned = False
    db.commit()
    db.refresh(conversation)
    
    return conversation


# ============================================================================
# VIDEOS CATALOG OPERATIONS
# ============================================================================


def get_or_create_catalog_entry(
    db: Session,
    video_id: str,
    video_title: str,
    video_url: str,
    synopsis: str = "",
    topics: List[str] = None,
    channel_title: Optional[str] = None,
    published_at: Optional[datetime] = None,
    embedding_version: Optional[str] = None
) -> VideosCatalog:
    """
    Get existing catalog entry or create new one.
    
    This is used during video embedding to ensure catalog metadata exists.
    
    Args:
        db: Database session
        video_id: YouTube video ID (11 chars, PK)
        video_title: Video title
        video_url: Full YouTube URL
        synopsis: Video synopsis (250-300 chars for General tab routing)
        topics: List of topic keywords
        channel_title: Optional channel name
        published_at: Optional publish date
        embedding_version: Optional embedding model version
    
    Returns:
        VideosCatalog object (existing or newly created)
    """
    if topics is None:
        topics = []
    
    # Try to get existing entry
    catalog = db.query(VideosCatalog).filter(
        VideosCatalog.video_id == video_id
    ).first()
    
    if catalog:
        return catalog
    
    # Create new entry
    catalog = VideosCatalog(
        video_id=video_id,
        video_title=video_title[:500],
        video_url=video_url,
        synopsis=synopsis[:300],  # Enforce length limit
        topics=topics,
        channel_title=channel_title[:300] if channel_title else None,
        published_at=published_at,
        embedding_version=embedding_version,
        last_refreshed_at=datetime.utcnow()
    )
    
    try:
        db.add(catalog)
        db.commit()
        db.refresh(catalog)
        return catalog
    except IntegrityError as e:
        # Handle race condition
        db.rollback()
        catalog = db.query(VideosCatalog).filter(
            VideosCatalog.video_id == video_id
        ).first()
        if catalog:
            return catalog
        raise e


def update_catalog_entry(
    db: Session,
    video_id: str,
    synopsis: Optional[str] = None,
    topics: Optional[List[str]] = None,
    video_title: Optional[str] = None,
    embedding_version: Optional[str] = None
) -> Optional[VideosCatalog]:
    """
    Update existing catalog entry (e.g., refresh synopsis/topics).
    
    Args:
        db: Database session
        video_id: YouTube video ID
        synopsis: Optional new synopsis
        topics: Optional new topics list
        video_title: Optional new title
        embedding_version: Optional new embedding version
    
    Returns:
        Updated VideosCatalog object or None if not found
    """
    catalog = db.query(VideosCatalog).filter(
        VideosCatalog.video_id == video_id
    ).first()
    
    if not catalog:
        return None
    
    if synopsis is not None:
        catalog.synopsis = synopsis[:300]
    if topics is not None:
        catalog.topics = topics
    if video_title is not None:
        catalog.video_title = video_title[:500]
    if embedding_version is not None:
        catalog.embedding_version = embedding_version
        catalog.last_embedded_at = datetime.utcnow()
    
    catalog.last_refreshed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(catalog)
    
    return catalog


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_conversation_by_user_and_video(
    db: Session,
    user_id: uuid.UUID,
    video_id: str
) -> Optional[VideoConversation]:
    """
    Get conversation by (user_id, video_id) pair.
    
    This is useful for checking if a conversation exists before creating.
    
    Args:
        db: Database session
        user_id: User UUID
        video_id: YouTube video ID
    
    Returns:
        VideoConversation object or None if not found
    """
    return db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id,
        VideoConversation.video_id == video_id
    ).first()


def get_message_count(
    db: Session,
    conversation_id: uuid.UUID
) -> int:
    """
    Get total message count for a conversation.
    
    This can be used to verify the denormalized message_count field.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
    
    Returns:
        Integer count of messages
    """
    return db.query(func.count(ConversationMessage.id)).filter(
        ConversationMessage.conversation_id == conversation_id
    ).scalar()


def serialize_conversation_for_api(conversation: VideoConversation) -> Dict[str, Any]:
    """
    Serialize VideoConversation object for API response.
    
    Args:
        conversation: VideoConversation object
    
    Returns:
        Dictionary with conversation data for frontend
    """
    return {
        "conversation_id": str(conversation.conversation_id),
        "video_id": conversation.video_id,
        "video_url": conversation.video_url,
        "video_title": conversation.video_title,
        "created_at": conversation.created_at.isoformat(),
        "last_message_at": conversation.last_message_at.isoformat(),
        "message_count": conversation.message_count,
        "last_user_message": conversation.last_user_message,
        "last_assistant_message": conversation.last_assistant_message,
        "is_pinned": conversation.is_pinned
    }


def serialize_message_for_api(message: ConversationMessage) -> Dict[str, Any]:
    """
    Serialize ConversationMessage object for API response.
    
    Args:
        message: ConversationMessage object
    
    Returns:
        Dictionary with message data for frontend
    """
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "user_id": str(message.user_id),
        "video_id": message.video_id,
        "message_index": message.message_index,
        "role": message.role,
        "content": message.content,
        "content_length": message.content_length,
        "tokens_estimate": message.tokens_estimate,
        "created_at": message.created_at.isoformat()
    }


# ============================================================================
# OWNERSHIP VERIFICATION UTILITIES (Phase 7)
# ============================================================================


def verify_conversation_ownership(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> Tuple[bool, Optional[str]]:
    """
    Verify user owns conversation. Returns (is_owner, error_message)
    
    Phase 7.16: Utility function for ownership verification.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID to check
        user_id: User UUID to verify ownership
    
    Returns:
        Tuple of (is_owner: bool, error_message: Optional[str])
        - (True, None) if user owns the conversation
        - (False, "Conversation not found") if conversation doesn't exist
        - (False, "Access denied: ...") if conversation exists but user doesn't own it
    """
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        return (False, "Conversation not found")
    
    if conversation.user_id != user_id:
        return (
            False, 
            f"Access denied: Conversation owned by user_id {conversation.user_id}, but current user is {user_id}"
        )
    
    return (True, None)


def get_user_conversation_ids(db: Session, user_id: uuid.UUID) -> List[uuid.UUID]:
    """
    Get all conversation_ids for a user (for frontend validation).
    
    Phase 7.17: Utility function to get all conversation IDs for a user.
    Useful for frontend validation to ensure only valid conversation_ids are sent.
    
    Args:
        db: Database session
        user_id: User UUID
    
    Returns:
        List of conversation UUIDs owned by the user
    """
    conversations = db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id
    ).all()
    return [c.conversation_id for c in conversations]

