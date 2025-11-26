"""drop_legacy_tables

Revision ID: b9f2b9f0d3a5
Revises: 471c628dd18d
Create Date: 2025-11-17 05:23:29.097821

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b9f2b9f0d3a5'
down_revision: Union[str, Sequence[str], None] = '471c628dd18d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop legacy tables and associated database objects."""
    
    # Step 1: Drop trigger first (depends on messages table)
    op.execute("DROP TRIGGER IF EXISTS trigger_sliding_window ON messages;")
    
    # Step 2: Drop functions (no longer needed)
    op.execute("DROP FUNCTION IF EXISTS maintain_sliding_window();")
    op.execute("DROP FUNCTION IF EXISTS cleanup_expired_sessions();")
    
    # Step 3: Drop messages table first (has foreign key to user_sessions)
    op.drop_table('messages')
    
    # Step 4: Drop user_sessions table
    op.drop_table('user_sessions')


def downgrade() -> None:
    """Recreate legacy tables (for rollback)."""
    
    # Recreate user_sessions table
    op.create_table('user_sessions',
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('current_video_id', sa.String(length=100), nullable=True),
        sa.Column('current_video_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_activity', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('session_id')
    )
    
    # Recreate messages table
    op.create_table('messages',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('message_index', sa.Integer(), nullable=False),
        sa.Column('message_type', sa.String(length=10), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('video_id', sa.String(length=100), nullable=True),
        sa.Column('tool_name', sa.String(length=100), nullable=True),
        sa.Column('tool_success', sa.Boolean(), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint("(message_type = 'tool' AND tool_name IS NOT NULL) OR (message_type != 'tool')", name='check_tool_fields'),
        sa.CheckConstraint("message_type IN ('human', 'ai', 'system', 'tool')", name='check_message_type'),
        sa.ForeignKeyConstraint(['session_id'], ['user_sessions.session_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Recreate indexes
    op.create_index('idx_sessions_user_active', 'user_sessions', ['user_id', 'is_active', 'last_activity'], postgresql_ops={'last_activity': 'DESC'})
    op.create_index('idx_sessions_cleanup', 'user_sessions', ['expires_at'], postgresql_where=sa.text('is_active = TRUE'))
    op.create_index('idx_messages_session_order', 'messages', ['session_id', 'message_index'], postgresql_ops={'message_index': 'DESC'})
    op.create_index('idx_messages_created', 'messages', ['created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_messages_video', 'messages', ['video_id'], postgresql_where=sa.text('video_id IS NOT NULL'))
    op.create_unique_constraint('uq_session_message_index', 'messages', ['session_id', 'message_index'])
    
    # Recreate functions and trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION maintain_sliding_window()
        RETURNS TRIGGER AS $$
        DECLARE
            window_size INTEGER := 20;
        BEGIN
            DELETE FROM messages
            WHERE session_id = NEW.session_id
            AND message_index <= (NEW.message_index - window_size);
            
            UPDATE user_sessions
            SET last_activity = NOW()
            WHERE session_id = NEW.session_id;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER trigger_sliding_window
        AFTER INSERT ON messages
        FOR EACH ROW
        EXECUTE FUNCTION maintain_sliding_window();
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            UPDATE user_sessions
            SET is_active = FALSE
            WHERE expires_at < NOW()
            AND is_active = TRUE;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            
            DELETE FROM user_sessions
            WHERE is_active = FALSE
            AND last_activity < (NOW() - INTERVAL '30 days');
            
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
