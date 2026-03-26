"""
Conversation memory management for session-based chat history.
Stores and retrieves chat history per session.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChatHistory:
    """Manages conversation history for multiple sessions."""

    def __init__(
        self,
        storage_dir: str = "sessions",
        max_history_length: int = 10
    ):
        """
        Initialize chat history manager.

        Args:
            storage_dir: Directory to store session files
            max_history_length: Maximum number of messages to keep per session
        """
        self.storage_dir = Path(storage_dir)
        self.max_history_length = max_history_length
        self._in_memory_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Chat history initialized with storage at {self.storage_dir}")

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.storage_dir / f"{session_id}.json"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to the conversation history.

        Args:
            session_id: Unique session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (e.g., sources, timestamp)

        Returns:
            Dictionary with operation result
        """
        try:
            # Create message object
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            # Load existing history
            history = await self.get_history(session_id)

            # Add new message
            history.append(message)

            # Trim history if it exceeds max length
            if len(history) > self.max_history_length:
                history = history[-self.max_history_length:]

            # Update cache
            self._in_memory_cache[session_id] = history

            # Save to disk asynchronously
            await self._save_to_disk(session_id, history)

            logger.info(f"Added {role} message to session {session_id}")

            return {
                "status": "success",
                "message": "Message added to history",
                "session_id": session_id,
                "history_length": len(history)
            }

        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to add message: {str(e)}"
            }

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Unique session identifier
            limit: Optional limit on number of messages to return

        Returns:
            List of message dictionaries
        """
        try:
            # Check cache first
            if session_id in self._in_memory_cache:
                history = self._in_memory_cache[session_id]
            else:
                # Load from disk
                history = await self._load_from_disk(session_id)
                self._in_memory_cache[session_id] = history

            # Apply limit if specified
            if limit is not None:
                history = history[-limit:]

            return history

        except Exception as e:
            logger.error(f"Error getting history for session {session_id}: {str(e)}")
            return []

    async def get_history_text(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> str:
        """
        Get conversation history as formatted text.

        Args:
            session_id: Unique session identifier
            limit: Optional limit on number of messages

        Returns:
            Formatted conversation history as string
        """
        history = await self.get_history(session_id, limit)

        formatted_messages = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_messages.append(f"{role.upper()}: {content}")

        return "\n\n".join(formatted_messages)

    async def clear_history(self, session_id: str) -> Dict[str, Any]:
        """
        Clear conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with operation result
        """
        try:
            # Clear cache
            if session_id in self._in_memory_cache:
                del self._in_memory_cache[session_id]

            # Delete file
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()

            logger.info(f"Cleared history for session {session_id}")

            return {
                "status": "success",
                "message": f"History cleared for session {session_id}"
            }

        except Exception as e:
            logger.error(f"Error clearing history for session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to clear history: {str(e)}"
            }

    async def _save_to_disk(self, session_id: str, history: List[Dict[str, Any]]):
        """Save history to disk asynchronously."""
        try:
            session_file = self._get_session_file(session_id)

            # Use asyncio to run blocking I/O in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_json_file,
                session_file,
                history
            )

        except Exception as e:
            logger.error(f"Error saving session {session_id} to disk: {str(e)}")

    def _write_json_file(self, file_path: Path, data: Any):
        """Blocking JSON write operation."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def _load_from_disk(self, session_id: str) -> List[Dict[str, Any]]:
        """Load history from disk asynchronously."""
        try:
            session_file = self._get_session_file(session_id)

            if not session_file.exists():
                return []

            # Use asyncio to run blocking I/O in thread pool
            loop = asyncio.get_event_loop()
            history = await loop.run_in_executor(
                None,
                self._read_json_file,
                session_file
            )

            return history

        except Exception as e:
            logger.error(f"Error loading session {session_id} from disk: {str(e)}")
            return []

    def _read_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Blocking JSON read operation."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.

        Returns:
            List of session information dictionaries
        """
        try:
            sessions = []

            # Get all session files
            for session_file in self.storage_dir.glob("*.json"):
                session_id = session_file.stem

                # Get history length
                history = await self.get_history(session_id)

                # Get last message timestamp
                last_message = history[-1] if history else None
                last_timestamp = last_message.get('timestamp') if last_message else None

                sessions.append({
                    "session_id": session_id,
                    "message_count": len(history),
                    "last_message_time": last_timestamp
                })

            return sessions

        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []

    async def get_context_for_query(
        self,
        session_id: str,
        max_messages: int = 5
    ) -> str:
        """
        Get formatted context from recent conversation for RAG query.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted context string
        """
        history = await self.get_history(session_id, limit=max_messages)

        if not history:
            return ""

        context_parts = ["Previous conversation:"]
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)
