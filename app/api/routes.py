"""
FastAPI routes for the RAG chatbot.
Provides endpoints for document upload, chat, and history management.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl

from app.ingestion.loader import DocumentLoader, save_upload_file
from app.rag.pipeline import RAGPipeline
from app.memory.chat_history import ChatHistory

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global instances (will be initialized in main.py)
document_loader: Optional[DocumentLoader] = None
rag_pipeline: Optional[RAGPipeline] = None
chat_history: Optional[ChatHistory] = None


def init_dependencies(loader: DocumentLoader, pipeline: RAGPipeline, history: ChatHistory):
    """Initialize route dependencies."""
    global document_loader, rag_pipeline, chat_history
    document_loader = loader
    rag_pipeline = pipeline
    chat_history = history


# Request/Response Models
class URLUploadRequest(BaseModel):
    url: HttpUrl
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    sources: list
    session_id: str
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list
    total_messages: int


# Routes

@router.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Upload and ingest a PDF file.

    Args:
        file: PDF file to upload
        session_id: Optional session ID for tracking

    Returns:
        Ingestion result with document count
    """
    try:
        logger.info(f"Received PDF upload: {file.filename}")

        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save uploaded file
        upload_dir = "data/uploads"
        file_path = await save_upload_file(file, upload_dir)

        # Load and process document
        documents = await document_loader.load_pdf(file_path)

        # Add to RAG pipeline
        result = await rag_pipeline.add_documents(documents)

        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "file_name": file.filename,
            "chunks_added": len(documents),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Upload and ingest a CSV file.

    Args:
        file: CSV file to upload
        session_id: Optional session ID for tracking

    Returns:
        Ingestion result with document count
    """
    try:
        logger.info(f"Received CSV upload: {file.filename}")

        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Save uploaded file
        upload_dir = "data/uploads"
        file_path = await save_upload_file(file, upload_dir)

        # Load and process document
        documents = await document_loader.load_csv(file_path)

        # Add to RAG pipeline
        result = await rag_pipeline.add_documents(documents)

        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "file_name": file.filename,
            "chunks_added": len(documents),
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


@router.post("/upload/url")
async def upload_url(request: URLUploadRequest):
    """
    Ingest content from a web URL.

    Args:
        request: URLUploadRequest with URL and optional session ID

    Returns:
        Ingestion result with document count
    """
    try:
        url = str(request.url)
        logger.info(f"Received URL upload: {url}")

        # Load and process document
        documents = await document_loader.load_url(url)

        # Add to RAG pipeline
        result = await rag_pipeline.add_documents(documents)

        return {
            "status": "success",
            "message": f"Successfully processed URL: {url}",
            "url": url,
            "chunks_added": len(documents),
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error uploading URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message and get a response from the RAG system.

    Args:
        request: ChatRequest with message, session_id, and stream flag

    Returns:
        Chat response with answer and sources (or streaming response)
    """
    try:
        logger.info(f"Received chat message from session {request.session_id}")

        # Add user message to history
        await chat_history.add_message(
            session_id=request.session_id,
            role="user",
            content=request.message
        )

        # Get conversation context
        history = await chat_history.get_history(request.session_id, limit=5)

        # Handle streaming response
        if request.stream:
            async def generate_stream():
                full_response = ""
                async for token in rag_pipeline.query_stream(request.message, history):
                    full_response += token
                    yield token

                # Save assistant response to history after streaming completes
                await chat_history.add_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=full_response
                )

            return StreamingResponse(generate_stream(), media_type="text/plain")

        # Non-streaming response
        result = await rag_pipeline.query(request.message, history)

        # Add assistant response to history
        await chat_history.add_message(
            session_id=request.session_id,
            role="assistant",
            content=result["response"],
            metadata={"sources": result.get("sources", [])}
        )

        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: Optional[int] = None):
    """
    Get conversation history for a session.

    Args:
        session_id: Session identifier
        limit: Optional limit on number of messages

    Returns:
        Conversation history
    """
    try:
        history = await chat_history.get_history(session_id, limit)

        return {
            "session_id": session_id,
            "messages": history,
            "total_messages": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    Clear conversation history for a session.

    Args:
        session_id: Session identifier

    Returns:
        Operation result
    """
    try:
        result = await chat_history.clear_history(session_id)

        return {
            "status": "success",
            "message": f"History cleared for session {session_id}",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@router.get("/sources")
async def get_sources():
    """
    Get information about all ingested documents.

    Returns:
        List of sources and document counts
    """
    try:
        sources_info = rag_pipeline.get_sources_info()

        return {
            "status": "success",
            "data": sources_info,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sources: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "rag-chatbot",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions.

    Returns:
        List of session information
    """
    try:
        sessions = await chat_history.list_sessions()

        return {
            "status": "success",
            "sessions": sessions,
            "total_sessions": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
