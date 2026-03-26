"""
Main FastAPI application for the Multi-Source RAG Chatbot.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from app.ingestion.loader import DocumentLoader
from app.rag.pipeline import RAGPipeline
from app.memory.chat_history import ChatHistory
from app.api.routes import router, init_dependencies

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Settings
class Settings(BaseSettings):
    """Application settings."""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ChromaDB
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    # App
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))

    # LLM
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))

    # RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))


settings = Settings()


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""

    logger.info("Starting RAG Chatbot application...")

    # Validate OpenAI API key
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        logger.error("OPENAI_API_KEY not set! Please set it in .env file")
        raise ValueError("OPENAI_API_KEY is required")

    # Initialize components
    logger.info("Initializing components...")

    # Document loader
    document_loader = DocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    # RAG pipeline
    rag_pipeline = RAGPipeline(
        openai_api_key=settings.openai_api_key,
        chroma_host=settings.chroma_host,
        chroma_port=settings.chroma_port,
        collection_name=settings.chroma_collection_name,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_k=settings.top_k_retrieval
    )

    # Chat history
    chat_history = ChatHistory(storage_dir="sessions", max_history_length=10)

    # Initialize route dependencies
    init_dependencies(document_loader, rag_pipeline, chat_history)

    logger.info("Application started successfully!")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Multi-Source RAG Chatbot",
    description="An intelligent chatbot with RAG capabilities supporting multiple knowledge sources",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("app/frontend/index.html")


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Multi-Source RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload_pdf": "POST /api/upload/pdf",
            "upload_csv": "POST /api/upload/csv",
            "upload_url": "POST /api/upload/url",
            "chat": "POST /api/chat",
            "get_history": "GET /api/history/{session_id}",
            "clear_history": "DELETE /api/history/{session_id}",
            "get_sources": "GET /api/sources",
            "list_sessions": "GET /api/sessions",
            "health": "GET /api/health"
        }
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.app_host}:{settings.app_port}")

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level="info"
    )
