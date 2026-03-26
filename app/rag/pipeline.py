"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
Integrates LlamaIndex with ChromaDB for vector storage and retrieval.
"""

import os
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline with ChromaDB vector store and OpenAI LLM."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = "rag_documents",
        llm_model: str = "gpt-4",
        embedding_model: str = "text-embedding-ada-002",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.

        Args:
            openai_api_key: OpenAI API key
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
            collection_name: Name of ChromaDB collection
            llm_model: OpenAI LLM model name
            embedding_model: OpenAI embedding model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM response
            top_k: Number of top documents to retrieve
        """
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        self.top_k = top_k

        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key
        )

        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=openai_api_key
        )

        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Initialize vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Initialize index (will be None until documents are added)
        self.index: Optional[VectorStoreIndex] = None
        self._load_or_create_index()

        logger.info("RAG pipeline initialized successfully")

    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            # Try to load existing index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            logger.info("Loaded existing vector index")
        except Exception as e:
            logger.info("No existing index found, will create on first document add")
            self.index = None

    async def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            Dictionary with ingestion results
        """
        try:
            if not documents:
                return {
                    "status": "error",
                    "message": "No documents provided",
                    "documents_added": 0
                }

            logger.info(f"Adding {len(documents)} documents to index")

            if self.index is None:
                # Create new index
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context
                )
                logger.info("Created new vector index")
            else:
                # Add to existing index
                for doc in documents:
                    self.index.insert(doc)
                logger.info(f"Added documents to existing index")

            return {
                "status": "success",
                "message": f"Successfully added {len(documents)} document chunks",
                "documents_added": len(documents),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to add documents: {str(e)}",
                "documents_added": 0
            }

    async def query(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system and get a response.

        Args:
            query_text: User's query
            chat_history: Optional conversation history

        Returns:
            Dictionary with response and sources
        """
        try:
            if self.index is None:
                return {
                    "response": "No documents have been indexed yet. Please upload some documents first.",
                    "sources": [],
                    "status": "error"
                }

            logger.info(f"Processing query: {query_text[:100]}...")

            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.top_k
            )

            # Execute query
            response = await query_engine.aquery(query_text)

            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(node.score) if hasattr(node, 'score') else None,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    })

            return {
                "response": str(response),
                "sources": sources,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "status": "error"
            }

    async def query_stream(
        self,
        query_text: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream query response token by token.

        Args:
            query_text: User's query
            chat_history: Optional conversation history

        Yields:
            Response tokens as they're generated
        """
        try:
            if self.index is None:
                yield "No documents have been indexed yet. Please upload some documents first."
                return

            logger.info(f"Processing streaming query: {query_text[:100]}...")

            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.top_k,
                streaming=True
            )

            # Execute query
            response = await query_engine.aquery(query_text)

            # Stream response
            if hasattr(response, 'response_gen'):
                for token in response.response_gen:
                    yield token
            else:
                yield str(response)

        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield f"Error: {str(e)}"

    def get_sources_info(self) -> Dict[str, Any]:
        """
        Get information about indexed documents.

        Returns:
            Dictionary with source information
        """
        try:
            # Get collection data
            collection_data = self.collection.get()

            # Extract unique sources
            sources = {}
            if collection_data and collection_data.get('metadatas'):
                for metadata in collection_data['metadatas']:
                    source = metadata.get('source', 'unknown')
                    source_type = metadata.get('source_type', 'unknown')

                    if source not in sources:
                        sources[source] = {
                            "source": source,
                            "source_type": source_type,
                            "chunk_count": 0
                        }
                    sources[source]["chunk_count"] += 1

            return {
                "total_chunks": len(collection_data.get('ids', [])) if collection_data else 0,
                "sources": list(sources.values()),
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Error getting sources info: {str(e)}")
            return {
                "total_chunks": 0,
                "sources": [],
                "error": str(e)
            }

    async def clear_index(self) -> Dict[str, Any]:
        """
        Clear all documents from the index.

        Returns:
            Dictionary with operation result
        """
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name
            )

            # Reset vector store and index
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = None

            logger.info("Index cleared successfully")

            return {
                "status": "success",
                "message": "All documents cleared from index"
            }

        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to clear index: {str(e)}"
            }
