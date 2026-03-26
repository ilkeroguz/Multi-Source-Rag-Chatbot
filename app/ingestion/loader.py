"""
Document ingestion module for multi-source RAG chatbot.
Supports PDF, Web URLs, and CSV files.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, CSVReader
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading and processing documents from multiple sources."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects with chunked content
        """
        try:
            logger.info(f"Loading PDF from {file_path}")

            # Load PDF using LlamaIndex PDFReader
            reader = PDFReader()
            documents = reader.load_data(file=Path(file_path))

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "source_type": "pdf",
                    "file_name": os.path.basename(file_path)
                })

            # Split into chunks
            nodes = self.text_splitter.get_nodes_from_documents(documents)

            # Convert nodes back to documents
            chunked_docs = [
                Document(
                    text=node.text,
                    metadata=node.metadata
                )
                for node in nodes
            ]

            logger.info(f"Successfully loaded {len(chunked_docs)} chunks from PDF")
            return chunked_docs

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise Exception(f"Failed to load PDF: {str(e)}")

    async def load_url(self, url: str) -> List[Document]:
        """
        Load and process content from a web URL.

        Args:
            url: Web URL to fetch content from

        Returns:
            List of Document objects with chunked content
        """
        try:
            logger.info(f"Loading content from URL: {url}")

            # Fetch URL content
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                html_content = response.text

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            # Create document
            document = Document(
                text=text,
                metadata={
                    "source": url,
                    "source_type": "url",
                    "url": url
                }
            )

            # Split into chunks
            nodes = self.text_splitter.get_nodes_from_documents([document])

            # Convert nodes back to documents
            chunked_docs = [
                Document(
                    text=node.text,
                    metadata=node.metadata
                )
                for node in nodes
            ]

            logger.info(f"Successfully loaded {len(chunked_docs)} chunks from URL")
            return chunked_docs

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading URL {url}: {str(e)}")
            raise Exception(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise Exception(f"Failed to load URL content: {str(e)}")

    async def load_csv(self, file_path: str) -> List[Document]:
        """
        Load and process a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of Document objects with chunked content
        """
        try:
            logger.info(f"Loading CSV from {file_path}")

            # Load CSV using LlamaIndex CSVReader
            reader = CSVReader()
            documents = reader.load_data(file=Path(file_path))

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "source_type": "csv",
                    "file_name": os.path.basename(file_path)
                })

            # Split into chunks
            nodes = self.text_splitter.get_nodes_from_documents(documents)

            # Convert nodes back to documents
            chunked_docs = [
                Document(
                    text=node.text,
                    metadata=node.metadata
                )
                for node in nodes
            ]

            logger.info(f"Successfully loaded {len(chunked_docs)} chunks from CSV")
            return chunked_docs

        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise Exception(f"Failed to load CSV: {str(e)}")

    async def load_documents(
        self,
        source: str,
        source_type: str
    ) -> List[Document]:
        """
        Load documents based on source type.

        Args:
            source: File path or URL
            source_type: Type of source ('pdf', 'url', 'csv')

        Returns:
            List of processed Document objects
        """
        if source_type == "pdf":
            return await self.load_pdf(source)
        elif source_type == "url":
            return await self.load_url(source)
        elif source_type == "csv":
            return await self.load_csv(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")


# Utility function to save uploaded file
async def save_upload_file(upload_file, destination: str) -> str:
    """
    Save an uploaded file to disk.

    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination directory path

    Returns:
        Path to saved file
    """
    try:
        os.makedirs(destination, exist_ok=True)
        file_path = os.path.join(destination, upload_file.filename)

        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)

        logger.info(f"File saved to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise Exception(f"Failed to save file: {str(e)}")
