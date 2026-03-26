# Multi-Source RAG Chatbot

An intelligent chatbot powered by Retrieval-Augmented Generation (RAG) that can ingest and query knowledge from multiple sources including PDFs, web URLs, and CSV files. Built with LlamaIndex, ChromaDB, and OpenAI GPT-4.

![Tech Stack](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.22-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

-  **Multi-Source Ingestion**: Upload PDFs, CSV files, or ingest web content from URLs
-  **RAG-Powered Responses**: Retrieve relevant context from your documents and generate accurate answers
-  **Conversation Memory**: Session-based chat history (stores last 10 messages)
-  **Source Citations**: Every answer shows which documents it came from
-  **Streaming Support**: Real-time response streaming for better UX
-  **Modern UI**: Dark-themed, responsive chat interface
-  **Fully Dockerized**: Easy deployment with docker-compose
-  **Production Ready**: Async/await, error handling, health checks

##  Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Frontend  │ ───> │   FastAPI   │ ───> │  LlamaIndex │
│  (HTML/JS)  │      │   Backend   │      │     RAG     │
└─────────────┘      └─────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌─────────────┐      ┌─────────────┐
                     │   Session   │      │  ChromaDB   │
                     │   Storage   │      │   Vector    │
                     └─────────────┘      └─────────────┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │   OpenAI    │
                                          │  GPT-4 API  │
                                          └─────────────┘
```

##  Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **RAG Framework**: LlamaIndex
- **Vector Database**: ChromaDB (local)
- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-ada-002
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Containerization**: Docker, docker-compose

##  Prerequisites

- Docker and docker-compose (recommended)
- OR Python 3.11+ (for local development)
- OpenAI API key

##  Quick Start with Docker (Recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd multi-source-rag-chatbot
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Start the Application

```bash
docker-compose up --build
```

### 4. Access the Application

Open your browser and navigate to:
```
http://localhost:8080
```

That's it! The application is now running with:
- Chat interface at `http://localhost:8080`
- API documentation at `http://localhost:8080/docs`
- ChromaDB at `http://localhost:8000`

##  Local Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up ChromaDB

Option A: Use Docker for ChromaDB only
```bash
docker run -d -p 8000:8000 \
  -v chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest
```

Option B: Install and run ChromaDB locally
```bash
pip install chromadb
chroma run --path ./chroma_db --port 8000
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the Application

```bash
cd app
python main.py
```

Or with uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

##  Project Structure

```
multi-source-rag-chatbot/
├── app/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py              # PDF, URL, CSV ingestion
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py            # LlamaIndex RAG pipeline
│   ├── memory/
│   │   ├── __init__.py
│   │   └── chat_history.py        # Conversation memory
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # FastAPI endpoints
│   ├── frontend/
│   │   └── index.html             # Chat UI
│   └── main.py                    # FastAPI application
├── data/                          # Uploaded documents
├── sessions/                      # Conversation history
├── .env.example                   # Environment template
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

##  API Endpoints

### Document Upload

- `POST /api/upload/pdf` - Upload a PDF file
- `POST /api/upload/csv` - Upload a CSV file
- `POST /api/upload/url` - Ingest content from a web URL

### Chat

- `POST /api/chat` - Send a message and get a response
  ```json
  {
    "message": "What is this document about?",
    "session_id": "session_123",
    "stream": false
  }
  ```

### History Management

- `GET /api/history/{session_id}` - Get conversation history
- `DELETE /api/history/{session_id}` - Clear conversation history

### Information

- `GET /api/sources` - List all ingested documents
- `GET /api/sessions` - List all active sessions
- `GET /api/health` - Health check endpoint

## 📖 Usage Examples

### Upload a PDF

```bash
curl -X POST "http://localhost:8080/api/upload/pdf" \
  -F "file=@document.pdf"
```

### Ingest a Web URL

```bash
curl -X POST "http://localhost:8080/api/upload/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

### Send a Chat Message

```bash
curl -X POST "http://localhost:8080/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main points?",
    "session_id": "my_session",
    "stream": false
  }'
```

### Get Sources

```bash
curl "http://localhost:8080/api/sources"
```

## ⚙️ Configuration

Edit the `.env` file to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `LLM_MODEL` | OpenAI model to use | gpt-4 |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-ada-002 |
| `CHUNK_SIZE` | Text chunk size | 512 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `TOP_K_RETRIEVAL` | Number of chunks to retrieve | 5 |
| `TEMPERATURE` | LLM temperature | 0.7 |
| `MAX_TOKENS` | Max response tokens | 1000 |
| `CHROMA_HOST` | ChromaDB host | chromadb |
| `CHROMA_PORT` | ChromaDB port | 8000 |

##  Features in Detail

### Multi-Source Document Ingestion

- **PDF Files**: Extracts text from PDF documents using LlamaIndex PDFReader
- **CSV Files**: Parses structured data from CSV files
- **Web URLs**: Crawls and extracts content from web pages using BeautifulSoup

All documents are chunked into smaller pieces (configurable) and stored as embeddings in ChromaDB.

### RAG Pipeline

1. **Document Processing**: Documents are split into chunks with configurable size and overlap
2. **Embedding Generation**: Each chunk is converted to a vector embedding using OpenAI's ada-002 model
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Retrieval**: When a query comes in, the most relevant chunks are retrieved (top-k)
5. **Generation**: The LLM generates a response based on the retrieved context

### Conversation Memory

- Stores last 10 messages per session
- Session-based storage in JSON files
- Context is included in each query for coherent conversations

### Source Citations

Every response includes:
- The exact text chunks used to generate the answer
- Metadata about the source document
- Similarity scores (relevance)

##  Troubleshooting

### ChromaDB Connection Issues

If you get connection errors to ChromaDB:

1. Check if ChromaDB container is running:
   ```bash
   docker ps | grep chroma
   ```

2. Check ChromaDB health:
   ```bash
   curl http://localhost:8000/api/v1/heartbeat
   ```

3. Restart the containers:
   ```bash
   docker-compose down
   docker-compose up
   ```

### OpenAI API Errors

- Verify your API key is correct in `.env`
- Check your OpenAI account has sufficient credits
- Ensure you have access to GPT-4 (or change to GPT-3.5-turbo in `.env`)

### Document Upload Fails

- Check file size (default max: 10MB)
- Verify file format is supported
- Check application logs: `docker-compose logs app`

##  Performance Tips

1. **Adjust Chunk Size**: Smaller chunks (256-512) for more precise retrieval, larger (1024+) for more context
2. **Top-K Retrieval**: Increase for more context, decrease for faster responses
3. **Model Selection**: Use GPT-3.5-turbo for faster/cheaper responses, GPT-4 for better quality
4. **Embedding Caching**: ChromaDB caches embeddings, so repeated queries are fast

##  Security Considerations

- Store `.env` file securely and never commit it to version control
- Use environment variables for sensitive data
- In production, configure proper CORS origins
- Add authentication middleware for production deployments
- Use HTTPS in production
- Regularly update dependencies

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License.

##  Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the amazing RAG framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [OpenAI](https://openai.com/) for GPT-4 and embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
