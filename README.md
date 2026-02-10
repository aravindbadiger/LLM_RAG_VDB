# 🤖 RAG Project - Python Documentation Assistant

A complete Retrieval-Augmented Generation (RAG) system for querying Python documentation using:
- **Sentence Transformers** for embeddings (no API key required!)
- **Qdrant** vector database for semantic search
- **Gradio** for the web UI
- **Ollama**, **OpenAI**, or **GitHub Copilot** for LLM responses

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd rag_project

# Copy the environment template
cp .env.example .env
```

### 2. Configure LLM Provider

Edit `.env` and set your preferred LLM provider:

**Option A: Ollama (Local, Free) - Recommended**

No API key needed - runs completely locally:
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

**Option B: GitHub Copilot (GHCP)**

Requires a GitHub Copilot subscription. Get your token using GitHub CLI:

```bash
# Install GitHub CLI if not already installed
brew install gh  # macOS
# or: sudo apt install gh  # Ubuntu

# Login to GitHub
gh auth login

# Get your auth token
gh auth token
```

Add the token to your `.env` file:
```bash
LLM_PROVIDER=ghcp
GITHUB_TOKEN=gho_your_token_here
GHCP_MODEL=gpt-4o
```

**Option C: OpenAI**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. Build and Start Services

```bash
# Build containers
make build

# Start Qdrant + App (automatically starts Ollama if LLM_PROVIDER=ollama)
make up

# Check status
docker compose ps
```

Services:
- **RAG App**: http://localhost:7860
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 4. Ingest Content

```bash
# Ingest Python documentation
make ingest-python-docs

# Or ingest Wikipedia articles
make ingest-wiki COLLECTION=python-doc QUERY="Python programming language" PAGES=10

# Or ingest from URLs
make ingest-url COLLECTION=my-docs URLS="https://example.com/page1 https://example.com/page2"
```

### 5. Use the Application

Open http://localhost:7860 in your browser and start asking questions!

### 6. Manage Collections

```bash
# List collections
make schema-list

# Create a new collection
make schema-create COLLECTION=my-docs

# Delete a collection
make schema-delete COLLECTION=my-docs

# Clear a collection (keep schema)
make schema-clear COLLECTION=my-docs

# Get collection info
make schema-info COLLECTION=python-doc
```

### 7. Stop Services

```bash
make down
```

---

## ✨ Features

- **No API Key Required for Embeddings**: Uses Sentence Transformers which run locally
- **Local LLM Support**: Works with Ollama for completely free, local AI
- **Semantic Search**: Find relevant documentation based on meaning, not just keywords
- **Interactive Chat UI**: User-friendly Gradio interface
- **Configurable**: Easy to customize models, thresholds, and providers
- **Docker Support**: Full Docker Compose setup for easy deployment
- **Multiple Content Sources**: Ingest Python docs, Wikipedia, URLs, or text files

## 📁 Project Structure

```
rag_project/
├── config.py           # Configuration settings
├── embeddings.py       # Sentence Transformer embeddings
├── chunking.py         # Document chunking utilities
├── qdrant_utils.py     # Qdrant database management
├── llm.py              # LLM providers (Ollama, OpenAI)
├── rag_client.py       # Main RAG application with Gradio UI
├── ingest_data.py      # Data ingestion script
├── schema_manager.py   # CLI for managing collections & content
├── demo.py             # Test script to verify setup
├── requirements.txt    # Python dependencies
├── Dockerfile          # App container definition
├── docker-compose.yml  # Multi-container orchestration
├── Makefile            # Docker command shortcuts
├── .env.example        # Environment variables template
└── data/               # Data directory (created automatically)
    ├── chunks/         # Processed text chunks
    └── qdrant_storage/ # Local Qdrant database
```

---

## � Local Setup (Without Docker)

### 1. Install Dependencies

```bash
cd rag_project
pip install -r requirements.txt
```

### 2. Install Ollama (Recommended - Free Local LLM)

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

### 3. Download and Ingest Python Documentation

```bash
# Download Python 3.10 docs and ingest into Qdrant
python ingest_data.py --download --version 3.10

# Or if you already have docs downloaded:
python ingest_data.py --docs-dir /path/to/python-docs
```

### 4. Run the RAG Client

```bash
python rag_client.py
```

Then open http://localhost:7860 in your browser!

## ⚙️ Configuration

Edit `config.py` to customize:

### Embedding Model
```python
# Options (all run locally, no API key needed):
EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # Fast, 384 dims
EMBEDDING_MODEL = "all-mpnet-base-v2"        # Better quality, 768 dims
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # Best quality, 1024 dims
```

### LLM Provider
```python
# Option 1: Ollama (local, free)
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3.2"  # or "mistral", "gemma2", etc.

# Option 2: OpenAI (requires API key)
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."  # Or set OPENAI_API_KEY env variable
OPENAI_MODEL = "gpt-3.5-turbo"
```

### RAG Settings
```python
SEARCH_THRESHOLD = 0.5  # Minimum similarity score (0-1)
TOP_K_RESULTS = 5       # Number of context chunks to retrieve
```

## 📖 Usage Examples

### Command Line Options

```bash
# Use OpenAI instead of Ollama
python rag_client.py --llm openai

# Use a different embedding model
python rag_client.py --embedding-model all-mpnet-base-v2

# Create a public Gradio link
python rag_client.py --share

# Use a different port
python rag_client.py --port 8080
```

### Programmatic Usage

```python
from rag_client import RAGClient

# Initialize
client = RAGClient(
    collection_name="python-doc",
    llm_provider="ollama"
)

# Search for context
results = client.search_context("What is a Python decorator?")
for text, score in results:
    print(f"Score: {score:.3f}, Text: {text[:100]}...")

# Generate a response
response = client.generate_response(
    query="Explain Python decorators",
    sys_prompt="You are a Python expert."
)
print(response)
```

## 🔧 Docker Compose Details

The Docker setup includes multiple services:

### Services

| Service | Description | Port |
|---------|-------------|------|
| `qdrant` | Vector database | 6333 |
| `app` | RAG application with Gradio UI | 7860 |
| `schema-manager` | CLI for managing collections | - |
| `ingest` | Data ingestion service | - |

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# LLM Provider (ollama, openai, mock)
LLM_PROVIDER=mock

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI (if using)
OPENAI_API_KEY=sk-your-key

# Ollama (runs on host machine)
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Using Ollama with Docker

Ollama runs on your host machine (not in Docker). The app container connects via `host.docker.internal`:

```bash
# On your host machine
ollama serve
ollama pull llama3.2

# In .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Mounting Content

Place files in the `content/` directory to make them available in the container:

```bash
# Copy files to content directory
cp /path/to/documents/* content/

# Ingest from container
make ingest-files COLLECTION=my-docs DIR=/app/content
```

## 🧪 Testing Components

```bash
# Test embeddings
python embeddings.py

# Test chunking
python chunking.py

# Test Qdrant utilities
python qdrant_utils.py

# Test LLM
python llm.py
```

## 📚 Available Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ⚡ Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| BAAI/bge-small-en-v1.5 | 384 | ⚡ Fast | Good |
| BAAI/bge-large-en-v1.5 | 1024 | Slow | Best |

## 🔧 Troubleshooting

### "Could not connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check if a model is installed: `ollama list`
- Pull a model if needed: `ollama pull llama3.2`

### "Collection does not exist"
Run the ingestion script first:
```bash
python ingest_data.py --download
```

### "Out of memory"
- Use a smaller embedding model: `all-MiniLM-L6-v2`
- Reduce batch size: `python ingest_data.py --batch-size 20`

### OpenAI API errors
- Check your API key is set correctly
- Verify you have credits in your OpenAI account

## 📄 License

This project is for educational purposes, based on the RX-M RAG Training labs.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for local embeddings
- [Qdrant](https://qdrant.tech/) for vector database
- [Gradio](https://gradio.app/) for the UI framework
- [Ollama](https://ollama.ai/) for local LLM inference
