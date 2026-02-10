# 🤖 RAG Project - Python Documentation Assistant

A complete Retrieval-Augmented Generation (RAG) system for querying Python documentation using:
- **Sentence Transformers** for embeddings (no API key required!)
- **Qdrant** vector database for semantic search
- **Gradio** for the web UI
- **Ollama**, **OpenAI**, or **GitHub Copilot** for LLM responses

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start (Docker)](#-quick-start-docker)
- [Local Setup (Without Docker)](#-local-setup-without-docker)
- [Configuration](#️-configuration)
- [Usage Examples](#-usage-examples)
- [Testing Components](#-testing-components)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

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
├── config.py           # Configuration settings (reads from .env)
├── embeddings.py       # Sentence Transformer embeddings
├── chunking.py         # Document chunking utilities
├── qdrant_utils.py     # Qdrant database management
├── llm.py              # LLM providers (Ollama, OpenAI, GHCP)
├── rag_client.py       # Main RAG application with Gradio UI
├── ingest_data.py      # Data ingestion script
├── schema_manager.py   # CLI for managing collections & content
├── demo.py             # Test script to verify setup
├── requirements.txt    # Python dependencies
├── Dockerfile          # App container definition
├── docker-compose.yml  # Multi-container orchestration
├── Makefile            # Docker command shortcuts
└── .env.example        # Environment variables template
```

---

## 🚀 Quick Start (Docker)

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

# macOS
brew install gh

# Ubuntu/Debian Linux
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install gh -y

# Windows (PowerShell)
winget install --id GitHub.cli

# Windows (Chocolatey)
choco install gh

# Windows (Scoop)
scoop install gh
```

```bash
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

Requires an OpenAI API key (paid service). To get your API key:

1. Go to https://platform.openai.com/signup and create an account (or sign in)
2. Navigate to https://platform.openai.com/api-keys
3. Click "Create new secret key" and copy the key
4. Add billing information at https://platform.openai.com/account/billing

Add the key to your `.env` file:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. Build and Start Services

The project uses Docker Compose with the following services:

| Service | Description | Port |
|---------|-------------|------|
| `qdrant` | Vector database for storing and searching embeddings | 6333 |
| `app` | Main RAG application with Gradio web UI | 7860 |
| `ollama` | Local LLM server (only starts when `LLM_PROVIDER=ollama`) | 11434 |
| `schema-manager` | CLI utility for managing Qdrant collections | - |
| `ingest` | Data ingestion service for loading documents | - |

```bash
# Build containers
make build

# Start Qdrant + App (automatically starts Ollama if LLM_PROVIDER=ollama)
make up

# Check status
docker compose ps
```

Access the services:
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

## 💻 Local Setup (Without Docker)

### 1. Install Dependencies

```bash
cd rag_project
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and configure your LLM provider (see Quick Start section for details)
```

To load `.env` variables when running locally, install python-dotenv:
```bash
pip install python-dotenv
```

Or export variables manually:
```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
```

### 3. Install Ollama (If using Ollama)

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

### 4. Start Qdrant (Vector Database)

```bash
# Option A: Run Qdrant via Docker (recommended)
docker run -d -p 6333:6333 qdrant/qdrant

# Option B: The app can also use local file storage (no Qdrant server needed)
# Set in .env: USE_QDRANT_SERVER=false
```

### 5. Download and Ingest Python Documentation

```bash
# Download Python 3.10 docs and ingest into Qdrant
python ingest_data.py --download --version 3.10

# Or if you already have docs downloaded:
python ingest_data.py --docs-dir /path/to/python-docs
```

### 6. Run the RAG Client

```bash
python rag_client.py
```

Then open http://localhost:7860 in your browser!

---

## ⚙️ Configuration

All settings are configured via environment variables (`.env` file). The `config.py` file reads these values with sensible defaults.

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `openai`, `ghcp`, or `mock` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model for embeddings |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `SEARCH_THRESHOLD` | `0.5` | Minimum similarity score (0-1) |
| `TOP_K_RESULTS` | `5` | Number of context chunks to retrieve |

### Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|--------|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best |

### LLM Provider Settings

**Ollama:**
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

**OpenAI:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-3.5-turbo
```

**GitHub Copilot:**
```bash
LLM_PROVIDER=ghcp
GITHUB_TOKEN=gho_your_token
GHCP_MODEL=gpt-4o
```

---

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

---

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

---

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

---

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for local embeddings
- [Qdrant](https://qdrant.tech/) for vector database
- [Gradio](https://gradio.app/) for the UI framework
- [Ollama](https://ollama.ai/) for local LLM inference
