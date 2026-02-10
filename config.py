"""
Configuration settings for the RAG project.
Supports environment variables for Docker deployment.
"""
import os
from pathlib import Path

# =============================================================================
# ENVIRONMENT LOADING
# =============================================================================
# Local development: Load .env.local (contains localhost URLs)
# Docker: .env.local does NOT exist (excluded via .dockerignore)
#         Environment variables come from docker-compose.yml's env_file: .env.docker
# =============================================================================
_ENV_LOCAL_FILE = Path(__file__).parent / ".env.local"
if _ENV_LOCAL_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_LOCAL_FILE)
        print(f"[config] Loaded environment from: {_ENV_LOCAL_FILE}")
    except ImportError:
        print("[config] Warning: python-dotenv not installed, using system environment variables")
else:
    # This branch runs in Docker where .env.local doesn't exist
    pass

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant_storage"
CONTENT_DIR = PROJECT_ROOT / "content"  # For mounting external content

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)
QDRANT_STORAGE_DIR.mkdir(exist_ok=True)
CONTENT_DIR.mkdir(exist_ok=True)

# Embedding model settings (using sentence-transformers - no API key needed!)
# Options: 
#   - "all-MiniLM-L6-v2" (384 dims, fast)
#   - "all-mpnet-base-v2" (768 dims, better quality)
#   - "BAAI/bge-large-en-v1.5" (1024 dims, high quality)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Alternative embedding models and their dimensions
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-small-en-v1.5": 384,
}

# Get embedding dimension based on model
EMBEDDING_DIM = EMBEDDING_MODELS.get(EMBEDDING_MODEL, 384)

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "python-doc")

# Whether to use local file storage or connect to Qdrant server
# In Docker, we connect to the qdrant service; locally we can use file storage
USE_QDRANT_SERVER = os.environ.get("USE_QDRANT_SERVER", "true").lower() == "true"

# Chunking settings
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "512"))
CHUNK_MIN_CHARS = int(os.environ.get("CHUNK_MIN_CHARS", "64"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

# RAG settings
SEARCH_THRESHOLD = float(os.environ.get("SEARCH_THRESHOLD", "0.5"))
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", "5"))

# LLM settings
# Option 1: Use Ollama (local, free, no API key)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # Options: "ollama", "openai", "mock"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")  # or "mistral", "gemma2", etc.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Option 2: Use OpenAI (requires API key)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Option 3: Use GitHub Models (requires GitHub token + Copilot subscription)
# Get token via: gh auth token, or set GITHUB_TOKEN env var
# Available models: gpt-4o, gpt-4o-mini, o1-preview, o1-mini, etc.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GHCP_MODEL = os.environ.get("GHCP_MODEL", "gpt-4o")  # or "gpt-4o-mini", "o1-preview"
GHCP_BASE_URL = os.environ.get("GHCP_BASE_URL", "https://models.inference.ai.azure.com")

# Gradio UI settings
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
