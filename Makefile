.PHONY: help build up up-ollama down logs shell schema-list schema-create schema-delete ingest-wiki ingest-url ingest-files ingest-python-docs clean pull-model list-models start-ollama stop-ollama

# Default target
help:
	@echo "RAG Project Docker Commands"
	@echo "============================"
	@echo ""
	@echo "Setup & Run:"
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start Qdrant + App (for OpenAI/GHCP)"
	@echo "  make up-ollama      - Start all services including Ollama"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - View logs"
	@echo "  make shell          - Open shell in app container"
	@echo ""
	@echo "Ollama Management:"
	@echo "  make start-ollama   - Start Ollama container only"
	@echo "  make stop-ollama    - Stop Ollama container"
	@echo "  make pull-model MODEL=llama3.2  - Pull a model into Ollama"
	@echo "  make list-models    - List available models"
	@echo ""
	@echo "Schema Management:"
	@echo "  make schema-list    - List all collections"
	@echo "  make schema-create COLLECTION=name  - Create a collection"
	@echo "  make schema-delete COLLECTION=name  - Delete a collection"
	@echo "  make schema-clear COLLECTION=name   - Clear a collection"
	@echo "  make schema-info COLLECTION=name    - Get collection info"
	@echo ""
	@echo "Content Ingestion:"
	@echo "  make ingest-wiki COLLECTION=name QUERY='search terms' [PAGES=5]"
	@echo "  make ingest-url COLLECTION=name URLS='url1 url2'"
	@echo "  make ingest-files COLLECTION=name DIR=/path/to/files"
	@echo "  make ingest-python-docs [VERSION=3.10]  - Download & ingest Python docs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove containers and volumes"
	@echo "  make rebuild        - Rebuild and restart"
	@echo ""
	@echo "Examples:"
	@echo "  make build && make up"
	@echo "  make schema-create COLLECTION=my-docs"
	@echo "  make ingest-wiki COLLECTION=my-docs QUERY='machine learning' PAGES=10"
	@echo "  make ingest-python-docs"

# =============================================================================
# Setup & Run
# =============================================================================

build:
	docker compose build

up:
	docker compose up -d qdrant app
	@echo ""
	@echo "Services started (Qdrant + App)!"
	@echo "  - Qdrant:  http://localhost:6333"
	@echo "  - RAG App: http://localhost:7860"
	@echo ""
	@echo "Using LLM_PROVIDER from .env (default: ollama)"
	@echo "For Ollama, run: make start-ollama && make pull-model"

up-ollama:
	docker compose --profile ollama up -d
	@echo ""
	@echo "All services started (including Ollama)!"
	@echo "  - Qdrant:  http://localhost:6333"
	@echo "  - Ollama:  http://localhost:11434"
	@echo "  - RAG App: http://localhost:7860"
	@echo ""
	@echo "Don't forget to pull a model: make pull-model MODEL=llama3.2"

down:
	docker compose --profile ollama down

logs:
	docker compose logs -f

shell:
	docker compose exec app /bin/bash

rebuild: down build up

# =============================================================================
# Ollama Management
# =============================================================================

start-ollama:
	docker compose --profile ollama up -d ollama
	@echo "Ollama started at http://localhost:11434"
	@echo "Pull a model with: make pull-model MODEL=llama3.2"

stop-ollama:
	docker compose --profile ollama stop ollama
	@echo "Ollama stopped"

MODEL ?= llama3.2

pull-model:
	@echo "Pulling model $(MODEL) into Ollama..."
	docker exec rag-ollama ollama pull $(MODEL)
	@echo "Model $(MODEL) pulled successfully!"

list-models:
	@echo "Available models in Ollama:"
	docker exec rag-ollama ollama list

# =============================================================================
# Schema Management
# =============================================================================

schema-list:
	docker compose run --rm schema-manager list

schema-create:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make schema-create COLLECTION=name)
endif
	docker compose run --rm schema-manager create $(COLLECTION)

schema-delete:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make schema-delete COLLECTION=name)
endif
	docker compose run --rm schema-manager delete $(COLLECTION) --force

schema-clear:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make schema-clear COLLECTION=name)
endif
	docker compose run --rm schema-manager clear $(COLLECTION) --force

schema-info:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make schema-info COLLECTION=name)
endif
	docker compose run --rm schema-manager info $(COLLECTION)

# =============================================================================
# Content Ingestion
# =============================================================================

PAGES ?= 5

ingest-wiki:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make ingest-wiki COLLECTION=name QUERY='terms')
endif
ifndef QUERY
	$(error QUERY is required. Usage: make ingest-wiki COLLECTION=name QUERY='terms')
endif
	docker compose run --rm schema-manager ingest-wiki $(COLLECTION) "$(QUERY)" --max-pages $(PAGES)

ingest-url:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make ingest-url COLLECTION=name URLS='url1 url2')
endif
ifndef URLS
	$(error URLS is required. Usage: make ingest-url COLLECTION=name URLS='url1 url2')
endif
	docker compose run --rm schema-manager ingest-url $(COLLECTION) $(URLS)

ingest-files:
ifndef COLLECTION
	$(error COLLECTION is required. Usage: make ingest-files COLLECTION=name DIR=/path)
endif
ifndef DIR
	$(error DIR is required. Usage: make ingest-files COLLECTION=name DIR=/path)
endif
	docker compose run --rm -v $(DIR):/app/content/external schema-manager ingest-files $(COLLECTION) /app/content/external

VERSION ?= 3.10

ingest-python-docs:
	@echo "Downloading and ingesting Python $(VERSION) documentation..."
	docker compose run --rm ingest --download --version $(VERSION) --collection python-doc

# =============================================================================
# Maintenance
# =============================================================================

clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up containers and volumes"

# Quick test
test:
	docker compose run --rm app python demo.py
