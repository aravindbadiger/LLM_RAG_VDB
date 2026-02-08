#!/usr/bin/env python3
"""
Schema Manager - CLI tool for managing Qdrant collections and content ingestion.
Supports Python docs, Wikipedia pages, text files, and URLs.
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from embeddings import get_embedding_model
from chunking import basic_chunk_text, chunk_html_unstructured, download_nltk_data
from qdrant_utils import QdrantManager


def get_qdrant_manager() -> QdrantManager:
    """Get Qdrant manager with Docker-aware configuration."""
    host = os.environ.get('QDRANT_HOST', config.QDRANT_HOST)
    port = int(os.environ.get('QDRANT_PORT', config.QDRANT_PORT))
    return QdrantManager(host=host, port=port)


def list_collections():
    """List all collections in Qdrant."""
    manager = get_qdrant_manager()
    collections = manager.list_collections()
    
    if not collections:
        print("No collections found.")
        return
    
    print("\n📚 Collections:")
    print("-" * 50)
    for name in collections:
        info = manager.get_collection_info(name)
        points = info.get('points_count', 0)
        status = info.get('status', 'unknown')
        print(f"  • {name}: {points} points ({status})")
    print()


def create_collection(name: str, embedding_dim: int = None):
    """Create a new collection."""
    manager = get_qdrant_manager()
    
    if embedding_dim is None:
        model = get_embedding_model()
        embedding_dim = model.embedding_dim
    
    if manager.collection_exists(name):
        print(f"❌ Collection '{name}' already exists.")
        return False
    
    success = manager.create_collection(name, embedding_dim)
    if success:
        print(f"✅ Collection '{name}' created (dim={embedding_dim})")
    return success


def delete_collection(name: str, force: bool = False):
    """Delete a collection."""
    manager = get_qdrant_manager()
    
    if not manager.collection_exists(name):
        print(f"❌ Collection '{name}' does not exist.")
        return False
    
    if not force:
        info = manager.get_collection_info(name)
        points = info.get('points_count', 0)
        confirm = input(f"⚠️  Delete '{name}' with {points} points? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return False
    
    success = manager.delete_collection(name)
    if success:
        print(f"✅ Collection '{name}' deleted.")
    return success


def clear_collection(name: str, force: bool = False):
    """Clear all points from a collection (delete and recreate)."""
    manager = get_qdrant_manager()
    
    if not manager.collection_exists(name):
        print(f"❌ Collection '{name}' does not exist.")
        return False
    
    info = manager.get_collection_info(name)
    points = info.get('points_count', 0)
    
    if not force:
        confirm = input(f"⚠️  Clear {points} points from '{name}'? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return False
    
    # Get embedding dimension before deleting
    model = get_embedding_model()
    embedding_dim = model.embedding_dim
    
    # Delete and recreate
    manager.delete_collection(name)
    manager.create_collection(name, embedding_dim)
    print(f"✅ Collection '{name}' cleared.")
    return True


def fetch_wikipedia(query: str, max_pages: int = 5) -> List[dict]:
    """
    Fetch Wikipedia pages related to a query.
    
    Args:
        query: Search query.
        max_pages: Maximum number of pages to fetch.
    
    Returns:
        List of dicts with 'title' and 'content'.
    """
    print(f"🔍 Searching Wikipedia for: {query}")
    
    # User agent required by Wikipedia API
    headers = {
        "User-Agent": "RAGApp/1.0 (Python; educational project)"
    }
    
    # Search for pages
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_pages,
        "format": "json"
    }
    
    try:
        response = requests.get(search_url, params=search_params, headers=headers, timeout=30)
        response.raise_for_status()
        search_results = response.json()
        
        pages = []
        for result in search_results.get("query", {}).get("search", []):
            title = result["title"]
            pageid = result["pageid"]
            
            # Fetch page content
            content_params = {
                "action": "query",
                "pageids": pageid,
                "prop": "extracts",
                "explaintext": True,
                "format": "json"
            }
            
            content_response = requests.get(search_url, params=content_params, headers=headers, timeout=30)
            content_response.raise_for_status()
            content_data = content_response.json()
            
            page_content = content_data.get("query", {}).get("pages", {}).get(str(pageid), {})
            extract = page_content.get("extract", "")
            
            if extract:
                pages.append({
                    "title": title,
                    "content": extract,
                    "source": f"wikipedia:{title}",
                    "type": "wikipedia"
                })
                print(f"  ✓ Fetched: {title}")
        
        return pages
    
    except Exception as e:
        print(f"❌ Error fetching Wikipedia: {e}")
        return []


def fetch_url(url: str) -> Optional[dict]:
    """
    Fetch and extract text from a URL.
    
    Args:
        url: URL to fetch.
    
    Returns:
        Dict with 'title' and 'content', or None on failure.
    """
    print(f"🌐 Fetching: {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract title
        title = soup.title.string if soup.title else url
        
        # Remove scripts, styles, nav, footer
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n\n')
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = '\n\n'.join(lines)
        
        return {
            "title": title,
            "content": content,
            "source": url,
            "type": "url"
        }
    
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None


def read_text_files(directory: str, extensions: List[str] = None) -> List[dict]:
    """
    Read text files from a directory.
    
    Args:
        directory: Directory path.
        extensions: List of file extensions to include.
    
    Returns:
        List of dicts with 'title' and 'content'.
    """
    extensions = extensions or ['.txt', '.md', '.rst']
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    documents = []
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'**/*{ext}'))
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if content.strip():
                documents.append({
                    "title": filepath.name,
                    "content": content,
                    "source": str(filepath),
                    "type": "file"
                })
                print(f"  ✓ Read: {filepath.name}")
        
        except Exception as e:
            print(f"  ✗ Error reading {filepath}: {e}")
    
    return documents


def ingest_documents(collection: str, documents: List[dict], batch_size: int = 50):
    """
    Chunk and ingest documents into a collection.
    
    Args:
        collection: Collection name.
        documents: List of document dicts.
        batch_size: Batch size for ingestion.
    """
    if not documents:
        print("No documents to ingest.")
        return 0
    
    manager = get_qdrant_manager()
    model = get_embedding_model()
    
    # Create collection if needed
    if not manager.collection_exists(collection):
        print(f"Creating collection '{collection}'...")
        manager.create_collection(collection, model.embedding_dim)
    
    # Chunk all documents
    all_chunks = []
    all_metadata = []
    
    print(f"\n📄 Processing {len(documents)} documents...")
    for doc in tqdm(documents, desc="Chunking"):
        chunks = basic_chunk_text(
            doc["content"],
            max_chars=config.CHUNK_MAX_CHARS,
            min_chars=config.CHUNK_MIN_CHARS
        )
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", ""),
                    "type": doc.get("type", "unknown")
                })
    
    print(f"\n📊 Total chunks: {len(all_chunks)}")
    
    # Ingest
    total = manager.ingest_documents(
        collection_name=collection,
        documents=all_chunks,
        embedding_model=model,
        metadata=all_metadata,
        batch_size=batch_size
    )
    
    return total


def ingest_wikipedia(collection: str, query: str, max_pages: int = 5):
    """Ingest Wikipedia pages into a collection."""
    documents = fetch_wikipedia(query, max_pages)
    return ingest_documents(collection, documents)


def ingest_urls(collection: str, urls: List[str]):
    """Ingest content from URLs into a collection."""
    documents = []
    for url in urls:
        doc = fetch_url(url)
        if doc:
            documents.append(doc)
    return ingest_documents(collection, documents)


def ingest_files(collection: str, directory: str, extensions: List[str] = None):
    """Ingest text files from a directory into a collection."""
    documents = read_text_files(directory, extensions)
    return ingest_documents(collection, documents)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Schema Manager - Manage Qdrant collections and content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python schema_manager.py list

  # Create a new collection
  python schema_manager.py create my-docs

  # Delete a collection
  python schema_manager.py delete my-docs

  # Clear a collection
  python schema_manager.py clear my-docs

  # Ingest Wikipedia pages about Python
  python schema_manager.py ingest-wiki python-doc "Python programming language" --max-pages 10

  # Ingest content from URLs
  python schema_manager.py ingest-url my-docs https://example.com/page1 https://example.com/page2

  # Ingest text files from a directory
  python schema_manager.py ingest-files my-docs /path/to/docs --extensions .txt .md

Docker Compose usage:
  # List collections
  docker compose run --rm schema-manager list

  # Create collection
  docker compose run --rm schema-manager create python-doc

  # Ingest Wikipedia
  docker compose run --rm schema-manager ingest-wiki python-doc "Python programming"
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    subparsers.add_parser("list", help="List all collections")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a collection")
    create_parser.add_argument("name", help="Collection name")
    create_parser.add_argument("--dim", type=int, help="Embedding dimension (auto-detected if not specified)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a collection")
    delete_parser.add_argument("name", help="Collection name")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all points from a collection")
    clear_parser.add_argument("name", help="Collection name")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    
    # Ingest Wikipedia command
    wiki_parser = subparsers.add_parser("ingest-wiki", help="Ingest Wikipedia pages")
    wiki_parser.add_argument("collection", help="Collection name")
    wiki_parser.add_argument("query", help="Wikipedia search query")
    wiki_parser.add_argument("--max-pages", type=int, default=5, help="Max pages to fetch")
    
    # Ingest URL command
    url_parser = subparsers.add_parser("ingest-url", help="Ingest content from URLs")
    url_parser.add_argument("collection", help="Collection name")
    url_parser.add_argument("urls", nargs="+", help="URLs to ingest")
    
    # Ingest files command
    files_parser = subparsers.add_parser("ingest-files", help="Ingest text files from directory")
    files_parser.add_argument("collection", help="Collection name")
    files_parser.add_argument("directory", help="Directory containing files")
    files_parser.add_argument("--extensions", nargs="+", default=[".txt", ".md"], 
                              help="File extensions to include")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get collection info")
    info_parser.add_argument("name", help="Collection name")
    
    args = parser.parse_args()
    
    # Download NLTK data
    download_nltk_data()
    
    if args.command == "list":
        list_collections()
    
    elif args.command == "create":
        create_collection(args.name, args.dim)
    
    elif args.command == "delete":
        delete_collection(args.name, args.force)
    
    elif args.command == "clear":
        clear_collection(args.name, args.force)
    
    elif args.command == "ingest-wiki":
        total = ingest_wikipedia(args.collection, args.query, args.max_pages)
        print(f"\n✅ Ingested {total} chunks from Wikipedia")
    
    elif args.command == "ingest-url":
        total = ingest_urls(args.collection, args.urls)
        print(f"\n✅ Ingested {total} chunks from URLs")
    
    elif args.command == "ingest-files":
        total = ingest_files(args.collection, args.directory, args.extensions)
        print(f"\n✅ Ingested {total} chunks from files")
    
    elif args.command == "info":
        manager = get_qdrant_manager()
        info = manager.get_collection_info(args.name)
        if info:
            print(f"\n📊 Collection: {args.name}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ Collection '{args.name}' not found")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
