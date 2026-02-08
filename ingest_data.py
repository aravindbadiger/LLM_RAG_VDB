"""
Data ingestion script.
Downloads Python documentation and ingests it into Qdrant.
"""
import os
import argparse
import zipfile
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

import config
from embeddings import get_embedding_model
from chunking import chunk_html_file, process_directory, download_nltk_data
from qdrant_utils import get_qdrant_manager


def download_python_docs(version: str = "3.10", output_dir: Path = None) -> Path:
    """
    Download Python documentation in HTML format.
    
    Args:
        version: Python version (e.g., "3.10", "3.11").
        output_dir: Directory to save the documentation.
        
    Returns:
        Path to the extracted documentation directory.
    """
    output_dir = output_dir or config.DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct URL
    # Note: The exact URL format may vary, this is a common pattern
    base_url = f"https://docs.python.org/{version}/archives/"
    
    # Try to find the latest patch version
    zip_filename = None
    for patch in range(20, -1, -1):  # Try from .20 down to .0
        try:
            test_url = f"{base_url}python-{version}.{patch}-docs-html.zip"
            response = requests.head(test_url, timeout=10)
            if response.status_code == 200:
                zip_filename = f"python-{version}.{patch}-docs-html.zip"
                break
        except:
            continue
    
    if not zip_filename:
        # Try without patch version
        zip_filename = f"python-{version}-docs-html.zip"
    
    url = f"{base_url}{zip_filename}"
    zip_path = output_dir / zip_filename
    extract_dir = output_dir / zip_filename.replace('.zip', '')
    
    # Check if already downloaded
    if extract_dir.exists():
        print(f"Documentation already exists at {extract_dir}")
        return extract_dir
    
    print(f"Downloading Python {version} documentation from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded to {zip_path}")
        
        # Extract
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        return extract_dir
    
    except requests.RequestException as e:
        print(f"Error downloading documentation: {e}")
        print("\nAlternative: Download manually from https://docs.python.org/3/download.html")
        raise


def clean_documentation(docs_dir: Path) -> None:
    """
    Remove unnecessary files from the documentation directory.
    
    Args:
        docs_dir: Path to the documentation directory.
    """
    # Remove root HTML files (indexes, etc.)
    for html_file in docs_dir.glob('*.html'):
        html_file.unlink()
        print(f"Removed: {html_file.name}")
    
    # Remove directories that aren't useful for RAG
    dirs_to_remove = ['distributing', 'whatsnew', 'installing', '_images', '_static']
    for dir_name in dirs_to_remove:
        dir_path = docs_dir / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Removed directory: {dir_name}")


def ingest_python_docs(docs_dir: Path = None,
                       collection_name: str = None,
                       batch_size: int = 50) -> int:
    """
    Process and ingest Python documentation into Qdrant.
    
    Args:
        docs_dir: Path to the documentation directory.
        collection_name: Qdrant collection name.
        batch_size: Number of documents to process at once.
        
    Returns:
        Number of documents ingested.
    """
    collection_name = collection_name or config.QDRANT_COLLECTION
    
    # Initialize components
    print("\nInitializing embedding model...")
    embedding_model = get_embedding_model()
    
    print("\nConnecting to Qdrant...")
    if config.USE_QDRANT_SERVER:
        qdrant = get_qdrant_manager(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    else:
        qdrant = get_qdrant_manager(path=str(config.QDRANT_STORAGE_DIR))
    
    # Create collection if needed
    if not qdrant.collection_exists(collection_name):
        print(f"\nCreating collection '{collection_name}'...")
        qdrant.create_collection(collection_name, embedding_model.embedding_dim)
    
    # Find documentation to process
    if docs_dir:
        docs_path = Path(docs_dir)
    else:
        # Look for downloaded docs
        docs_path = None
        for path in config.DATA_DIR.glob('python-*-docs-html'):
            docs_path = path
            break
    
    if not docs_path or not docs_path.exists():
        raise FileNotFoundError(
            "Documentation directory not found. "
            "Run with --download flag to download Python documentation."
        )
    
    # Process library documentation (most useful for RAG)
    library_dir = docs_path / 'library'
    if not library_dir.exists():
        library_dir = docs_path  # Fall back to root
    
    print(f"\nProcessing documentation from {library_dir}...")
    
    # Collect all chunks
    all_chunks = []
    all_metadata = []
    
    html_files = list(library_dir.glob('**/*.html'))
    print(f"Found {len(html_files)} HTML files")
    
    for html_file in tqdm(html_files, desc="Chunking files"):
        try:
            chunks = chunk_html_file(str(html_file))
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Skip very small chunks
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": html_file.name,
                        "type": "python_doc"
                    })
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
    
    print(f"\nTotal chunks to ingest: {len(all_chunks)}")
    
    # Ingest into Qdrant
    total_ingested = qdrant.ingest_documents(
        collection_name=collection_name,
        documents=all_chunks,
        embedding_model=embedding_model,
        metadata=all_metadata,
        batch_size=batch_size
    )
    
    # Print collection info
    info = qdrant.get_collection_info(collection_name)
    print(f"\nCollection '{collection_name}' now has {info.get('points_count', 0)} points")
    
    return total_ingested


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest Python documentation into Qdrant")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Python documentation"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="3.10",
        help="Python documentation version to download"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        help="Path to existing Python documentation directory"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=config.QDRANT_COLLECTION,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for ingestion"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean unnecessary files from documentation"
    )
    
    args = parser.parse_args()
    
    # Download NLTK data
    print("Downloading NLTK data...")
    download_nltk_data()
    
    # Download documentation if requested
    if args.download:
        docs_dir = download_python_docs(args.version)
        
        if args.clean:
            print("\nCleaning documentation...")
            clean_documentation(docs_dir)
    else:
        docs_dir = Path(args.docs_dir) if args.docs_dir else None
    
    # Ingest documentation
    print("\nStarting ingestion...")
    total = ingest_python_docs(
        docs_dir=docs_dir,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ Successfully ingested {total} document chunks!")
    print(f"You can now run: python rag_client.py")


if __name__ == "__main__":
    main()
