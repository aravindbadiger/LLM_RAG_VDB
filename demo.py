#!/usr/bin/env python3
"""
Quick demo script to test the RAG setup.
Run this after installing dependencies to verify everything works.
"""
import sys


def check_imports():
    """Check that all required packages are installed."""
    print("Checking imports...")
    
    packages = [
        ("gradio", "gradio"),
        ("qdrant_client", "qdrant-client"),
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
        ("bs4", "beautifulsoup4"),
        ("numpy", "numpy"),
    ]
    
    missing = []
    for module, package in packages:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\nAll required packages are installed!")
    return True


def check_ollama():
    """Check if Ollama is available."""
    print("\nChecking Ollama...")
    
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        
        if model_names:
            print(f"  ✓ Ollama is running with models: {', '.join(model_names)}")
            return True
        else:
            print("  ⚠ Ollama is running but no models installed")
            print("    Install a model with: ollama pull llama3.2")
            return False
    except ImportError:
        print("  ⚠ ollama package not installed (optional)")
        return False
    except Exception as e:
        print(f"  ⚠ Could not connect to Ollama: {e}")
        print("    Start Ollama with: ollama serve")
        return False


def test_embeddings():
    """Test the embedding model."""
    print("\nTesting embedding model...")
    
    try:
        from embeddings import get_embedding_model
        
        model = get_embedding_model()
        
        # Test embedding
        text = "Python is a programming language."
        embedding = model.embed_query(text)
        
        print(f"  ✓ Embedding model loaded: {model.model_name}")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_qdrant():
    """Test Qdrant connection."""
    print("\nTesting Qdrant...")
    
    try:
        from qdrant_utils import get_qdrant_manager, QdrantManager
        import config
        
        if config.USE_QDRANT_SERVER:
            manager = QdrantManager(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            mode = f"server ({config.QDRANT_HOST}:{config.QDRANT_PORT})"
        else:
            manager = QdrantManager(path=str(config.QDRANT_STORAGE_DIR))
            mode = "local storage"
        
        collections = manager.list_collections()
        
        print(f"  ✓ Qdrant working ({mode})")
        print(f"  ✓ Existing collections: {collections if collections else 'none'}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_full_pipeline():
    """Test the full RAG pipeline with sample data."""
    print("\nTesting full pipeline...")
    
    try:
        from embeddings import get_embedding_model
        from qdrant_utils import QdrantManager
        import config
        
        # Initialize
        model = get_embedding_model()
        if config.USE_QDRANT_SERVER:
            manager = QdrantManager(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        else:
            manager = QdrantManager(path=str(config.QDRANT_STORAGE_DIR))
        
        # Create test collection
        test_collection = "demo-test"
        manager.delete_collection(test_collection)  # Clean up
        manager.create_collection(test_collection, model.embedding_dim)
        
        # Ingest test documents
        test_docs = [
            "Python decorators are functions that modify the behavior of other functions.",
            "List comprehension is a concise way to create lists in Python.",
            "The asyncio module provides infrastructure for writing concurrent code.",
            "Tuples are immutable sequences in Python, unlike lists.",
            "File handling in Python uses the open() function with context managers.",
        ]
        
        manager.ingest_documents(test_collection, test_docs, model)
        
        # Search
        query = "How do decorators work in Python?"
        results = manager.search(test_collection, query, model, top_k=3)
        
        print(f"  ✓ Ingested {len(test_docs)} documents")
        print(f"  ✓ Search query: '{query}'")
        print(f"  ✓ Found {len(results)} results:")
        
        for text, score in results:
            print(f"      [{score:.3f}] {text[:60]}...")
        
        # Clean up
        manager.delete_collection(test_collection)
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("RAG Project Demo & Test")
    print("=" * 60)
    
    results = {}
    
    # Check imports
    results['imports'] = check_imports()
    if not results['imports']:
        print("\n❌ Please install missing packages first.")
        sys.exit(1)
    
    # Check Ollama
    results['ollama'] = check_ollama()
    
    # Test embeddings
    results['embeddings'] = test_embeddings()
    
    # Test Qdrant
    results['qdrant'] = test_qdrant()
    
    # Test full pipeline
    if results['embeddings'] and results['qdrant']:
        results['pipeline'] = test_full_pipeline()
    else:
        results['pipeline'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed and name not in ['ollama']:  # Ollama is optional
            all_passed = False
    
    print()
    
    if all_passed or (results['embeddings'] and results['qdrant'] and results['pipeline']):
        print("🎉 All core tests passed!")
        print("\nNext steps:")
        print("  1. Ingest data:    python ingest_data.py --download")
        print("  2. Start RAG app:  python rag_client.py")
        print("  3. Open browser:   http://localhost:7860")
        
        if not results['ollama']:
            print("\n⚠️  Note: Ollama not available. Install it for local LLM support,")
            print("   or set LLM_PROVIDER='openai' in config.py with your API key.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
