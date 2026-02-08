"""
Document chunking module for processing HTML/text documents.
Based on the RAG training lab exercises.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup, SoupStrainer
from tqdm import tqdm

import config

# Try to import unstructured - it's optional
try:
    from unstructured.partition.html import partition_html
    from unstructured.chunking.title import chunk_by_title
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: 'unstructured' package not available. Using basic chunking.")


def download_nltk_data():
    """Download required NLTK data for text processing."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def basic_chunk_text(text: str, 
                     max_chars: int = None,
                     min_chars: int = None,
                     overlap: int = None) -> List[str]:
    """
    Simple text chunking by splitting on paragraphs/sentences.
    
    Args:
        text: The text to chunk.
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.
        overlap: Number of characters to overlap between chunks.
    
    Returns:
        List of text chunks.
    """
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    min_chars = min_chars or config.CHUNK_MIN_CHARS
    overlap = overlap or config.CHUNK_OVERLAP
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed max_chars
        if len(current_chunk) + len(para) + 1 > max_chars:
            # Save current chunk if it meets minimum
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + para
            else:
                current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if len(current_chunk) >= min_chars:
        chunks.append(current_chunk.strip())
    elif chunks and current_chunk:
        # Append to last chunk if too small
        chunks[-1] += "\n\n" + current_chunk.strip()
    
    return chunks


def chunk_html_unstructured(html_content: str,
                            max_chars: int = None,
                            min_chars: int = None) -> List[str]:
    """
    Chunk HTML content using the unstructured library.
    
    Args:
        html_content: The HTML content as a string.
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.
    
    Returns:
        List of text chunks.
    """
    if not UNSTRUCTURED_AVAILABLE:
        # Fall back to basic chunking
        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text(separator='\n\n')
        return basic_chunk_text(text, max_chars, min_chars)
    
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    min_chars = min_chars or config.CHUNK_MIN_CHARS
    
    # Filter to document content
    soup = BeautifulSoup(
        html_content,
        'lxml',
        parse_only=SoupStrainer(class_='document')
    )
    html_filtered = soup.text if soup.text else html_content
    
    try:
        # Partition the HTML
        elements = partition_html(text=html_filtered)
        
        # Chunk by title
        chunks = chunk_by_title(
            elements,
            multipage_sections=True,
            combine_text_under_n_chars=min_chars,
            max_characters=max_chars,
        )
        
        return [chunk.text for chunk in chunks]
    
    except Exception as e:
        print(f"Unstructured chunking failed: {e}")
        # Fall back to basic chunking
        return basic_chunk_text(html_filtered, max_chars, min_chars)


def chunk_html_file(filepath: str,
                    max_chars: int = None,
                    min_chars: int = None) -> List[str]:
    """
    Read and chunk an HTML file.
    
    Args:
        filepath: Path to the HTML file.
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.
    
    Returns:
        List of text chunks.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    return chunk_html_unstructured(html_content, max_chars, min_chars)


def chunk_text_file(filepath: str,
                    max_chars: int = None,
                    min_chars: int = None,
                    overlap: int = None) -> List[str]:
    """
    Read and chunk a text file.
    
    Args:
        filepath: Path to the text file.
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.
        overlap: Number of characters to overlap between chunks.
    
    Returns:
        List of text chunks.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    return basic_chunk_text(text, max_chars, min_chars, overlap)


def process_directory(input_dir: str,
                      output_dir: str = None,
                      file_extension: str = '.html',
                      max_chars: int = None,
                      min_chars: int = None) -> List[Tuple[str, List[str]]]:
    """
    Process all files in a directory and chunk them.
    
    Args:
        input_dir: Directory containing files to process.
        output_dir: Optional directory to save chunks as text files.
        file_extension: File extension to process (e.g., '.html', '.txt').
        max_chars: Maximum characters per chunk.
        min_chars: Minimum characters per chunk.
    
    Returns:
        List of tuples (filename, list of chunks).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    files = list(input_path.glob(f'**/*{file_extension}'))
    
    print(f"Processing {len(files)} {file_extension} files...")
    
    for filepath in tqdm(files, desc="Chunking files"):
        try:
            if file_extension == '.html':
                chunks = chunk_html_file(str(filepath), max_chars, min_chars)
            else:
                chunks = chunk_text_file(str(filepath), max_chars, min_chars)
            
            results.append((filepath.name, chunks))
            
            # Save chunks to files if output directory specified
            if output_path:
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{filepath.stem}-{i:03d}.txt"
                    chunk_path = output_path / chunk_filename
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
            
            print(f"Processed {filepath.name}: {len(chunks)} chunks")
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return results


def load_chunks_from_directory(chunks_dir: str) -> List[Tuple[str, str]]:
    """
    Load pre-processed chunks from a directory.
    
    Args:
        chunks_dir: Directory containing chunk text files.
    
    Returns:
        List of tuples (filename, chunk_text).
    """
    chunks_path = Path(chunks_dir)
    chunks = []
    
    for chunk_file in sorted(chunks_path.glob('*.txt')):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk_text = f.read().strip()
            if chunk_text:
                chunks.append((chunk_file.name, chunk_text))
    
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")
    return chunks


if __name__ == "__main__":
    # Test chunking
    print("Testing chunking functionality...")
    
    # Download NLTK data
    download_nltk_data()
    
    # Test basic chunking
    sample_text = """
This is the first paragraph. It contains some text about Python programming.
Python is a versatile language used for web development, data science, and more.

This is the second paragraph. It discusses machine learning and AI.
Neural networks are a fundamental concept in deep learning.

This is the third paragraph about databases. Vector databases store embeddings
for semantic search. Qdrant is a popular vector database option.

This is the fourth paragraph. It's about natural language processing.
NLP enables computers to understand and generate human language.
    """
    
    chunks = basic_chunk_text(sample_text, max_chars=200, min_chars=50)
    print(f"\nBasic chunking produced {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
