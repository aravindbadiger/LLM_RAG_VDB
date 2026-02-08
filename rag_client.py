"""
RAG Client with Gradio UI.
Main application that combines embeddings, vector database, and LLM for RAG.
"""
import gradio as gr
from typing import List, Tuple, Generator
import time

import config
from embeddings import EmbeddingModel, get_embedding_model
from qdrant_utils import QdrantManager, get_qdrant_manager
from llm import get_llm, make_inputs, make_context_inputs


class RAGClient:
    """
    RAG (Retrieval-Augmented Generation) client.
    Combines vector search with LLM for context-aware responses.
    """
    
    def __init__(self,
                 collection_name: str = None,
                 llm_provider: str = None,
                 embedding_model: str = None):
        """
        Initialize the RAG client.
        
        Args:
            collection_name: Qdrant collection to use.
            llm_provider: LLM provider ("ollama", "openai", "mock").
            embedding_model: Sentence transformer model name.
        """
        self.collection_name = collection_name or config.QDRANT_COLLECTION
        
        print("Initializing RAG Client...")
        
        # Initialize embedding model (no API key needed!)
        print("\n1. Loading embedding model...")
        self.embedding_model = get_embedding_model(embedding_model)
        
        # Initialize Qdrant manager
        print("\n2. Connecting to Qdrant...")
        if config.USE_QDRANT_SERVER:
            self.qdrant = get_qdrant_manager(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        else:
            self.qdrant = get_qdrant_manager(path=str(config.QDRANT_STORAGE_DIR))
        
        # Initialize LLM
        print("\n3. Initializing LLM...")
        self.llm = get_llm(llm_provider)
        
        print("\nRAG Client initialized successfully!")
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        return self.qdrant.list_collections()
    
    def set_collection(self, collection_name: str) -> bool:
        """Switch to a different collection."""
        if collection_name in self.list_collections():
            self.collection_name = collection_name
            return True
        return False
    
    def search_context(self, 
                       query: str, 
                       top_k: int = None,
                       threshold: float = None) -> List[Tuple[str, float]]:
        """
        Search for relevant context chunks.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            threshold: Minimum similarity score.
            
        Returns:
            List of (text, score) tuples.
        """
        return self.qdrant.search(
            collection_name=self.collection_name,
            query_text=query,
            embedding_model=self.embedding_model,
            top_k=top_k or config.TOP_K_RESULTS,
            score_threshold=threshold or config.SEARCH_THRESHOLD
        )
    
    def generate_response(self,
                          query: str,
                          history: List[List[str]] = None,
                          sys_prompt: str = None,
                          use_rag: bool = True) -> str:
        """
        Generate a response using RAG.
        
        Args:
            query: User query.
            history: Chat history.
            sys_prompt: System prompt.
            use_rag: Whether to use RAG context.
            
        Returns:
            Generated response text.
        """
        history = history or []
        sys_prompt = sys_prompt or ""
        
        # Add current query to history
        current_history = history + [[query, None]]
        
        if use_rag:
            # Search for context
            context_results = self.search_context(query)
            context_texts = [text for text, score in context_results]
            
            if context_texts:
                messages = make_context_inputs(sys_prompt, current_history, context_texts)
            else:
                messages = make_inputs(sys_prompt, current_history)
        else:
            messages = make_inputs(sys_prompt, current_history)
        
        # Generate response
        response = self.llm.generate(messages)
        return response
    
    def generate_response_stream(self,
                                  query: str,
                                  history: List[List[str]] = None,
                                  sys_prompt: str = None,
                                  use_rag: bool = True) -> Generator[str, None, None]:
        """
        Generate a streaming response using RAG.
        
        Args:
            query: User query.
            history: Chat history.
            sys_prompt: System prompt.
            use_rag: Whether to use RAG context.
            
        Yields:
            Response text chunks.
        """
        history = history or []
        sys_prompt = sys_prompt or ""
        
        # Add current query to history
        current_history = history + [[query, None]]
        
        if use_rag:
            # Search for context
            context_results = self.search_context(query)
            context_texts = [text for text, score in context_results]
            
            if context_texts:
                messages = make_context_inputs(sys_prompt, current_history, context_texts)
            else:
                messages = make_inputs(sys_prompt, current_history)
        else:
            messages = make_inputs(sys_prompt, current_history)
        
        # Generate streaming response
        for chunk in self.llm.generate_stream(messages):
            yield chunk


def create_gradio_app(rag_client: RAGClient = None) -> gr.Blocks:
    """
    Create the Gradio UI application.
    
    Args:
        rag_client: RAGClient instance. Creates one if not provided.
        
    Returns:
        Gradio Blocks app.
    """
    if rag_client is None:
        rag_client = RAGClient()
    
    # Get available collections
    available_collections = rag_client.list_collections()
    
    # Check if collection exists and has data
    collection_info = rag_client.qdrant.get_collection_info(rag_client.collection_name)
    has_data = collection_info.get('points_count', 0) > 0
    
    with gr.Blocks(
        title="RAG Chat - Python Documentation",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🤖 RAG Chat - Python Documentation Assistant
        
        Ask questions about Python 3.10 documentation! This app uses:
        - **Sentence Transformers** for embeddings (no API key needed!)
        - **Qdrant** vector database for semantic search
        - **Local LLM** (Ollama) or OpenAI for response generation
        """)
        
        if not has_data:
            gr.Markdown("""
            ⚠️ **Note:** The vector database is empty. Please run `python ingest_data.py` first 
            to populate it with Python documentation chunks.
            """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about Python...",
                        label="Your Question",
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    use_rag = gr.Checkbox(value=True, label="Use RAG Context")
            
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                
                sys_prompt = gr.Textbox(
                    value="You are a helpful Python programming assistant. Answer questions based on the Python documentation context provided.",
                    label="System Prompt",
                    lines=4
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=config.TOP_K_RESULTS,
                    step=1,
                    label="Number of Context Chunks"
                )
                
                threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=config.SEARCH_THRESHOLD,
                    step=0.05,
                    label="Similarity Threshold"
                )
                
                gr.Markdown("### Collection")
                collection_dropdown = gr.Dropdown(
                    choices=available_collections,
                    value=rag_client.collection_name,
                    label="Select Collection",
                    interactive=True
                )
                
                info_text = gr.Textbox(
                    value=f"Points: {collection_info.get('points_count', 0)}\n"
                          f"Status: {collection_info.get('status', 'unknown')}",
                    label="Collection Info",
                    interactive=False,
                    lines=2
                )
        
        # Event handlers
        def user_message(user_msg, history):
            """Handle user message submission."""
            if not user_msg.strip():
                return "", history
            history = history or []
            history.append({"role": "user", "content": user_msg})
            return "", history
        
        def bot_response(history, sys_prompt, use_rag, top_k, threshold):
            """Generate bot response."""
            if not history:
                return history
            
            # Ensure sys_prompt is a string
            sys_prompt = str(sys_prompt) if sys_prompt else ""
            
            # Get the last user message, ensuring it's a string
            user_msg = history[-1].get("content", "")
            if not isinstance(user_msg, str):
                user_msg = str(user_msg)
            
            # Convert history to list format for RAG client (excluding current)
            history_pairs = []
            for i in range(0, len(history) - 1, 2):
                if i + 1 < len(history):
                    user_content = history[i].get("content", "")
                    assistant_content = history[i + 1].get("content", "")
                    # Ensure strings
                    if not isinstance(user_content, str):
                        user_content = str(user_content)
                    if not isinstance(assistant_content, str):
                        assistant_content = str(assistant_content)
                    history_pairs.append([user_content, assistant_content])
            
            # Update RAG settings
            config.TOP_K_RESULTS = int(top_k)
            config.SEARCH_THRESHOLD = threshold
            
            # Add placeholder for assistant response
            history.append({"role": "assistant", "content": ""})
            
            # Generate response
            response = ""
            
            try:
                for chunk in rag_client.generate_response_stream(
                    query=user_msg,
                    history=history_pairs,
                    sys_prompt=sys_prompt,
                    use_rag=use_rag
                ):
                    # Ensure chunk is a string
                    if not isinstance(chunk, str):
                        chunk = str(chunk)
                    response += chunk
                    history[-1]["content"] = response
                    yield history
            except Exception as e:
                import traceback
                error_msg = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                history[-1]["content"] = f"Error: {str(e)}"
                yield history
        
        def clear_chat():
            """Clear the chat history."""
            return []
        
        def change_collection(new_collection):
            """Switch to a different collection."""
            rag_client.set_collection(new_collection)
            info = rag_client.qdrant.get_collection_info(new_collection)
            info_str = f"Points: {info.get('points_count', 0)}\nStatus: {info.get('status', 'unknown')}"
            return info_str
        
        # Wire up events
        collection_dropdown.change(
            change_collection,
            [collection_dropdown],
            [info_text]
        )
        
        submit_btn.click(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, sys_prompt, use_rag, top_k, threshold],
            chatbot
        )
        
        msg.submit(
            user_message,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, sys_prompt, use_rag, top_k, threshold],
            chatbot
        )
        
        clear_btn.click(clear_chat, None, chatbot)
        
        # Example questions
        gr.Markdown("### Example Questions")
        examples = gr.Examples(
            examples=[
                ["What is a Python decorator?"],
                ["How do I use list comprehension in Python?"],
                ["Explain the asyncio module."],
                ["What is the difference between a list and a tuple?"],
                ["How do I read and write files in Python?"],
            ],
            inputs=msg
        )
    
    return app


def main():
    """Main entry point for the RAG client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chat Client")
    parser.add_argument(
        "--collection", 
        type=str, 
        default=config.QDRANT_COLLECTION,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        default=config.LLM_PROVIDER,
        choices=["ollama", "openai", "mock"],
        help="LLM provider"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.EMBEDDING_MODEL,
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.GRADIO_SERVER_PORT,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG client
    rag_client = RAGClient(
        collection_name=args.collection,
        llm_provider=args.llm,
        embedding_model=args.embedding_model
    )
    
    # Create and launch app
    app = create_gradio_app(rag_client)
    app.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
