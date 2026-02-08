"""
LLM (Large Language Model) module.
Supports multiple providers: Ollama (local) and OpenAI.
"""
from typing import List, Dict, Generator, Optional
from abc import ABC, abstractmethod

import config


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM."""
        pass


class OllamaLLM(BaseLLM):
    """
    Ollama LLM provider.
    Runs locally - no API key required!
    """
    
    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.2", "mistral", "gemma2").
            base_url: Ollama server URL.
        """
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        
        # Set OLLAMA_HOST env var for the ollama library
        import os
        os.environ["OLLAMA_HOST"] = self.base_url
        
        try:
            import ollama
            self._client = ollama.Client(host=self.base_url)
            # Test connection
            self._client.list()
            print(f"Connected to Ollama at {self.base_url}. Using model: {self.model}")
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )
    
    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
        Generate a response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            stream: Whether to stream (ignored, use generate_stream instead).
            
        Returns:
            The generated response text.
        """
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=False
        )
        return response['message']['content']
    
    def generate_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Generate a streaming response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Yields:
            Chunks of the generated response.
        """
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider.
    Requires OPENAI_API_KEY environment variable or config.
    """
    
    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4").
            api_key: OpenAI API key.
        """
        self.model = model or config.OPENAI_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or update config.py"
            )
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            print(f"Connected to OpenAI. Using model: {self.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
        Generate a response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            stream: Whether to stream (ignored, use generate_stream instead).
            
        Returns:
            The generated response text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    
    def generate_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Generate a streaming response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Yields:
            Chunks of the generated response.
        """
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockLLM(BaseLLM):
    """
    Mock LLM for testing when no LLM is available.
    """
    
    def __init__(self):
        print("Using Mock LLM - responses will be placeholders")
    
    def generate(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """Return a mock response."""
        last_message = messages[-1]['content'] if messages else "nothing"
        return f"[Mock LLM Response] This is a placeholder response to: {last_message[:100]}..."
    
    def generate_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Yield mock response in chunks."""
        response = self.generate(messages)
        words = response.split()
        for word in words:
            yield word + " "


def get_llm(provider: str = None) -> BaseLLM:
    """
    Get an LLM instance based on the specified provider.
    
    Args:
        provider: LLM provider ("ollama", "openai", "mock").
                 Defaults to config.LLM_PROVIDER.
    
    Returns:
        An LLM instance.
    """
    provider = provider or config.LLM_PROVIDER
    
    if provider.lower() == "ollama":
        try:
            return OllamaLLM()
        except Exception as e:
            print(f"Failed to initialize Ollama: {e}")
            print("Falling back to Mock LLM")
            return MockLLM()
    
    elif provider.lower() == "openai":
        try:
            return OpenAILLM()
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            print("Falling back to Mock LLM")
            return MockLLM()
    
    elif provider.lower() == "mock":
        return MockLLM()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def make_inputs(sys_prompt: str, history: List[List[str]]) -> List[Dict[str, str]]:
    """
    Create input messages from system prompt and chat history.
    Compatible function signature with the lab exercises.
    
    Args:
        sys_prompt: System prompt text.
        history: Chat history as list of [user_msg, assistant_msg] pairs.
        
    Returns:
        List of message dicts for the LLM.
    """
    messages = []
    
    if sys_prompt:
        # Ensure sys_prompt is a string
        sys_prompt_str = str(sys_prompt) if not isinstance(sys_prompt, str) else sys_prompt
        messages.append({"role": "system", "content": sys_prompt_str})
    
    for pair in history:
        user_msg, assistant_msg = pair
        # Ensure all messages are strings
        user_msg_str = str(user_msg) if not isinstance(user_msg, str) else user_msg
        messages.append({"role": "user", "content": user_msg_str})
        if assistant_msg:
            assistant_msg_str = str(assistant_msg) if not isinstance(assistant_msg, str) else assistant_msg
            messages.append({"role": "assistant", "content": assistant_msg_str})
    
    return messages


def make_context_inputs(sys_prompt: str, 
                        history: List[List[str]], 
                        context: List[str]) -> List[Dict[str, str]]:
    """
    Create input messages with context for RAG.
    Compatible function signature with the lab exercises.
    
    Args:
        sys_prompt: System prompt text.
        history: Chat history as list of [user_msg, assistant_msg] pairs.
        context: List of context chunks to include.
        
    Returns:
        List of message dicts for the LLM.
    """
    from copy import deepcopy
    
    history_ctx = deepcopy(history)
    
    if not history_ctx:
        return make_inputs(sys_prompt, history)
    
    prompt = history_ctx[-1][0]
    if not isinstance(prompt, str):
        prompt = str(prompt)
    
    # Ensure all context items are strings
    context_strs = [str(c) if not isinstance(c, str) else c for c in context]
    
    # Build context-augmented prompt
    prompt_ctx = (
        "Using the following information, enclosed within three apostrophes "
        "(''') at the beginning and end, respond to the request that follows "
        "to the best of your ability.\n\n'''\n"
    )
    prompt_ctx += "\n\n".join(context_strs)
    prompt_ctx += "\n'''\n\n"
    prompt_ctx += prompt
    
    history_ctx[-1][0] = prompt_ctx
    
    return make_inputs(sys_prompt, history_ctx)


if __name__ == "__main__":
    # Test LLM
    print("Testing LLM module...")
    
    # Try to get an LLM
    llm = get_llm("mock")  # Use mock for testing
    
    # Test generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
    
    response = llm.generate(messages)
    print(f"\nGenerate response:\n{response}")
    
    # Test streaming
    print("\nStreaming response:")
    for chunk in llm.generate_stream(messages):
        print(chunk, end="", flush=True)
    print()
