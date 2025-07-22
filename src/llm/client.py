import os
import time
import random
from typing import Optional, Dict, Any, Union
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

def get_llm_client():
    """
    Get the LLM client for generating responses.
    
    Returns:
        A client object that can generate text.
    """
    return LLMClient()

class LLMClient:
    def __init__(self):
        """Initialize LLM client."""
        # Initialize OpenAI client with API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
    
    def _make_api_call(self, messages: list, max_retries: int = 3, initial_delay: float = 1.0) -> str:
        """Make an API call with retry logic and exponential backoff.
        
        Args:
            messages: List of message dictionaries for the chat completion
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            The generated response content
            
        Raises:
            APIError: If the API request fails after all retries
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1500,
                    request_timeout=30  # 30 seconds timeout
                )
                return response.choices[0].message.content
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exception = e
                if attempt == max_retries:
                    break
                    
                # Exponential backoff with jitter
                sleep_time = delay * (2 ** attempt) * (0.5 + random.random() * 0.5)
                time.sleep(min(sleep_time, 60))  # Cap at 60 seconds
                continue
                
            except Exception as e:
                last_exception = e
                break
        
        # If we get here, all retries failed
        error_msg = self._format_error_message(last_exception, attempt)
        raise APIError(f"API request failed after {max_retries} attempts: {error_msg}")
    
    def _format_error_message(self, error: Exception, attempt: int) -> str:
        """Format an error message based on the exception type."""
        if isinstance(error, RateLimitError):
            return "Rate limit exceeded. Please try again later."
        elif isinstance(error, APITimeoutError):
            return "Request timed out. The server took too long to respond."
        elif isinstance(error, APIConnectionError):
            return "Connection error. Please check your internet connection."
        return f"Error: {str(error)}"
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate a response to the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM.
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            The generated response.
            
        Raises:
            APIError: If the API request fails after all retries
        """
        try:
            messages = [{"role": "system", "content": "You are a senior software engineer with expertise in code analysis."}, {"role": "user", "content": prompt}]
            return self._make_api_call(messages, max_retries)
            
        except Exception as e:
            raise APIError(f"Failed to generate response: {str(e)}")
    
    def generate_with_system_prompt(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Generate a response with a custom system prompt.
        
        Args:
            system_prompt: The system prompt defining the assistant's role and context
            user_prompt: The user's question or prompt
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            The generated response.
            
        Raises:
            APIError: If the API request fails after all retries
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return self._make_api_call(messages, max_retries)
            
        except Exception as e:
            raise APIError(f"Failed to generate response with system prompt: {str(e)}")