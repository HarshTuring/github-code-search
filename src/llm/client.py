import os
from openai import OpenAI

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
    
    def generate(self, prompt):
        """Generate a response to the given prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            str: The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or whichever model you're using
                messages=[
                    {"role": "system", "content": "You are a senior software engineer with expertise in code analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_with_system_prompt(self, system_prompt, user_prompt):
        """Generate a response with a custom system prompt.
        
        Args:
            system_prompt (str): The system prompt defining the assistant's role and context
            user_prompt (str): The user's question or prompt
            
        Returns:
            str: The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or whichever model you're using
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")