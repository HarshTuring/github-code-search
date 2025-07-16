import os
from typing import Dict, List, Optional, Union, Any

class QueryProcessor:
    """
    Processes natural language queries about code repositories.
    Converts queries to embeddings and prepares them for retrieval.
    """
    
    def __init__(self, embedding_generator, model_name: str = "text-embedding-3-small"):
        """
        Initialize the query processor with the embedding generator.
        
        Args:
            embedding_generator: The embedding generator instance
            model_name: The name of the embedding model to use
        """
        self.embedding_generator = embedding_generator
        self.model_name = model_name
        self.conversation_history = []
        
    def process_query(self, query_text: str, 
                      include_history: bool = True,
                      filters: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Process a natural language query and prepare it for retrieval.
        
        Args:
            query_text: The natural language query text
            include_history: Whether to include conversation history
            filters: Optional filters to narrow the search (language, file type, etc.)
            
        Returns:
            Dictionary containing the processed query data including embeddings
        """
        # Format the query with conversation context if needed
        formatted_query = self._format_query(query_text, include_history)
        
        # Generate embedding for the query
        # Generate embedding for the query using the correct method for single texts.
        query_embedding = self.embedding_generator.embed_text(formatted_query)
        
        # Update conversation history
        self._update_history(query_text)
        
        # Prepare query data with metadata
        query_data = {
            "original_query": query_text,
            "formatted_query": formatted_query,
            "embedding": query_embedding,
            "filters": filters or {},
            "timestamp": self._get_timestamp()
        }
        
        return query_data
    
    def _format_query(self, query_text: str, include_history: bool) -> str:
        """
        Format the query text, optionally including conversation history.
        
        Args:
            query_text: The raw query text
            include_history: Whether to include conversation history
            
        Returns:
            Formatted query text
        """
        if not include_history or not self.conversation_history:
            return query_text
        
        # Include limited relevant history for context
        # Only include last few exchanges to keep the context relevant
        recent_history = self.conversation_history[-3:]
        history_text = "\n".join(recent_history)
        
        return f"Given the previous conversation:\n{history_text}\n\nCurrent question: {query_text}"
    
    def _update_history(self, query_text: str) -> None:
        """
        Update the conversation history with the new query.
        
        Args:
            query_text: The query to add to history
        """
        self.conversation_history.append(f"User: {query_text}")
        
        # Keep history to a reasonable size (last 10 exchanges)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def add_to_history(self, response: str) -> None:
        """
        Add a system response to the conversation history.
        
        Args:
            response: The system response to add
        """
        # Store a summarized version to save context space
        if len(response) > 500:
            summary = response[:250] + "..." + response[-250:]
            self.conversation_history.append(f"System: {summary}")
        else:
            self.conversation_history.append(f"System: {response}")
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()