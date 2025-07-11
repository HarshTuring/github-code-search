from typing import Dict, List, Optional, Union, Any
from .query_processor import QueryProcessor
from .retrieval_engine import RetrievalEngine
from .response_generator import ResponseGenerator

class QueryController:
    """
    Main controller for the code query system.
    Orchestrates the entire process from query to response.
    """
 
    def __init__(self, embedding_generator, vector_store, model_name: str = "gpt-4o-mini", default_top_k: int = 5):
        """
        Initialize the query controller with all required components.

        Args:
        embedding_generator: The embedding generator for queries
        vector_store: The vector store for retrieving chunks
        model_name: LLM model for response generation
        default_top_k: Default number of results to retrieve
        """
        # Initialize components
        self.query_processor = QueryProcessor(embedding_generator)
        self.retrieval_engine = RetrievalEngine(vector_store, default_top_k)
        self.response_generator = ResponseGenerator(model=model_name)

    def process_query(self, 
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    include_history: bool = True) -> Dict[str, Any]:
        """
        Process a user query and generate a response.

        Args:
        query_text: The user's natural language query
        filters: Optional filters to narrow the search
        top_k: Number of results to retrieve
        include_history: Whether to include conversation history

        Returns:
        Dict with the response and metadata
        """
        # Step 1: Process the query to generate embeddings
        query_data = self.query_processor.process_query(
        query_text=query_text,
        include_history=include_history,
        filters=filters
        )

        # Step 2: Retrieve relevant code chunks with context
        retrieved_results = self.retrieval_engine.retrieve_with_context(
        query_data=query_data,
        top_k=top_k
        )

        # Step 3: Generate a response with the retrieved chunks
        response_data = self.response_generator.generate_response(
        query=query_text,
        retrieved_results=retrieved_results
        )

        # Step 4: Update conversation history with the response
        if include_history:
            self.query_processor.add_to_history(response_data["response"])

        # Step 5: Return the complete response data
        return {
        "query": query_text,
        "response": response_data["response"],
        "sources": response_data["sources"],
        "retrieved_chunks": len(retrieved_results.get("primary_results", [])),
        "context_chunks": len(retrieved_results.get("context_chunks", [])),
        "used_chunks": response_data["used_chunks"],
        "filters_applied": filters or {}
        }

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.query_processor.clear_history()