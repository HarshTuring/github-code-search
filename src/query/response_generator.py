import os
import json
from typing import Dict, List, Optional, Union, Any
import openai

class ResponseGenerator:
    """
    Generates natural language responses to code questions with proper citations.
    Uses a two-pass approach for handling complex queries across multiple code chunks.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the response generator with OpenAI configuration.

        Args:
        api_key: OpenAI API key (will use env var if None)
        model: LLM model to use
        temperature: Response temperature (lower for more precise responses)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as argument.")

        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=self.api_key)

        # Templates for different prompt parts
        self.system_template = self._get_system_template()
        self.chunk_template = self._get_chunk_template()
        self.query_template = self._get_query_template()

    def generate_response(self, query: str, retrieved_results: Dict[str, Any], max_chunks: int = 10) -> Dict[str, Any]:
        """
        Generate a response to a code query using the two-pass approach.
 
        Args:
        query: The original query string
        retrieved_results: Results from the retrieval engine
        max_chunks: Maximum number of chunks to include
        
        Returns:
        Dict containing the response and citation information
        """
        # Combine primary results and context chunks
        primary_results = retrieved_results.get("primary_results", [])
        context_chunks = retrieved_results.get("context_chunks", [])
 
            # Limit the number of chunks to prevent token overflow
        all_chunks = primary_results[:max_chunks//2]
        if len(all_chunks) < max_chunks:
            all_chunks.extend(context_chunks[:max_chunks - len(all_chunks)])
 
        # If no chunks found, return a fallback response
        if not all_chunks:
            return self._generate_fallback_response(query)
 
        # First pass: Determine relevant chunks and their importance to the query
        relevance_map = self._first_pass_analysis(query, all_chunks)
 
        # Sort chunks by relevance score
        relevant_chunks = sorted(
            relevance_map["chunk_relevance"],
            key=lambda x: x["relevance_score"],
            reverse=True
        )
 
        # Get the most relevant chunks for the second pass
        top_chunks = [chunk for chunk in all_chunks 
        if chunk["id"] in [rc["chunk_id"] for rc in relevant_chunks[:max_chunks//2]]]
        
        # Second pass: Generate comprehensive answer with proper citations
        response = self._second_pass_generation(query, top_chunks, relevance_map["analysis"])
        
        return {
        "response": response["content"],
        "sources": self._format_sources(top_chunks),
        "analysis": relevance_map["analysis"],
        "used_chunks": [chunk["id"] for chunk in top_chunks]
        }
 
    def _first_pass_analysis(self, query: str, chunks: List[Dict]) -> Dict:
        """
        First pass: Analyze which chunks are most relevant to answering the query.
 
        Args:
        query: The original query
        chunks: List of code chunks
 
        Returns:
        Dict with relevance analysis and scores
        """
        # Format chunks for the prompt
        formatted_chunks = self._format_chunks_for_prompt(chunks)
 
        # Create the first pass prompt
        first_pass_prompt = f"""
        Analyze which of the following code chunks are relevant to answering this query: "{query}"

        For each chunk, assign a relevance score from 0-10 where:
        - 10: Critical to answering the question
        - 7-9: Very relevant with specific information needed
        - 4-6: Somewhat relevant with context or related information
        - 1-3: Marginally related but not directly helpful
        - 0: Not relevant to this query

        {formatted_chunks}

        Respond in valid JSON format with:
        1. "analysis": Brief explanation of which chunks are most relevant and why
        2. "chunk_relevance": Array of objects with "chunk_id" and "relevance_score"
        """
        
        # Get response from LLM
        response = self._call_llm(first_pass_prompt, system_prompt="""
        You are an expert code analyst. Your task is to analyze code chunks and determine which are most relevant to a user's query.
        Respond ONLY with valid JSON format that can be parsed with json.loads(). 
        Include a brief analysis and relevance scores for each chunk.
        """, response_format={"type": "json_object"})
        
        # Parse the response
        try:
            relevance_data = json.loads(response)
            # Ensure proper format
            if not isinstance(relevance_data.get("chunk_relevance"), list):
                relevance_data["chunk_relevance"] = []
            if not relevance_data.get("analysis"):
                relevance_data["analysis"] = "Unable to determine chunk relevance."
                return relevance_data
        except (json.JSONDecodeError, AttributeError):
            # Fallback if response isn't valid JSON
            return {
            "analysis": "Error analyzing chunks. Using all available chunks.",
            "chunk_relevance": [
            {"chunk_id": chunk["id"], "relevance_score": 5} 
            for chunk in chunks
            ]
            }
 
    def _second_pass_generation(self, 
    query: str, 
    chunks: List[Dict],
    analysis: str) -> Dict:
        """
        Second pass: Generate a comprehensive answer with citations.
        
        Args:
        query: The original query
        chunks: Relevant code chunks to reference
        analysis: First-pass analysis of chunk relevance
        
        Returns:
        Dict with generated response
        """
        # Format chunks with source information
        formatted_chunks = self._format_chunks_for_prompt(chunks)
        
        # Prepare the system prompt
        system_prompt = self.system_template
        
        # Prepare the user prompt
        user_prompt = f"""
        I'll answer a question about a codebase using the provided code chunks.

        CHUNK RELEVANCE ANALYSIS:
        {analysis}

        CODE CHUNKS:
        {formatted_chunks}

        USER QUERY: {query}
        """
        
        # Call the LLM with the full context
        response = self._call_llm(user_prompt, system_prompt=system_prompt)
        
        return {"content": response}
        
    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """
        Format code chunks for inclusion in prompts.
        
        Args:
        chunks: List of code chunks with metadata
        
        Returns:
        Formatted string representation
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            chunk_id = chunk.get("id", f"chunk_{i}")
        
        # Get key metadata
        path = metadata.get("path", "unknown_file")
        chunk_type = metadata.get("chunk_type", "unknown")
        language = metadata.get("language", "unknown")
        signature = metadata.get("signature", "")
        
        formatted_chunk = self.chunk_template.format(
        chunk_id=chunk_id,
        path=path,
        chunk_type=chunk_type,
        language=language,
        signature=signature,
        content=chunk.get("content", "No content available")
        )
        
        formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
        
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Format source information for the response metadata.
        
        Args:
        chunks: List of chunks used in the response
        
        Returns:
        List of formatted source references
        """
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            sources.append({
            "id": chunk.get("id"),
            "path": metadata.get("path", "unknown_file"),
            "chunk_type": metadata.get("chunk_type", "unknown"),
            "signature": metadata.get("signature", "")
            })
        return sources
    
    def _generate_fallback_response(self, query: str) -> Dict:
        """
        Generate a fallback response when no relevant chunks are found.
        
        Args:
        query: The original query
        
        Returns:
        Dict with fallback response
        """
        fallback_prompt = f"""
        I couldn't find specific code in the repository that addresses this query: "{query}"

        Please provide a general response that acknowledges the lack of specific information
        and offers general guidance or suggestions about what might be involved.
        """
        
        response = self._call_llm(fallback_prompt, 
        system_prompt="You are a helpful code assistant. When specific code can't be found, provide general guidance.")
        
        return {
        "response": response,
        "sources": [],
        "analysis": "No relevant code chunks found in the repository.",
        "used_chunks": []
        }
    
    def _call_llm(self, 
    user_prompt: str, 
    system_prompt: str = None,
    response_format: Dict = None) -> str:
        """
        Call the LLM with the given prompts.
        
        Args:
        user_prompt: The user's prompt
        system_prompt: SystemPrompt for the LLM
        response_format: Optional response format specification
        
        Returns:
        LLM response as a string
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
        "role": "user",
        "content": user_prompt
        })
        
        # Prepare the API call parameters
        params = {
        "model": self.model,
        "messages": messages,
        "temperature": self.temperature,
        }
        
        # Add response_format if specified
        if response_format:
            params["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def _get_system_template(self) -> str:
        """Get the system prompt template."""
        return """You are an expert code assistant that explains code repositories with precision and clarity.

        When answering questions about code:
        1. Directly address the user's question with a clear, concise answer
        2. Include specific file references using [filename:line] or [filename:function_name] format
        3. Explain how the code works with accurate technical details
        4. When referencing multiple files, clarify how they connect
        5. Include short relevant code snippets with proper attribution

        Your response should follow this structure:
        - Brief summary answer (1-2 sentences)
        - Detailed explanation with inline citations
        - Code examples with source attribution
        - Clear connections between components when relevant

        Always prioritize accuracy over comprehensiveness. If you're uncertain about any detail, acknowledge the limitation rather than speculating."""
    
    def _get_chunk_template(self) -> str:
        """Get the code chunk template."""
        return """CHUNK ID: {chunk_id}
        FILE: {path}
        TYPE: {chunk_type}
        LANGUAGE: {language}
        SIGNATURE: {signature}

        CONTENT:
        {content}
        """
    
    def _get_query_template(self) -> str:
        """Get the query template."""
        return """USER QUERY: {query}

        Based on the code chunks provided, please answer this query with specific references to the codebase.
        Include file paths and function names in your explanations.
        """