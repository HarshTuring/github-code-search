import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union
import openai
import numpy as np

# Fix the import path
from parser.models.chunk import Chunk

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for code chunks using OpenAI's embedding models.
    
    This class handles:
    - Preparing chunk data with necessary metadata
    - Batching requests to the OpenAI API
    - Error handling and retries
    - Attaching embeddings to chunk objects
    """
    
    # Model dimensions map
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536  # Legacy model
    }
    
    # Token limits
    MODEL_TOKEN_LIMITS = {
        "text-embedding-3-small": 8192,
        "text-embedding-3-large": 8192,
        "text-embedding-ada-002": 8191
    }
    
    def __init__(self, 
                 model_name: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None,
                 batch_size: int = 20,
                 max_retries: int = 5,
                 retry_delay: int = 2,
                 max_tokens: Optional[int] = None):
        """
        Initialize the embedding generator with specified parameters.
        
        Args:
            model_name: The OpenAI embedding model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            batch_size: Number of chunks to process in each API call
            max_retries: Maximum number of retry attempts for failed calls
            retry_delay: Base delay between retries (uses exponential backoff)
            max_tokens: Maximum tokens to allow (defaults to model's limit)
        """
        self.model_name = model_name
        
        # Verify model is supported
        if model_name not in self.MODEL_DIMENSIONS:
            supported = list(self.MODEL_DIMENSIONS.keys())
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported}")
        
        self.dimension = self.MODEL_DIMENSIONS[model_name]
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set token limit
        self.max_tokens = max_tokens or self.MODEL_TOKEN_LIMITS.get(model_name, 8000)
        # Add a safety margin
        self.max_tokens = int(self.max_tokens * 0.95)
        
        # Set up OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as an argument or as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        logger.info(f"Initialized EmbeddingGenerator with model {model_name} (dimension {self.dimension}, max tokens {self.max_tokens})")

    def generate_embeddings(self, chunks: List[Chunk], show_progress: bool = True) -> List[Chunk]:
        """
        Generate embeddings for a list of code chunks.
        
        Args:
            chunks: List of code chunk objects
            show_progress: Whether to log progress information
            
        Returns:
            List of chunks with embeddings attached
        """
        if not chunks:
            logger.warning("No chunks provided to generate embeddings")
            return []
        
        total_chunks = len(chunks)
        logger.info(f"Generating embeddings for {total_chunks} chunks using {self.model_name}")
        
        processed_chunks = []
        batch_count = (total_chunks + self.batch_size - 1) // self.batch_size
        skipped_chunks = 0
        
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i+self.batch_size]
            batch_texts = []
            valid_indices = []
            
            # Prepare texts and track which ones are valid
            for j, chunk in enumerate(batch):
                try:
                    text = self._prepare_chunk_text(chunk)
                    # Estimate tokens and skip if too large
                    est_tokens = self._estimate_tokens(text)
                    if est_tokens > self.max_tokens:
                        # Try to truncate
                        text = self._truncate_text(text, self.max_tokens)
                        est_tokens = self._estimate_tokens(text)
                        if est_tokens > self.max_tokens:
                            logger.warning(f"Skipping chunk {chunk.id} ({chunk.chunk_type}) - too large even after truncation: ~{est_tokens} tokens")
                            continue
                    
                    batch_texts.append(text)
                    valid_indices.append(j)
                except Exception as e:
                    logger.error(f"Error preparing chunk {chunk.id}: {str(e)}")
                    continue
            
            if not batch_texts:
                # Skip this batch if all chunks were filtered out
                skipped_chunks += len(batch)
                continue
            
            if show_progress:
                current_batch = i // self.batch_size + 1
                logger.info(f"Processing batch {current_batch}/{batch_count} ({len(batch_texts)}/{len(batch)} chunks)")
            
            try:
                # Generate embeddings with retry logic
                embeddings = self._generate_batch_with_retry(batch_texts)
                
                # Attach embeddings to chunks
                for idx, embedding in zip(valid_indices, embeddings):
                    chunk = batch[idx]
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                        chunk.metadata = {}
                    
                    chunk.metadata['embedding'] = embedding
                    chunk.metadata['embedding_model'] = self.model_name
                    chunk.metadata['embedding_dimension'] = self.dimension
                    processed_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                # Add these chunks without embeddings
                for idx in valid_indices:
                    processed_chunks.append(batch[idx])
        
        chunks_with_embeddings = sum(1 for chunk in processed_chunks if chunk.metadata and 'embedding' in chunk.metadata)
        logger.info(f"Generated embeddings for {chunks_with_embeddings}/{total_chunks} chunks")
        if skipped_chunks > 0:
            logger.warning(f"Skipped {skipped_chunks} chunks due to size limitations")
        
        return processed_chunks
    
    def _generate_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                # Extract embeddings in the same order as input texts
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} retries")
                    raise
                
                # Exponential backoff
                sleep_time = self.retry_delay * (2 ** (retry_count - 1))
                logger.warning(f"Rate limit or connection error: {str(e)}. Retrying in {sleep_time}s (attempt {retry_count})")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise
    
    def _prepare_chunk_text(self, chunk: Chunk) -> str:
        """
        Format chunk content with metadata for more contextual embeddings.
        
        Args:
            chunk: A code chunk object
            
        Returns:
            Formatted text string combining metadata and code content
        """
        # Start with chunk type and language as a header
        formatted_text = f"TYPE: {chunk.chunk_type} | LANGUAGE: {chunk.language or 'unknown'}\n"
        
        # Add file path
        formatted_text += f"PATH: {chunk.file_path}\n"
        
        # Add name
        if chunk.name:
            formatted_text += f"NAME: {chunk.name}\n"
        
        # Add line numbers if available
        if chunk.start_line and chunk.end_line:
            formatted_text += f"LINES: {chunk.start_line}-{chunk.end_line}\n"
        
        # Add metadata that might be useful
        if hasattr(chunk, 'metadata') and chunk.metadata:
            # Extract parent classes if available
            parent_classes = chunk.metadata.get('parent_classes')
            if parent_classes:
                if isinstance(parent_classes, list):
                    parent_str = '.'.join(parent_classes)
                else:
                    parent_str = str(parent_classes)
                formatted_text += f"PARENT: {parent_str}\n"
            
            # Add signature if available
            signature = chunk.metadata.get('signature')
            if signature:
                formatted_text += f"SIGNATURE: {signature}\n"
            
            # Include imports/dependencies
            imports = chunk.metadata.get('imports')
            if imports:
                if len(imports) <= 5:
                    imports_text = ', '.join(imports)
                else:
                    imports_text = ', '.join(imports[:5]) + f" and {len(imports)-5} more"
                formatted_text += f"IMPORTS: {imports_text}\n"
            
            # Add docstring if available
            docstring = chunk.metadata.get('docstring')
            if docstring:
                # Truncate long docstrings
                doc = docstring if len(docstring) < 200 else docstring[:197] + "..."
                formatted_text += f"DOC: {doc}\n"
        
        # Finally add the actual code content
        formatted_text += f"CODE:\n{chunk.content}"
        
        return formatted_text
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # A very rough approximation: 1 token ≈ 4 characters for English text
        # This will overestimate for code which is good for our safety margin
        return len(text) // 3
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within the token limit while preserving metadata.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text
        """
        # Split into metadata and code sections
        parts = text.split("CODE:\n", 1)
        if len(parts) != 2:
            # If we can't split properly, just truncate from the end
            return text[:max_tokens*3]
        
        metadata, code = parts
        
        # Estimate tokens in metadata
        metadata_tokens = self._estimate_tokens(metadata)
        
        # Calculate how many tokens we have left for code
        remaining_tokens = max_tokens - metadata_tokens - 10  # 10 token buffer
        
        if remaining_tokens <= 0:
            # Even metadata is too long, truncate it
            return text[:max_tokens*3]
        
        # Truncate code based on remaining tokens
        max_code_chars = remaining_tokens * 3
        
        if len(code) > max_code_chars:
            truncated_code = code[:max_code_chars] 
            return metadata + "CODE:\n" + truncated_code + "\n[TRUNCATED due to size]"
        
        return text
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Useful for generating query embeddings for similarity search.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        # Estimate tokens and truncate if needed
        est_tokens = self._estimate_tokens(text)
        if est_tokens > self.max_tokens:
            logger.warning(f"Text too large (~{est_tokens} tokens), truncating to fit {self.max_tokens} token limit")
            text = text[:self.max_tokens*3]  # Simple truncation for queries
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise