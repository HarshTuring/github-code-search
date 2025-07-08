import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
import openai
import numpy as np

# from ..parser.models.chunk import Chunk
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
    
    def __init__(self, 
                 model_name: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None,
                 batch_size: int = 20,
                 max_retries: int = 5,
                 retry_delay: int = 2):
        """
        Initialize the embedding generator with specified parameters.
        
        Args:
            model_name: The OpenAI embedding model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            batch_size: Number of chunks to process in each API call
            max_retries: Maximum number of retry attempts for failed calls
            retry_delay: Base delay between retries (uses exponential backoff)
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
        
        # Set up OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as an argument or as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        logger.info(f"Initialized EmbeddingGenerator with model {model_name} (dimension {self.dimension})")

    def generate_embeddings(self, chunks: List[Chunk], show_progress: bool = True) -> List[Chunk]:
        """
        Generate embeddings for a list of code chunks.
        
        Args:
            chunks: List of code chunk objects
            show_progress: Whether to log progress information
            
        Returns:
            List of chunks with embeddings attached
        """
        print("GENERATE EMBEDDINGS CALLED", "\n\n\n\n")
        if not chunks:
            logger.warning("No chunks provided to generate embeddings")
            return []
        
        total_chunks = len(chunks)
        logger.info(f"Generating embeddings for {total_chunks} chunks using {self.model_name}")
        
        processed_chunks = []
        batch_count = (total_chunks + self.batch_size - 1) // self.batch_size
        
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i+self.batch_size]
            batch_texts = [self._prepare_chunk_text(chunk) for chunk in batch]
            
            if show_progress:
                current_batch = i // self.batch_size + 1
                logger.info(f"Processing batch {current_batch}/{batch_count} ({len(batch)} chunks)")
            
            # Generate embeddings with retry logic
            embeddings = self._generate_batch_with_retry(batch_texts)
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                # Add embedding to the chunk's metadata
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                
                chunk.metadata['embedding'] = embedding
                chunk.metadata['embedding_model'] = self.model_name
                chunk.metadata['embedding_dimension'] = self.dimension
                processed_chunks.append(chunk)
        
        logger.info(f"Successfully generated embeddings for {len(processed_chunks)} chunks")
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
                print("CREATING EMBEDDINGS")
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                print(response.data, "\n")
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
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Useful for generating query embeddings for similarity search.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise