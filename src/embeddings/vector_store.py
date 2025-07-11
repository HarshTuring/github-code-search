import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import shutil
import json

# Import ChromaDB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Import your existing Chunk model
from parser.models.chunk import Chunk

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages storage and retrieval of code chunk embeddings using ChromaDB.
    
    This class provides a persistent vector database for storing embeddings
    with one collection per repository.
    """
    
    def __init__(self, 
                 repo_name: str, 
                 base_dir: str = "./data/embeddings",
                 create_if_missing: bool = True):
        """
        Initialize a vector store for a specific repository.
        
        Args:
            repo_name: Name of the repository (used as collection name)
            base_dir: Base directory for storing embeddings
            create_if_missing: Whether to create the collection if it doesn't exist
        """
        self.repo_name = repo_name
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / f"{repo_name}.chromadb"
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False  # Disable telemetry for privacy
            )
        )
        
        # Get or create collection for this repository
        if create_if_missing:
            self.collection = self.client.get_or_create_collection(
                name=repo_name,
                metadata={"repo": repo_name}
            )
            logger.info(f"Opened or created collection for {repo_name}")
        else:
            try:
                self.collection = self.client.get_collection(name=repo_name)
                logger.info(f"Opened existing collection for {repo_name}")
            except ValueError:
                logger.error(f"Collection {repo_name} does not exist and create_if_missing is False")
                raise
        
        # Track statistics
        self.add_count = 0
        self.query_count = 0
    
    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of code chunks with embeddings in their metadata
            
        Returns:
            List of added chunk IDs
        """
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return []
        
        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.metadata and 'embedding' in c.metadata]
        
        if len(valid_chunks) != len(chunks):
            logger.warning(f"Only {len(valid_chunks)}/{len(chunks)} chunks have embeddings and will be added")
        
        if not valid_chunks:
            return []
        
        # Prepare batch data for ChromaDB
        ids = [chunk.id for chunk in valid_chunks]
        embeddings = [chunk.metadata['embedding'] for chunk in valid_chunks]
        
        # Prepare metadata for each chunk
        metadatas = []
        documents = []
        
        for chunk in valid_chunks:
            # Extract relevant metadata, excluding the embedding itself
            metadata = {
                "chunk_type": chunk.chunk_type,
                "language": chunk.language or "unknown",
                "file_path": chunk.file_path,
                "name": chunk.name or "",
            }
            
            # Add line numbers if available
            if chunk.start_line and chunk.end_line:
                metadata["start_line"] = chunk.start_line
                metadata["end_line"] = chunk.end_line
            
            # Add additional metadata from the chunk
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    # Skip the embedding itself
                    if key == 'embedding' or key == 'embedding_dimension' or key == 'embedding_model':
                        continue
                    
                    # Ensure metadata values are strings or numbers for ChromaDB compatibility
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                        # For simple lists of primitive types, join as string
                        metadata[key] = ", ".join(str(x) for x in value)
                    elif value is not None:
                        # Convert other types to string representation
                        metadata[key] = str(value)
            
            metadatas.append(metadata)
            documents.append(chunk.content)
        
        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            self.add_count += len(valid_chunks)
            logger.info(f"Added {len(valid_chunks)} chunks to collection {self.repo_name}")
            
            # Return the added chunk IDs
            return ids
            
        except Exception as e:
            logger.error(f"Error adding chunks to collection: {e}")
            raise
    
    def query_similar(self, 
                      embedding: List[float], 
                      top_k: int = 5, 
                      filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Find chunks similar to the given embedding vector.
        
        Args:
            embedding: The query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter criteria
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            self.query_count += 1
            
            # Format results for easier consumption
            formatted_results = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "score": results['distances'][0][i] if 'distances' in results else None,
                        "metadata": results['metadatas'][0][i],
                        "content": results['documents'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying similar chunks: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents", "embeddings"]
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return {
                    "id": results['ids'][0],
                    "metadata": results['metadatas'][0],
                    "content": results['documents'][0],
                    "embedding": results['embeddings'][0] if 'embeddings' in results else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about this vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            count = self.collection.count()
            return {
                "repo_name": self.repo_name,
                "chunk_count": count,
                "path": str(self.db_path),
                "add_operations": self.add_count,
                "query_operations": self.query_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "repo_name": self.repo_name,
                "error": str(e)
            }
    
    def delete(self) -> bool:
        """
        Delete this vector store completely.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First try to delete using ChromaDB API
            self.client.delete_collection(self.repo_name)
            
            # Also delete the directory to ensure clean removal
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
                
            logger.info(f"Deleted vector store for {self.repo_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB handles backend cleanup automatically
        # Just log the closure for tracking
        logger.info(f"Closed vector store for {self.repo_name}")


class VectorStoreManager:
    """
    Manages multiple vector stores with lifecycle management.
    
    This class prevents memory issues when working with many repositories
    by limiting the number of active collections.
    """
    
    def __init__(self, base_dir: str = "./data/embeddings", max_active: int = 5):
        """
        Initialize the vector store manager.
        
        Args:
            base_dir: Base directory for storing embeddings
            max_active: Maximum number of active vector stores to keep in memory
        """
        self.base_dir = Path(base_dir)
        self.max_active = max_active
        self.active_stores = {}  # repo_name -> VectorStore
        self.access_log = []     # List of repo_names in order of access
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VectorStoreManager with base directory {base_dir}")
    
    def get_store(self, repo_name: str, create_if_missing: bool = True) -> VectorStore:
        """
        Get a vector store for a repository, loading it if necessary.
        
        Args:
            repo_name: Name of the repository
            create_if_missing: Whether to create the store if it doesn't exist
            
        Returns:
            VectorStore instance for the repository
        """
        # If already active, just update access log and return
        if repo_name in self.active_stores:
            self._update_access_log(repo_name)
            return self.active_stores[repo_name]
        
        # If we've reached max active, unload the least recently used
        if len(self.active_stores) >= self.max_active:
            self._unload_least_recently_used()
        
        # Load/create the store
        try:
            store = VectorStore(repo_name, str(self.base_dir), create_if_missing)
            self.active_stores[repo_name] = store
            self._update_access_log(repo_name)
            return store
        except Exception as e:
            logger.error(f"Error loading vector store for {repo_name}: {e}")
            raise
    
    def list_repositories(self) -> List[str]:
        """
        List all repositories with vector stores.
        
        Returns:
            List of repository names
        """
        repos = []
        
        # Check all .chromadb directories in the base directory
        for item in self.base_dir.glob("*.chromadb"):
            if item.is_dir():
                repo_name = item.name.replace(".chromadb", "")
                repos.append(repo_name)
        
        return repos
    
    def close_all(self) -> None:
        """Close all active vector stores."""
        for repo_name, store in self.active_stores.items():
            try:
                store.close()
            except Exception as e:
                logger.error(f"Error closing vector store for {repo_name}: {e}")
        
        self.active_stores = {}
        self.access_log = []
    
    def _update_access_log(self, repo_name: str) -> None:
        """Update the access log for a repository."""
        # Remove if already in log
        if repo_name in self.access_log:
            self.access_log.remove(repo_name)
        
        # Add to front of log (most recently used)
        self.access_log.insert(0, repo_name)
    
    def _unload_least_recently_used(self) -> bool:
        """
        Unload the least recently used vector store.
        
        Returns:
            True if a store was unloaded, False otherwise
        """
        if not self.access_log:
            return False
        
        # Get least recently used repo
        repo_name = self.access_log.pop()
        
        # Unload it if active
        if repo_name in self.active_stores:
            try:
                self.active_stores[repo_name].close()
                del self.active_stores[repo_name]
                logger.info(f"Unloaded vector store for {repo_name} (least recently used)")
                return True
            except Exception as e:
                logger.error(f"Error unloading vector store for {repo_name}: {e}")
        
        return False