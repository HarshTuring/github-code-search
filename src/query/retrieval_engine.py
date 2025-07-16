from typing import Dict, List, Optional, Union, Any
import numpy as np

class RetrievalEngine:
    """
    Retrieval engine that searches for relevant code chunks using vector similarity.
    Uses ChromaDB to perform efficient similarity search on code embeddings.
    """
    
    def __init__(self, vector_store, default_top_k: int = 5):
        """
        Initialize the retrieval engine with a vector store.
        
        Args:
            vector_store: The ChromaDB vector store instance
            default_top_k: Default number of results to return
        """
        self.vector_store = vector_store
        self.default_top_k = default_top_k
    
    def retrieve(self, 
                query_data: Dict[str, Any], 
                top_k: Optional[int] = None, 
                threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant code chunks for a query.
        
        Args:
            query_data: The processed query data containing embeddings
            top_k: Number of results to return (uses default if None)
            threshold: Similarity threshold for results (0.0 to 1.0)
            
        Returns:
            List of relevant code chunks with similarity scores
        """
        import logging
        logger = logging.getLogger(__name__)
        
        embedding = query_data.get("embedding")
        if not embedding:
            raise ValueError("Query data is missing embedding")
        
        # Log the query text and embedding shape for debugging
        query_text = query_data.get("query_text", "[no query text]")
        logger.debug(f"Processing query: {query_text}")
        logger.debug(f"Embedding shape: {len(embedding) if embedding else 0} dimensions")
        
        # Apply filters from query
        filters = query_data.get("filters", {})
        if filters:
            logger.debug(f"Applying filters: {filters}")
        
        # Set the number of results to retrieve
        k = top_k if top_k is not None else self.default_top_k
        
        # Perform similarity search in ChromaDB
        results = self._perform_search(embedding, filters, k)
        
        # Process and filter results
        processed_results = self._process_results(results, threshold)
        
        return processed_results
    
    def _perform_search(self, embedding, filters, k):
        """
        Perform the actual similarity search in ChromaDB.
        
        Args:
            embedding: The query embedding vector
            filters: Filters to apply to the search
            k: Number of results to return
            
        Returns:
            Raw results from ChromaDB
        """
        # Convert filters to ChromaDB format
        chroma_filters = self._format_filters(filters)
        
        # Perform the search and include all metadata fields
        results = self.vector_store.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=chroma_filters if chroma_filters else None,
            include=["metadatas", "documents", "distances", "embeddings"]
        )
        
        return results
    
    def _format_filters(self, filters: Dict) -> Optional[Dict]:
        """
        Convert query filters to ChromaDB filter format.
        
        Args:
            filters: The filter dictionary from the query
            
        Returns:
            ChromaDB compatible filters or None
        """
        if not filters:
            return None
        
        chroma_filters = {}
        
        # Process language filter
        if "language" in filters:
            chroma_filters["language"] = {"$eq": filters["language"]}
        
        # Process file type filter
        if "file_type" in filters:
            chroma_filters["file_type"] = {"$eq": filters["file_type"]}
        
        # Process chunk type filter (function, class, etc.)
        if "chunk_type" in filters:
            chroma_filters["chunk_type"] = {"$eq": filters["chunk_type"]}
        
        # Process path filter (exact or prefix match)
        if "path" in filters:
            if filters.get("path_match_type") == "prefix":
                chroma_filters["path"] = {"$contains": filters["path"]}
            else:
                chroma_filters["path"] = {"$eq": filters["path"]}
        
        return chroma_filters
    
    def _process_results(self, results, threshold):
        """
        Process raw ChromaDB results into a more usable format.
        Filter results below the similarity threshold.
        
        Args:
            results: Raw results from ChromaDB
            threshold: Similarity threshold
            
        Returns:
            Processed and filtered results
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Processing results with threshold: {threshold}")
        
        if not results:
            logger.debug("No results returned from ChromaDB")
            return []
            
        if "ids" not in results:
            logger.debug(f"Unexpected results format: {results.keys()}")
            return []
        
        processed_results = []
        
        # ChromaDB returns results for one query at index 0
        try:
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents = results.get("documents", [[]])[0]
            
            logger.debug(f"Found {len(ids)} results from ChromaDB")
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error processing ChromaDB results: {e}")
            logger.debug(f"Results structure: {results.keys()}")
            if "metadatas" in results and results["metadatas"]:
                logger.debug(f"Metadatas structure: {[type(m) for m in results['metadatas']]}")
            return []
        
        # Convert distance to similarity score (ChromaDB returns distances, not similarities)
        similarities = [1.0 / (1.0 + dist) for dist in distances]  # Using inverse of (1 + distance) for better similarity
        
        # Log the similarity scores for debugging
        logger.debug(f"Similarity scores: {similarities}")
        
        # Combine results with their similarity scores
        for i in range(len(ids)):
            # Log each result's similarity score and threshold comparison
            logger.debug(f"Result {i}: similarity={similarities[i]:.4f}, threshold={threshold}")
            
            # Skip results below threshold
            if similarities[i] < threshold:
                logger.debug(f"Skipping result {i} (similarity {similarities[i]:.4f} < {threshold})")
                continue
                
            result = {
                "id": ids[i],
                "similarity": similarities[i],
                "metadata": metadatas[i],
                "content": documents[i] if documents and i < len(documents) else "[No content]"
            }
            logger.debug(f"Including result {i} with metadata: {metadatas[i].get('path', 'unknown')}")
            
            processed_results.append(result)
        
        return processed_results
    
    def retrieve_with_context(self, 
                             query_data: Dict[str, Any], 
                             top_k: Optional[int] = None,
                             context_depth: int = 1) -> Dict[str, Any]:
        """
        Retrieve relevant code chunks with their contextual relationships.
        This builds a more complete picture by including related chunks.
        
        Args:
            query_data: The processed query data
            top_k: Number of initial results to consider
            context_depth: How many levels of relationships to include
            
        Returns:
            Dict with primary results and contextual chunks
        """
        # Get initial results
        primary_results = self.retrieve(query_data, top_k)
        
        # If no results, return empty context
        if not primary_results:
            return {"primary_results": [], "context_chunks": []}
        
        # Get context chunks
        context_chunks = self._get_context_chunks(primary_results, context_depth)
        
        return {
            "primary_results": primary_results,
            "context_chunks": context_chunks
        }
    
    def _get_context_chunks(self, primary_results, context_depth):
        """
        Get context chunks related to the primary results.
        
        Args:
            primary_results: The primary search results
            context_depth: How many levels of relationships to include
            
        Returns:
            List of context chunks
        """
        context_chunks = []
        processed_ids = set(result["id"] for result in primary_results)
        
        # For each primary result, get related chunks
        for result in primary_results:
            metadata = result["metadata"]
            
            # Get parent chunks if applicable
            if metadata.get("parent_id") and context_depth > 0:
                parent_chunks = self._get_chunk_by_id(metadata["parent_id"])
                for chunk in parent_chunks:
                    if chunk["id"] not in processed_ids:
                        context_chunks.append(chunk)
                        processed_ids.add(chunk["id"])
            
            # Get child chunks if applicable
            if metadata.get("has_children") and context_depth > 0:
                child_chunks = self._get_child_chunks(result["id"])
                for chunk in child_chunks:
                    if chunk["id"] not in processed_ids:
                        context_chunks.append(chunk)
                        processed_ids.add(chunk["id"])
            
            # Get imported modules/related chunks
            if metadata.get("related_ids") and context_depth > 0:
                related_ids = metadata["related_ids"]
                if isinstance(related_ids, str):
                    # Handle case where related_ids might be stored as a string
                    import json
                    try:
                        related_ids = json.loads(related_ids)
                    except:
                        related_ids = [related_ids]
                
                for related_id in related_ids:
                    related_chunks = self._get_chunk_by_id(related_id)
                    for chunk in related_chunks:
                        if chunk["id"] not in processed_ids:
                            context_chunks.append(chunk)
                            processed_ids.add(chunk["id"])
        
        return context_chunks
    
    def _get_chunk_by_id(self, chunk_id):
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve
            
        Returns:
            List containing the chunk if found, empty list otherwise
        """
        results = self.vector_store.collection.get(
            ids=[chunk_id],
            include=["metadatas", "documents", "embeddings"]
        )
        
        if not results or not results["ids"]:
            return []
        
        processed_results = []
        for i in range(len(results["ids"])):
            result = {
                "id": results["ids"][i],
                "metadata": results["metadatas"][i],
                "content": results["documents"][i] if "documents" in results else None,
                "is_context": True  # Mark as context chunk
            }
            processed_results.append(result)
        
        return processed_results
    
    def _get_child_chunks(self, parent_id):
        """
        Get child chunks for a parent chunk.
        
        Args:
            parent_id: The ID of the parent chunk
            
        Returns:
            List of child chunks
        """
        # Search for chunks with this parent ID
        results = self.vector_store.collection.get(
            where={"parent_id": {"$eq": parent_id}},
            include=["metadatas", "documents"]
        )
        
        if not results or not results["ids"]:
            return []
        
        processed_results = []
        for i in range(len(results["ids"])):
            result = {
                "id": results["ids"][i],
                "metadata": results["metadatas"][i],
                "content": results["documents"][i] if "documents" in results else None,
                "is_context": True  # Mark as context chunk
            }
            processed_results.append(result)
        
        return processed_results