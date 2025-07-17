import os
import logging
import time
from pathlib import Path
from concurrent.futures import as_completed
import gc

from .resource_monitor import ResourceAwareThreadPool
from .file_lock import file_lock
from .progress_tracker import ProgressTracker
from ..parser.file_type_detector import FileTypeDetector
from ..parser.tree_sitter_factory import TreeSitterFactory
from ..parser.models.chunk import Chunk

logger = logging.getLogger(__name__)

class ParallelRepositoryProcessor:
    """
    Process a repository in parallel with thread safety and resource awareness.
    """
    
    def __init__(self, 
                 repo_path,
                 min_workers=2,
                 max_workers=None,
                 chunk_batch_size=50,
                 memory_threshold=80,
                 cpu_threshold=90):
        """
        Initialize the parallel repository processor.
        
        Args:
            repo_path: Path to the repository to process
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads (default: CPU count)
            chunk_batch_size: Number of chunks to process in a batch
            memory_threshold: Memory usage percentage threshold to reduce workers
            cpu_threshold: CPU usage percentage threshold to reduce workers
        """
        self.repo_path = Path(repo_path)
        self.chunk_batch_size = chunk_batch_size
        
        # Initialize the thread pool
        self.thread_pool = ResourceAwareThreadPool(
            min_workers=min_workers,
            max_workers=max_workers,
            memory_threshold=memory_threshold,
            cpu_threshold=cpu_threshold
        )
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(repo_path)
        
        # Cache for tree-sitter parsers
        self._parser_cache = {}
    
    def get_parser(self, language):
        """Get a Tree-sitter parser for the specified language with caching."""
        if language not in self._parser_cache:
            parser = TreeSitterFactory.get_parser(language)
            if parser:
                self._parser_cache[language] = parser
        return self._parser_cache.get(language)
    
    def process_file(self, file_path):
        """
        Process a single file with file locking for safety.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            list: List of chunks created from the file
        """
        file_path = Path(file_path)
        rel_path = file_path.relative_to(self.repo_path)
        
        # Check if file has already been processed
        if self.progress_tracker.is_file_completed(rel_path):
            logger.debug(f"Skipping already processed file: {rel_path}")
            return []
        
        # Use file lock to prevent concurrent access
        lock_path = file_path.with_suffix(file_path.suffix + '.lock')
        
        try:
            # Try to acquire lock
            with file_lock(lock_path, timeout=30):
                # Detect language and check if binary
                language, file_type, is_binary = FileTypeDetector.detect(file_path)
                
                # Skip binary files
                if is_binary:
                    self.progress_tracker.mark_file_completed(
                        rel_path, 
                        status="skipped",
                        metadata={"reason": "binary_file"}
                    )
                    return []
                
                # Skip unsupported languages
                if not language or not self.get_parser(language):
                    self.progress_tracker.mark_file_completed(
                        rel_path, 
                        status="skipped",
                        metadata={"reason": "unsupported_language", "language": language}
                    )
                    return []
                
                # Parse the file
                try:
                    parser = self.get_parser(language)
                    file_info = parser.parse_file(file_path)
                    chunks = parser.create_chunks(file_info)
                    
                    # Mark file as completed
                    self.progress_tracker.mark_file_completed(
                        rel_path, 
                        status="success",
                        metadata={
                            "language": language, 
                            "chunk_count": len(chunks)
                        }
                    )
                    
                    return chunks
                except Exception as e:
                    logger.error(f"Error processing file {rel_path}: {e}")
                    self.progress_tracker.mark_file_completed(
                        rel_path, 
                        status="failure",
                        metadata={"error": str(e)}
                    )
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to acquire lock for {rel_path}: {e}")
            return []
    
    def discover_files(self):
        """
        Discover all files in the repository.
        
        Returns:
            list: List of file paths
        """
        # Skip directories typically not needed
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', 'dist', 'build'}
        
        files = []
        
        # Walk the repository
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            # Process each file
            for filename in filenames:
                file_path = Path(root) / filename
                files.append(file_path)
        
        # Update progress tracker with file count
        self.progress_tracker.mark_files_as_found(len(files))
        
        return files
    
    def process_repository(self, on_progress=None):
        """
        Process the entire repository in parallel.
        
        Args:
            on_progress: Optional callback function for progress updates
            
        Returns:
            list: List of all chunks created from the repository
        """
        logger.info(f"Starting parallel processing of repository: {self.repo_path}")
        start_time = time.time()
        
        # Discover files
        files = self.discover_files()
        logger.info(f"Found {len(files)} files in the repository")
        
        # Filter out files that have already been processed
        completed_files = set(self.progress_tracker.get_completed_files())
        remaining_files = [
            f for f in files 
            if str(f.relative_to(self.repo_path)) not in completed_files
        ]
        
        logger.info(f"{len(remaining_files)} files need processing ({len(completed_files)} already processed)")
        
        # If all files are already processed, return
        if not remaining_files:
            logger.info("All files already processed, nothing to do")
            return []
        
        # Process files in batches to control memory usage
        all_chunks = []
        batch_size = 100  # Process 100 files at a time
        
        for i in range(0, len(remaining_files), batch_size):
            batch = remaining_files[i:i+batch_size]
            logger.info(f"Processing batch of {len(batch)} files ({i+1}-{min(i+batch_size, len(remaining_files))} of {len(remaining_files)})")
            
            # Submit files for processing
            futures = {
                self.thread_pool.submit(self.process_file, file_path): file_path
                for file_path in batch
            }
            
            # Process results as they complete
            batch_chunks = []
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_chunks = future.result()
                    batch_chunks.extend(file_chunks)
                    
                    # Report progress
                    status = self.progress_tracker.get_completion_status()
                    if on_progress:
                        on_progress(status)
                    
                    # Log progress
                    if len(file_chunks) > 0:
                        logger.debug(f"Processed {file_path}: {len(file_chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
            
            # Add batch chunks to all chunks
            all_chunks.extend(batch_chunks)
            
            # Log batch completion
            logger.info(f"Completed batch with {len(batch_chunks)} chunks. Total: {len(all_chunks)}")
            
            # Perform garbage collection between batches
            gc.collect()
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Repository processing completed in {elapsed_time:.2f} seconds. "
                   f"Generated {len(all_chunks)} chunks from {len(files)} files.")
        
        return all_chunks
    
    def shutdown(self):
        """Clean up resources."""
        self.thread_pool.shutdown()
        logger.info("Parallel processor shut down")