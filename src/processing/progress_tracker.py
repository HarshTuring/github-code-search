import json
import logging
import os
import time
from pathlib import Path

from .atomic_writer import AtomicWriter
from .file_lock import file_lock

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Thread-safe progress tracker for repository processing.
    """
    
    def __init__(self, repo_path, name="repository_processing"):
        """
        Initialize the progress tracker.
        
        Args:
            repo_path: Path to the repository being processed
            name: Name for the progress file
        """
        self.repo_path = Path(repo_path)
        self.progress_file = self.repo_path / f".{name}_progress.json"
        self.progress_data = self._load_progress()
    
    def _load_progress(self):
        """
        Load existing progress data with locking.
        
        Returns:
            dict: The progress data or an empty dictionary
        """
        try:
            with file_lock(self.progress_file):
                return AtomicWriter.read_json(self.progress_file, default={
                    "started_at": time.time(),
                    "last_updated": time.time(),
                    "completed_files": {},
                    "stats": {
                        "total_files": 0,
                        "processed_files": 0,
                        "failed_files": 0,
                        "skipped_files": 0
                    }
                })
        except Exception as e:
            logger.error(f"Failed to load progress data: {e}")
            # Return a new progress object
            return {
                "started_at": time.time(),
                "last_updated": time.time(),
                "completed_files": {},
                "stats": {
                    "total_files": 0,
                    "processed_files": 0,
                    "failed_files": 0,
                    "skipped_files": 0
                }
            }
    
    def save_progress(self):
        """Save progress data atomically with file locking."""
        try:
            # Update timestamp
            self.progress_data["last_updated"] = time.time()
            
            # Calculate the progress percentage
            stats = self.progress_data["stats"]
            total = stats["total_files"]
            processed = stats["processed_files"]
            failed = stats["failed_files"]
            skipped = stats["skipped_files"]
            
            if total > 0:
                percentage = (processed + failed + skipped) / total * 100
                self.progress_data["completion_percentage"] = percentage
            
            # Save with file locking
            with file_lock(self.progress_file):
                AtomicWriter.write_json(self.progress_file, self.progress_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            return False
    
    def mark_file_completed(self, file_path, status="success", metadata=None):
        """
        Mark a file as processed.
        
        Args:
            file_path: Path to the file that was processed
            status: Status of processing ("success", "failure", "skipped")
            metadata: Optional metadata about the processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert file_path to a string relative to repo_path
            if isinstance(file_path, Path):
                file_path = str(file_path.relative_to(self.repo_path))
            elif file_path.startswith(str(self.repo_path)):
                file_path = file_path[len(str(self.repo_path)):].lstrip('/')
            
            # Update progress data
            self.progress_data["completed_files"][file_path] = {
                "status": status,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Update statistics
            stats = self.progress_data["stats"]
            if status == "success":
                stats["processed_files"] += 1
            elif status == "failure":
                stats["failed_files"] += 1
            elif status == "skipped":
                stats["skipped_files"] += 1
                
            # Save progress
            return self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to mark file completed: {e}")
            return False
    
    def mark_files_as_found(self, file_count):
        """Update the total file count."""
        self.progress_data["stats"]["total_files"] = file_count
        return self.save_progress()
    
    def is_file_completed(self, file_path):
        """Check if a file has already been processed."""
        # Convert file_path to a string relative to repo_path
        if isinstance(file_path, Path):
            file_path = str(file_path.relative_to(self.repo_path))
        elif file_path.startswith(str(self.repo_path)):
            file_path = file_path[len(str(self.repo_path)):].lstrip('/')
            
        return file_path in self.progress_data["completed_files"]
    
    def get_completed_files(self):
        """Get a list of completed files."""
        return list(self.progress_data["completed_files"].keys())
    
    def get_completion_status(self):
        """Get the overall completion status."""
        stats = self.progress_data["stats"]
        total = stats["total_files"]
        processed = stats["processed_files"]
        failed = stats["failed_files"]
        skipped = stats["skipped_files"]
        
        if total == 0:
            percentage = 0
        else:
            percentage = (processed + failed + skipped) / total * 100
            
        return {
            "total_files": total,
            "processed_files": processed,
            "failed_files": failed,
            "skipped_files": skipped,
            "completion_percentage": percentage,
            "started_at": self.progress_data["started_at"],
            "last_updated": self.progress_data["last_updated"],
            "elapsed_seconds": time.time() - self.progress_data["started_at"]
        }