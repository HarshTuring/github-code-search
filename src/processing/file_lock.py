import os
import time
import errno
import logging
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class FileLockException(Exception):
    """Exception raised when file locking fails."""
    pass

class FileLock:
    """
    A file locking mechanism that works across threads and processes.
    """
    
    def __init__(self, file_path, timeout=10, delay=0.05):
        """
        Initialize a file lock.
        
        Args:
            file_path: Path to the file to lock
            timeout: Timeout in seconds (default: 10)
            delay: Delay between lock acquisition attempts (default: 0.05)
        """
        self.file_path = Path(file_path).with_suffix('.lock')
        self.timeout = timeout
        self.delay = delay
        self.is_locked = False
    
    def acquire(self):
        """
        Acquire the lock with timeout.
        
        Raises:
            FileLockException: If the lock cannot be acquired within the timeout.
        """
        start_time = time.time()
        
        while True:
            try:
                # Try to create the lock file exclusively
                self.fd = os.open(self.file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # Mark as locked
                self.is_locked = True
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    # Something unexpected happened
                    raise FileLockException(f"Failed to acquire lock: {e}")
                
                # Check if we've hit the timeout
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException(f"Timeout waiting to acquire lock on {self.file_path}")
                
                # Wait and try again
                time.sleep(self.delay)
    
    def release(self):
        """Release the lock."""
        if self.is_locked:
            try:
                os.close(self.fd)
                os.unlink(self.file_path)
                self.is_locked = False
            except OSError as e:
                logger.error(f"Error releasing lock on {self.file_path}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Ensure lock is released when object is garbage collected."""
        self.release()

@contextmanager
def file_lock(file_path, timeout=10, delay=0.05):
    """
    Context manager for file locking.
    
    Args:
        file_path: Path to the file to lock
        timeout: Timeout in seconds (default: 10)
        delay: Delay between lock acquisition attempts (default: 0.05)
        
    Yields:
        FileLock: The file lock object
        
    Raises:
        FileLockException: If the lock cannot be acquired within the timeout.
    """
    lock = FileLock(file_path, timeout, delay)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()