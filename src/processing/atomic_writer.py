import os
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class AtomicWriter:
    """
    Utility for atomic file operations to prevent corruption during parallel processing.
    """
    
    @staticmethod
    def write(file_path, content, binary=False):
        """
        Write content to a file atomically.
        
        Args:
            file_path: Path to the file to write
            content: Content to write (string or bytes)
            binary: Whether the content is binary
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file in the same directory
        try:
            fd, temp_path = tempfile.mkstemp(dir=file_path.parent)
            success = False
            
            try:
                # Write content to temporary file
                if binary:
                    with os.fdopen(fd, 'wb') as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                else:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                
                # Atomic rename
                os.rename(temp_path, file_path)
                success = True
                return True
                
            finally:
                # Clean up if something went wrong
                if not success and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to clean up temporary file: {e}")
        
        except Exception as e:
            logger.error(f"Atomic write failed for {file_path}: {e}")
            return False
    
    @staticmethod
    def write_json(file_path, data, indent=2):
        """
        Write JSON data to a file atomically.
        
        Args:
            file_path: Path to the file to write
            data: Data to write (will be serialized to JSON)
            indent: JSON indentation (default: 2)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            json_content = json.dumps(data, indent=indent)
            return AtomicWriter.write(file_path, json_content)
        except Exception as e:
            logger.error(f"Failed to write JSON data: {e}")
            return False
    
    @staticmethod
    def read_json(file_path, default=None):
        """
        Read JSON data from a file with error handling.
        
        Args:
            file_path: Path to the file to read
            default: Default value to return if file doesn't exist or is invalid
            
        Returns:
            The parsed JSON data or the default value
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to read JSON from {file_path}: {e}")
            return default