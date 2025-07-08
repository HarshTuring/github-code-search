import os
import re
from typing import Tuple, Optional
from pathlib import Path

# Try to import magic, but provide fallback if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available. File type detection will be limited.")

class FileTypeDetector:
    """Detects file types and languages based on content and extension."""
    
    # Maps file extensions to languages
    EXTENSION_MAP = {
        # Python
        '.py': 'python',
        '.pyw': 'python',
        '.pyx': 'python',
        # JavaScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        # Java
        '.java': 'java',
        # Go
        '.go': 'go',
        # Ruby
        '.rb': 'ruby',
        '.rake': 'ruby',
        # C/C++
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        # Other
        '.md': 'markdown',
        '.rst': 'restructuredtext',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sql': 'sql',
    }
    
    # Binary file extensions
    BINARY_EXTENSIONS = {
        '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.exe', '.dll', '.so', '.dylib', '.class', '.jar', '.war',
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
        '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wmv',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.bin', '.dat', '.db', '.sqlite', '.pyc', '.pyo'
    }
    
    # File types categorization
    FILE_TYPE_PATTERNS = {
        'test': [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'.*\.spec\.js$',
            r'.*\.test\.js$',
            r'.*Test\.java$'
        ],
        'config': [
            r'.*\.config\.[^.]+$',
            r'.*\.conf$',
            r'.*\.ini$',
            r'config\..*$',
            r'settings\..*$',
            r'.*rc$'
        ],
        'documentation': [
            r'README\.[^.]+$',
            r'CONTRIBUTING\.[^.]+$',
            r'CHANGELOG\.[^.]+$',
            r'.*\.md$',
            r'.*\.rst$',
            r'docs/.*$'
        ]
    }
    
    @classmethod
    def detect(cls, file_path: Path) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Detect the language, file type, and whether a file is binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (language, file_type, is_binary)
        """
        try:
            # Check if file is binary
            is_binary = cls._is_binary(file_path)
            
            # Get file extension and detect language
            extension = file_path.suffix.lower()
            language = cls.EXTENSION_MAP.get(extension)
            
            # Detect file type
            file_type = cls._detect_file_type(file_path)
            
            # For source code without a specific type
            if language and not file_type:
                file_type = 'source'
                
            return language, file_type, is_binary
            
        except Exception as e:
            print(f"Error detecting file type for {file_path}: {e}")
            # Make a best guess based on extension
            extension = file_path.suffix.lower()
            is_binary = extension in cls.BINARY_EXTENSIONS
            return cls.EXTENSION_MAP.get(extension), None, is_binary
    
    @classmethod
    def _is_binary(cls, file_path: Path) -> bool:
        """Check if a file is binary, with special handling for JS/TS files."""
        # JavaScript/TypeScript files are always text files
        js_extensions = {'.js', '.jsx', '.ts', '.tsx', '.json'}
        if file_path.suffix.lower() in js_extensions:
            return False
            
        # Quick check based on extension for known binary types
        if file_path.suffix.lower() in cls.BINARY_EXTENSIONS:
            return True
                
        # Use python-magic if available
        if MAGIC_AVAILABLE:
            try:
                mime = magic.from_file(str(file_path), mime=True)
                # Consider all text/* MIME types as non-binary
                if mime.startswith('text/'):
                    return False
                # Special case for JavaScript and JSON
                if mime in ('application/javascript', 'application/json'):
                    return False
                # If we get here with a MIME type, it's likely binary
                if mime != 'application/octet-stream':  # Sometimes magic returns this for text
                    return True
            except Exception as e:
                print(f"Warning: magic library failed for {file_path}: {e}")
                # Fall through to fallback
                    
        # Fallback method for text detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read a chunk as text
                chunk = f.read(1024)
                return False  # If we got here, it's readable as text
        except UnicodeDecodeError:
            # If we can't decode as UTF-8, it's likely binary
            return True
                
        # Final fallback: check for null bytes
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Check for null bytes which likely indicate binary
                return b'\0' in chunk
        except Exception:
            # If all else fails, assume it's not binary
            return False
    
    @classmethod
    def _detect_file_type(cls, file_path: Path) -> Optional[str]:
        """Detect the type of file based on patterns."""
        file_str = str(file_path)
        
        # Special files
        filename = file_path.name.lower()
        if filename in {'license', 'license.txt', 'license.md'}:
            return 'license'
        elif filename in {'dockerfile'}:
            return 'docker'
        elif filename in {'.gitignore', '.gitattributes'}:
            return 'git'
            
        # Check patterns
        for file_type, patterns in cls.FILE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, file_str, re.IGNORECASE):
                    return file_type
                    
        return None