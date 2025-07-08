import os
import magic
import re
from typing import Tuple, Optional
from pathlib import Path

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
        # Check if file is binary
        try:
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
            return None, None, False
    
    @staticmethod
    def _is_binary(file_path: Path) -> bool:
        """Check if a file is binary."""
        # Use python-magic if available
        try:
            mime = magic.from_file(str(file_path), mime=True)
            return not mime.startswith('text/')
        except (ImportError, AttributeError):
            # Fallback method if python-magic is not available
            try:
                with open(file_path, 'rb') as f:
                    chunk = f.read(1024)
                    return b'\0' in chunk
            except Exception:
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