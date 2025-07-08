import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from .models.repository_structure import RepositoryStructure
from .models.file_info import FileInfo
from .models.chunk import Chunk
from .file_type_detector import FileTypeDetector
from .language_parsers.python_parser import PythonParser
from .language_parsers.base_parser import BaseParser

class RepositoryParser:
    """Parses a repository and creates chunks of code."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {repo_path}")
            
        # Initialize repository structure
        self.repo_structure = RepositoryStructure(root_path=self.repo_path)
        
        # Initialize parsers
        self.parsers: Dict[str, BaseParser] = {
            "python": PythonParser(),
            # Add other language parsers here
        }
        
    def parse(self) -> RepositoryStructure:
        """Parse the repository structure."""
        # Traverse the repository
        self._discover_files()
        return self.repo_structure
        
    def create_chunks(self) -> List[Chunk]:
        """Create code chunks from the repository."""
        if not self.repo_structure.files:
            self.parse()
            
        chunks = []
        
        # Process each file based on its type
        for file_path, file_info in self.repo_structure.files.items():
            if file_info.is_binary:
                continue  # Skip binary files
                
            if not file_info.language:
                continue  # Skip files with unknown language
                
            # Get the appropriate parser
            parser = self.parsers.get(file_info.language)
            if parser:
                # Create chunks using the language parser
                file_chunks = parser.create_chunks(file_info)
                chunks.extend(file_chunks)
                
        return chunks
    
    def _discover_files(self) -> None:
        """Traverse the repository and discover files."""
        # Skip directories typically not needed
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', 'dist', 'build'}
        
        # Walk the repository
        for root, dirs, files in os.walk(self.repo_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            # Process each file
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.repo_path)
                
                # Detect file type
                language, file_type, is_binary = FileTypeDetector.detect(file_path)
                
                # Create file info
                file_info = FileInfo(
                    path=str(file_path),
                    language=language,
                    file_type=file_type,
                    is_binary=is_binary,
                    size_bytes=file_path.stat().st_size if file_path.exists() else 0
                )
                
                # If we have a parser for this language, parse the file
                if language in self.parsers and not is_binary:
                    parser = self.parsers[language]
                    file_info = parser.parse_file(file_path)
                    
                # Add to repository structure
                self.repo_structure.add_file(file_info)
                
            # Track directories
            for directory in dirs:
                dir_path = Path(root) / directory
                rel_dir = str(dir_path.relative_to(self.repo_path))
                self.repo_structure.directories.append(rel_dir)