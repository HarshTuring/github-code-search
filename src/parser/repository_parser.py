import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from .models.repository_structure import RepositoryStructure
from .models.file_info import FileInfo
from .models.chunk import Chunk
from .file_type_detector import FileTypeDetector
from .language_parsers.python_parser import PythonParser
from .language_parsers.base_parser import BaseParser
from .language_parsers.javascript_parser import JavaScriptParser
from .language_parsers.infrastructure_parser import InfrastructureParser

from .tree_sitter_factory import TreeSitterFactory

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
            # Maintain existing parsers for backward compatibility
            "python": PythonParser(),
            "javascript": JavaScriptParser(),
            "typescript": JavaScriptParser(),
            "jsx": JavaScriptParser(),
            "yaml": InfrastructureParser(),
            "json": InfrastructureParser(),
            "dockerfile": InfrastructureParser(),
            "terraform": InfrastructureParser(),
            "shell": InfrastructureParser(),
        }
        self.infrastructure_parser = InfrastructureParser()

        # Use Tree-sitter parsers for newly supported languages
        self.use_tree_sitter = True  # Flag to enable/disable Tree-sitter
        from .tree_sitter_factory import TreeSitterFactory
        TreeSitterFactory.ensure_query_files_exist()

        
    def parse(self) -> RepositoryStructure:
        """Parse the repository structure."""
        # Traverse the repository
        self._discover_files()
        
        # Second pass: Check for infrastructure files specifically
        infrastructure_files = []
        for file_path, file_info in self.repo_structure.files.items():
            # Skip binary files
            if file_info.is_binary:
                continue
                
            path_obj = Path(file_path)
            
            # Check if this might be an infrastructure file
            config_type = self.infrastructure_parser._detect_config_type(path_obj)
            if config_type:
                # This is an infrastructure file, re-parse it with the infrastructure parser
                updated_file_info = self.infrastructure_parser.parse_file(path_obj)
                # Update the file info
                self.repo_structure.files[file_path] = updated_file_info
        
        return self.repo_structure

    def create_chunks(self, verbose=False) -> List[Chunk]:
        """Create code chunks from the repository."""
        if not self.repo_structure.files:
            self.parse()
            
        chunks = []
        
        # Debug stats
        total_files = len(self.repo_structure.files)
        processed_files = 0
        binary_files = 0
        unsupported_files = 0
        
        # Process each file based on its type
        for file_path, file_info in self.repo_structure.files.items():
            if verbose:
                print(f"Processing file: {file_path}, Language: {file_info.language}, Binary: {file_info.is_binary}")
                
            if file_info.is_binary:
                binary_files += 1
                if verbose:
                    print(f"  Skipping binary file: {file_path}")
                continue
                
            if not file_info.language:
                unsupported_files += 1
                if verbose:
                    print(f"  Skipping unsupported file: {file_path}")
                continue
                
            # Try to get a parser
            parser = None
            
            # First check if we have a custom parser
            parser = self.parsers.get(file_info.language)
            
            # If no custom parser and Tree-sitter is enabled, try Tree-sitter
            if not parser and self.use_tree_sitter:
                parser = TreeSitterFactory.get_parser(file_info.language)
                
            if parser:
                # Create chunks using the parser
                try:
                    file_chunks = parser.create_chunks(file_info)
                    if file_chunks:
                        chunks.extend(file_chunks)
                        processed_files += 1
                        if verbose:
                            print(f"  Created {len(file_chunks)} chunks for {file_path}")
                    else:
                        if verbose:
                            print(f"  No chunks created for {file_path}")
                except Exception as e:
                    print(f"Error creating chunks for {file_path}: {e}")
            else:
                unsupported_files += 1
                if verbose:
                    print(f"  No parser available for language: {file_info.language}")
        
        # Print summary statistics
        print(f"\nRepository Processing Summary:")
        print(f"  Total files: {total_files}")
        print(f"  Processed files: {processed_files}")
        print(f"  Binary files skipped: {binary_files}")
        print(f"  Unsupported files skipped: {unsupported_files}")
        print(f"  Total chunks created: {len(chunks)}")
        
        # Print language-specific stats if verbose
        if verbose:
            lang_stats = {}
            for chunk in chunks:
                lang = chunk.language
                if lang not in lang_stats:
                    lang_stats[lang] = 0
                lang_stats[lang] += 1
            
            print("\nChunks by language:")
            for lang, count in lang_stats.items():
                print(f"  {lang}: {count} chunks")
        
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
                
                # If we have a custom parser for this language, use it
                if language in self.parsers and not is_binary:
                    parser = self.parsers[language]
                    file_info = parser.parse_file(file_path)
                # Otherwise, if Tree-sitter supports this language, use it
                elif self.use_tree_sitter and TreeSitterFactory.supports_language(language) and not is_binary:
                    ts_parser = TreeSitterFactory.get_parser(language)
                    if ts_parser:
                        file_info = ts_parser.parse_file(file_path)
                    
                # Add to repository structure
                self.repo_structure.add_file(file_info)
                
            # Track directories
            for directory in dirs:
                dir_path = Path(root) / directory
                rel_dir = str(dir_path.relative_to(self.repo_path))
                self.repo_structure.directories.append(rel_dir)