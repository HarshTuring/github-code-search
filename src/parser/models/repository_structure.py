from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from pathlib import Path
from .file_info import FileInfo

@dataclass
class RepositoryStructure:
    """Represents the structure of a repository."""
    root_path: Path
    files: Dict[str, FileInfo] = field(default_factory=dict)
    directories: List[str] = field(default_factory=list)
    
    def add_file(self, file_info: FileInfo):
        """Add a file to the repository structure."""
        self.files[file_info.path] = file_info
    
    def get_files_by_language(self, language: str) -> List[FileInfo]:
        """Get all files of a specific language."""
        return [f for f in self.files.values() if f.language == language]
    
    def get_files_by_type(self, file_type: str) -> List[FileInfo]:
        """Get all files of a specific type."""
        return [f for f in self.files.values() if f.file_type == file_type]
    
    @property
    def file_count(self) -> int:
        return len(self.files)