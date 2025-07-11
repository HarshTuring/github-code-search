from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..models.file_info import FileInfo
from ..models.chunk import Chunk

class BaseParser(ABC):
    """Abstract base class for language-specific parsers."""
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> FileInfo:
        """Parse a file and extract information."""
        pass
    
    @abstractmethod
    def create_chunks(self, file_info: FileInfo) -> List[Chunk]:
        """Create chunks from a parsed file."""
        pass
    
    def get_language(self) -> str:
        """Get the language this parser handles."""
        pass