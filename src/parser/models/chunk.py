from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class Chunk:
    """Represents a chunk of code with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def file_path(self) -> Optional[str]:
        return self.metadata.get("file_path")
    
    @property
    def language(self) -> Optional[str]:
        return self.metadata.get("language")
    
    @property
    def chunk_type(self) -> Optional[str]:
        return self.metadata.get("type")
    
    @property
    def name(self) -> Optional[str]:
        return self.metadata.get("name")
    
    @property
    def start_line(self) -> Optional[int]:
        return self.metadata.get("start_line")
    
    @property
    def end_line(self) -> Optional[int]:
        return self.metadata.get("end_line")
    
    @property
    def references(self) -> List[str]:
        return self.metadata.get("references", [])
    
    @property
    def related_chunks(self) -> List[str]:
        return self.metadata.get("related_chunks", [])