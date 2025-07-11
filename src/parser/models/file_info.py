from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class FileInfo:
    """Information about a file in the repository."""
    path: str
    language: Optional[str] = None
    file_type: Optional[str] = None  # source, test, config, etc.
    imports: List[str] = field(default_factory=list)
    symbols: Dict[str, Dict] = field(default_factory=dict)  # Defined symbols (functions, classes)
    size_bytes: Optional[int] = None
    is_binary: bool = False