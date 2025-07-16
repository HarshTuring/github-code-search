import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import tree_sitter
import tree_sitter_languages as tsl

from .base_parser import BaseParser
from ..models.file_info import FileInfo
from ..models.chunk import Chunk

class TreeSitterParser(BaseParser):
    """Base class for language-agnostic code parsing using Tree-sitter."""
    
    # Class variables for sharing parser and language instances
    _parsers = {}
    _languages = {}
    _queries = {}
    
    def __init__(self, language_name):
        """
        Initialize a Tree-sitter parser for the specified language.
        
        Args:
            language_name: Name of the language to parse
        """
        self.language_name = language_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize parser for this language
        self._initialize_parser()
    
    def get_language(self) -> str:
        """Get the language this parser handles."""
        return self.language_name
    
    def _initialize_parser(self) -> bool:
        """Initialize Tree-sitter parser for the language."""
        if self.language_name in self._parsers:
            return True
            
        try:
            # Get Tree-sitter language
            ts_language = self._get_ts_language()
            
            # Create parser
            parser = tree_sitter.Parser()
            parser.set_language(ts_language)
            
            # Store parser in class variable
            self._parsers[self.language_name] = parser
            
            # Load queries for this language
            self._load_queries()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Tree-sitter parser for {self.language_name}: {str(e)}")
            return False
    
    def _get_ts_language(self) -> tree_sitter.Language:
        """Get Tree-sitter language for the specified language name."""
        if self.language_name in self._languages:
            return self._languages[self.language_name]
            
        # Map our language names to Tree-sitter language names
        ts_lang_map = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'jsx': 'tsx',
            'tsx': 'tsx',
            'go': 'go',
            'java': 'java',
            'ruby': 'ruby',
            'rust': 'rust',
            'c': 'c',
            'cpp': 'cpp',
            'c_sharp': 'c_sharp',
            'php': 'php'
        }
        
        ts_lang_name = ts_lang_map.get(self.language_name, self.language_name)
        
        try:
            language = tsl.get_language(ts_lang_name)
            self._languages[self.language_name] = language
            return language
        except Exception as e:
            self.logger.error(f"Failed to load Tree-sitter language for {self.language_name}: {str(e)}")
            raise ValueError(f"Unsupported language: {self.language_name}")
    
    def _load_queries(self) -> None:
        """Load Tree-sitter queries for the language."""
        if self.language_name in self._queries:
            return
            
        # Path to query file
        query_path = Path(__file__).parent.parent / "language_queries" / f"{self.language_name}_queries.scm"
        
        # Create empty queries dict for this language
        self._queries[self.language_name] = None
        
        # Check if query file exists
        if not query_path.exists():
            self.logger.warning(f"No query file found for {self.language_name} at {query_path}")
            return
            
        try:
            # Load query file
            with open(query_path, 'r', encoding='utf-8') as f:
                query_content = f.read()
                
            # Compile query
            language = self._get_ts_language()
            self._queries[self.language_name] = language.query(query_content)
            
        except Exception as e:
            self.logger.error(f"Failed to load queries for {self.language_name}: {str(e)}")
    
    def parse_file(self, file_path: Path) -> FileInfo:
        """Parse a file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            file_info = FileInfo(
                path=str(file_path),
                language=self.language_name,
                file_type="source",
                size_bytes=os.path.getsize(file_path),
                is_binary=False
            )
            
            # If parser initialization failed, return basic file info
            if self.language_name not in self._parsers:
                return file_info
                
            # Parse file with Tree-sitter
            parser = self._parsers[self.language_name]
            tree = parser.parse(bytes(content, 'utf-8'))
            
            # Extract imports
            imports = self._extract_imports(tree.root_node, content)
            file_info.imports = imports
            
            # Extract symbols (classes, functions, etc.)
            symbols = self._extract_symbols(tree.root_node, content, file_path)
            file_info.symbols = symbols
            
            return file_info
                
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            # Return with limited information
            return FileInfo(
                path=str(file_path), 
                language=self.language_name, 
                is_binary=False
            )
    
    def create_chunks(self, file_info: FileInfo) -> List[Chunk]:
        """Create chunks from a parsed file."""
        chunks = []
        
        try:
            # Open the file to get content
            with open(file_info.path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # If parser initialization failed, return empty chunks
            if self.language_name not in self._parsers or self.language_name not in self._queries:
                return []
                
            # Parse file with Tree-sitter
            parser = self._parsers[self.language_name]
            tree = parser.parse(bytes(content, 'utf-8'))
            
            # Get queries for this language
            query = self._queries.get(self.language_name)
            if not query:
                return []
                
            # Execute query to find code components
            captures = query.captures(tree.root_node)
            
            # Group captures by component and type
            components = self._group_captures(captures, content)
            
            # Create a chunk for each component
            for component_type, components_list in components.items():
                for component_id, component in components_list.items():
                    chunk = self._create_chunk_from_component(
                        component_type,
                        component,
                        file_info.path,
                        content
                    )
                    if chunk:
                        chunks.append(chunk)
            
            # Create file-level chunk for imports and top-level code if needed
            file_level_chunk = self._create_file_level_chunk(file_info, content, components)
            if file_level_chunk:
                chunks.append(file_level_chunk)
                
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating chunks for {file_info.path}: {str(e)}")
            return []
    
    def _extract_imports(self, root_node, content: str) -> List[str]:
        """Extract import statements using Tree-sitter queries."""
        imports = []
        
        # Check if we have queries
        if self.language_name not in self._queries or not self._queries[self.language_name]:
            return imports
            
        # Find import statements in the query results
        query = self._queries[self.language_name]
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if 'import' in capture_name:
                # Extract the import text
                import_text = content[node.start_byte:node.end_byte]
                # Further processing depends on the language
                if self.language_name == 'python':
                    # For Python, extract module names from import statements
                    if 'import ' in import_text:
                        modules = import_text.split('import ')[1].strip()
                        for module in modules.split(','):
                            imports.append(module.strip())
                elif self.language_name in ['javascript', 'typescript', 'jsx', 'tsx']:
                    # For JS/TS, extract from import statements
                    if 'from ' in import_text:
                        module = import_text.split('from ')[1].strip().strip('\'";')
                        imports.append(module)
                else:
                    # Generic handling for other languages
                    imports.append(import_text)
        
        return imports
    
    def _extract_symbols(self, root_node, content: str, file_path: Path) -> Dict[str, Dict]:
        """Extract symbols (functions, classes) using Tree-sitter queries."""
        symbols = {}
        
        # Check if we have queries
        if self.language_name not in self._queries or not self._queries[self.language_name]:
            return symbols
            
        # Find symbols in the query results
        query = self._queries[self.language_name]
        captures = query.captures(root_node)
        
        # Group captures by component and type
        components = self._group_captures(captures, content)
        
        # Create a symbol entry for each component
        for component_type, components_list in components.items():
            for component_id, component in components_list.items():
                # Skip if component doesn't have a name
                if 'name' not in component['parts']:
                    continue
                    
                name = component['parts']['name']['text']
                
                # Create symbol information
                symbol_info = {
                    'type': component_type,
                    'start_line': component['node'].start_point[0] + 1,  # Convert to 1-indexed
                    'end_line': component['node'].end_point[0] + 1,      # Convert to 1-indexed
                }
                
                # Add extra information based on component type
                if component_type == 'function' or component_type == 'method':
                    if 'parameters' in component['parts']:
                        params_text = component['parts']['parameters']['text']
                        symbol_info['parameters'] = params_text
                        
                elif component_type == 'class':
                    if 'body' in component['parts']:
                        # Could extract methods here if needed
                        pass
                
                symbols[name] = symbol_info
        
        return symbols
    
    def _group_captures(self, captures, content: str) -> Dict[str, Dict]:
        """Group query captures by component type and ID."""
        components = {}
        
        for node, capture_name in captures:
            # Skip if capture name doesn't follow our pattern
            if '.' not in capture_name:
                continue
                
            component_type, part_type = capture_name.split('.', 1)
            
            # Initialize component type dictionary if needed
            if component_type not in components:
                components[component_type] = {}
                
            # Find or create component by ID (using parent node ID)
            node_id = id(node.parent)
            if node_id not in components[component_type]:
                components[component_type][node_id] = {
                    'node': node.parent,
                    'parts': {},
                    'full_text': content[node.parent.start_byte:node.parent.end_byte]
                }
                
            # Store this part
            components[component_type][node_id]['parts'][part_type] = {
                'text': content[node.start_byte:node.end_byte],
                'node': node,
                'range': (node.start_byte, node.end_byte)
            }
        
        return components
    
    def _create_chunk_from_component(self, 
                                    component_type: str,
                                    component: Dict,
                                    file_path: str,
                                    content: str) -> Optional[Chunk]:
        """Create a Chunk object from a code component."""
        try:
            # Extract component parts
            parts = component['parts']
            node = component['node']
            full_text = component['full_text']
            
            # Skip if it doesn't have a name or body
            if 'name' not in parts:
                return None
                
            name = parts['name']['text']
            
            # Generate a unique ID based on language, type and name
            import uuid
            chunk_id = f"{self.language_name}_{uuid.uuid4().hex[:8]}_{component_type}_{name}"
            
            # Create metadata
            metadata = {
                'file_path': file_path,
                'language': self.language_name,
                'type': component_type,
                'name': name,
                'start_line': node.start_point[0] + 1,  # Convert to 1-indexed
                'end_line': node.end_point[0] + 1,      # Convert to 1-indexed
            }
            
            # Add specific metadata based on component type
            if component_type in ['function', 'method']:
                if 'parameters' in parts:
                    metadata['parameters'] = parts['parameters']['text']
                    
            elif component_type == 'class':
                if 'body' in parts:
                    # Add class-specific metadata
                    pass
                    
            # Create chunk
            chunk = Chunk(
                id=chunk_id,
                content=full_text,
                metadata=metadata
            )
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error creating chunk from component: {str(e)}")
            return None
    
    def _create_file_level_chunk(self, file_info: FileInfo, content: str, components: Dict) -> Optional[Chunk]:
        """Create a chunk for file-level code (imports, etc.)."""
        # This is a simple implementation that just includes imports
        # A more sophisticated version would exclude code already in other chunks
        
        import uuid
        from pathlib import Path
        
        # Extract just the imports section
        imports_section = ""
        lines = content.split('\n')
        
        for line in lines:
            if ('import ' in line or 'from ' in line) and (
                    self.language_name == 'python' or 
                    self.language_name in ['javascript', 'typescript', 'jsx', 'tsx']):
                imports_section += line + '\n'
                
        # Only create a chunk if we found imports
        if imports_section.strip():
            chunk_id = f"{self.language_name}_{uuid.uuid4().hex[:8]}_imports_{Path(file_info.path).stem}"
            
            return Chunk(
                id=chunk_id,
                content=imports_section,
                metadata={
                    'file_path': file_info.path,
                    'language': self.language_name,
                    'type': 'imports',
                    'name': 'Imports',
                    'imports': file_info.imports
                }
            )
            
        return None