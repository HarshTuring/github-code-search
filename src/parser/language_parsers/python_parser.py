import ast
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..models.file_info import FileInfo
from ..models.chunk import Chunk
from .base_parser import BaseParser

class PythonParser(BaseParser):
    """Parser for Python files."""
    
    def get_language(self) -> str:
        return "python"
    
    def parse_file(self, file_path: Path) -> FileInfo:
        """Parse a Python file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_info = FileInfo(
                path=str(file_path),
                language="python",
                file_type="source",  # May be overridden by file type detector
                size_bytes=os.path.getsize(file_path),
                is_binary=False
            )
            
            try:
                # Parse the AST
                tree = ast.parse(content)
                
                # Extract imports
                file_info.imports = self._extract_imports(tree)
                
                # Extract defined symbols (classes, functions)
                file_info.symbols = self._extract_symbols(tree, content)
                
                return file_info
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return file_info
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # Return with limited information
            return FileInfo(path=str(file_path), language="python", is_binary=False)
    
    def create_chunks(self, file_info: FileInfo) -> List[Chunk]:
        """Create chunks from a parsed Python file."""
        chunks = []
        
        try:
            # Open the file to get content
            with open(file_info.path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a chunk for file level (imports, top-level code)
            file_level_chunk = self._create_file_level_chunk(file_info, content)
            if file_level_chunk:
                chunks.append(file_level_chunk)
            
            # Create chunks for each class and function
            for symbol_name, symbol_info in file_info.symbols.items():
                symbol_type = symbol_info.get('type')
                start_line = symbol_info.get('start_line')
                end_line = symbol_info.get('end_line')
                
                if symbol_type and start_line and end_line:
                    # Get the content for this symbol
                    symbol_content = self._extract_content_lines(content, start_line, end_line)
                    
                    # Create chunk
                    chunk_id = f"python_{uuid.uuid4().hex[:8]}_{symbol_type}_{symbol_name}"
                    chunk = Chunk(
                        id=chunk_id,
                        content=symbol_content,
                        metadata={
                            "file_path": file_info.path,
                            "language": "python",
                            "type": symbol_type,
                            "name": symbol_name,
                            "start_line": start_line,
                            "end_line": end_line,
                            "references": symbol_info.get('references', []),
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error creating chunks for {file_info.path}: {e}")
            return chunks
    
    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Extract import statements from Python AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    if module:
                        imports.append(f"{module}.{name.name}")
                    else:
                        imports.append(name.name)
                        
        return imports
    
    def _extract_symbols(self, tree: ast.Module, content: str) -> Dict[str, Dict]:
        """Extract classes and functions from Python AST."""
        symbols = {}
        
        for node in ast.iter_child_nodes(tree):
            # Handle classes
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, content)
                symbols[node.name] = class_info
                
            # Handle functions
            elif isinstance(node, ast.FunctionDef):
                func_info = self._extract_function_info(node, content)
                symbols[node.name] = func_info
                
        return symbols
    
    def _extract_class_info(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Extract information about a class definition."""
        # Get line numbers
        start_line = node.lineno
        end_line = self._find_end_line(node, content)
        
        # Extract methods
        methods = {}
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                method_info = self._extract_function_info(child, content)
                methods[child.name] = method_info
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        return {
            'type': 'class',
            'start_line': start_line,
            'end_line': end_line,
            'docstring': docstring,
            'bases': bases,
            'methods': methods
        }
    
    def _extract_function_info(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract information about a function definition."""
        # Get line numbers
        start_line = node.lineno
        end_line = self._find_end_line(node, content)
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            parameters.append(arg.arg)
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract references to other functions/classes
        references = self._extract_references(node)
        
        return {
            'type': 'function',
            'start_line': start_line,
            'end_line': end_line,
            'docstring': docstring,
            'parameters': parameters,
            'references': references
        }
    
    def _extract_references(self, node: ast.AST) -> List[str]:
        """Extract references to other symbols within a node."""
        references = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                references.append(child.id)
                
        return list(set(references))  # Remove duplicates
    
    def _find_end_line(self, node: ast.AST, content: str) -> int:
        """Find the end line number for an AST node."""
        try:
            # For Python 3.8+
            if hasattr(node, 'end_lineno'):
                return node.end_lineno
        except Exception:
            pass
            
        # Fallback: approximate using the next node or indentation
        lines = content.split('\n')
        start_line = node.lineno - 1  # Convert to 0-indexed
        
        # Try to find where this block ends based on indentation
        if start_line < len(lines):
            target_line = lines[start_line]
            indent = len(target_line) - len(target_line.lstrip())
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    return i  # Found a line with same/less indentation
                    
        # If we can't determine, just return a reasonable guess
        return min(start_line + 20, len(lines))
    
    def _extract_content_lines(self, content: str, start_line: int, end_line: int) -> str:
        """Extract specific line range from content."""
        lines = content.split('\n')
        # Convert from 1-indexed to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def _create_file_level_chunk(self, file_info: FileInfo, content: str) -> Optional[Chunk]:
        """Create a chunk for file-level code (imports, etc.)."""
        # Get all lines not in any function/class
        used_lines = set()
        
        for symbol_info in file_info.symbols.values():
            start = symbol_info.get('start_line', 0)
            end = symbol_info.get('end_line', 0)
            if start and end:
                for line_num in range(start, end + 1):
                    used_lines.add(line_num)
        
        # Get lines that are not part of any function/class
        content_lines = content.split('\n')
        file_level_lines = []
        
        for i, line in enumerate(content_lines):
            line_num = i + 1  # Convert to 1-indexed
            if line_num not in used_lines:
                file_level_lines.append(line)
        
        file_level_content = '\n'.join(file_level_lines)
        
        # Only create a chunk if there's meaningful content
        if file_level_content.strip():
            return Chunk(
                id=f"python_{uuid.uuid4().hex[:8]}_module_{Path(file_info.path).stem}",
                content=file_level_content,
                metadata={
                    "file_path": file_info.path,
                    "language": "python",
                    "type": "module",
                    "name": Path(file_info.path).stem,
                    "imports": file_info.imports,
                }
            )
        
        return None