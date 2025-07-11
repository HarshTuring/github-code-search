import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from ..models.file_info import FileInfo
from ..models.chunk import Chunk
from .base_parser import BaseParser

class JavaScriptParser(BaseParser):
    """Parser for JavaScript/TypeScript files, with special handling for React components."""
    
    # Regex patterns for React component detection
    FUNC_COMPONENT_PATTERNS = [
        # Arrow function components
        r'(?:export\s+)?(?:const|let|var)\s+([A-Z][a-zA-Z0-9_]*)\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>\s*(?:{|\()',
        # Function declaration components
        r'(?:export\s+)?function\s+([A-Z][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{',
        # Export default for components
        r'export\s+default\s+(?:function)?\s*([A-Z][a-zA-Z0-9_]*)'
    ]
    
    CLASS_COMPONENT_PATTERNS = [
        # Class components
        r'(?:export\s+)?class\s+([A-Z][a-zA-Z0-9_]*)\s+extends\s+(?:React\.)?Component',
        r'(?:export\s+)?class\s+([A-Z][a-zA-Z0-9_]*)\s+extends\s+(?:React\.)?PureComponent'
    ]
    
    HOOK_PATTERNS = [
        # React hooks
        r'const\s+\[([a-zA-Z0-9_]+),\s*set([A-Z][a-zA-Z0-9_]+)\]\s*=\s*(?:React\.)?useState',
        r'(?:React\.)?useEffect\(\s*\(\s*\)\s*=>\s*{',
        r'(?:React\.)?useContext\(',
        r'(?:React\.)?useRef\(',
        r'(?:React\.)?useCallback\(',
        r'(?:React\.)?useMemo\(',
        r'(?:React\.)?useReducer\(',
        # Custom hooks detection
        r'(?:export\s+)?(?:function|const)\s+(use[A-Z][a-zA-Z0-9_]*)'
    ]
    
    IMPORT_PATTERN = r'import\s+(?:{([^}]*)},?\s*)?(?:([a-zA-Z0-9_]+)\s*,?\s*)?(?:from\s+)?[\'"]([^\'"]+)[\'"];?'
    
    def get_language(self) -> str:
        return "javascript"
    
    def _get_file_language(self, file_path: Path) -> str:
        """Determine the specific language variant based on extension."""
        suffix = file_path.suffix.lower()
        if suffix in ['.ts', '.tsx']:
            return "typescript"
        elif suffix == '.jsx':
            return "jsx"
        return "javascript"
    
    def parse_file(self, file_path: Path) -> FileInfo:
        """Parse a JavaScript/TypeScript file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            language = self._get_file_language(file_path)
            file_type = "source"
            
            # Check if it's likely a React component file
            has_jsx = self._has_jsx(content)
            has_react_import = "react" in content.lower() and "import" in content.lower()
            is_likely_component = has_jsx or has_react_import
            
            if is_likely_component:
                file_type = "react_component"
            
            file_info = FileInfo(
                path=str(file_path),
                language=language,
                file_type=file_type,
                size_bytes=os.path.getsize(file_path),
                is_binary=False
            )
            
            # Extract imports
            file_info.imports = self._extract_imports(content)
            
            # Extract components and hooks (symbols)
            file_info.symbols = self._extract_symbols(content, file_path)
            
            return file_info
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # Return with limited information
            return FileInfo(path=str(file_path), language=self._get_file_language(file_path), is_binary=False)
    
    def create_chunks(self, file_info: FileInfo) -> List[Chunk]:
        """Create chunks from a parsed JavaScript/TypeScript file."""
        chunks = []
        
        try:
            # Open the file to get content
            with open(file_info.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a chunk for file level (imports, top-level code)
            file_level_chunk = self._create_file_level_chunk(file_info, content)
            if file_level_chunk:
                chunks.append(file_level_chunk)
            
            # Process each symbol to create component/hook chunks
            used_lines = set()
            for symbol_name, symbol_info in file_info.symbols.items():
                # Skip symbols without line information
                if 'start_line' not in symbol_info or 'end_line' not in symbol_info:
                    continue
                    
                symbol_type = symbol_info.get('type', 'unknown')
                start_line = symbol_info['start_line']
                end_line = symbol_info['end_line']
                
                # Extract the content for this symbol
                symbol_content = self._extract_content_lines(content, start_line, end_line)
                
                # Mark these lines as used
                for line_num in range(start_line, end_line + 1):
                    used_lines.add(line_num)
                
                language = file_info.language
                lang_prefix = "js" if language == "javascript" else "ts" if language == "typescript" else "jsx"
                
                # Create chunk ID and determine chunk type name
                chunk_type = symbol_type
                if symbol_type == 'function_component' or symbol_type == 'class_component':
                    chunk_type = 'component'
                
                chunk_id = f"{lang_prefix}_{uuid.uuid4().hex[:8]}_{chunk_type}_{symbol_name}"
                
                # Create chunk
                chunk = Chunk(
                    id=chunk_id,
                    content=symbol_content,
                    metadata={
                        "file_path": file_info.path,
                        "language": language,
                        "type": symbol_type,
                        "name": symbol_name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "references": symbol_info.get('references', []),
                        "props": symbol_info.get('props', []),
                        "hooks": symbol_info.get('hooks', [])
                    }
                )
                chunks.append(chunk)
            
            # Create a chunk for any remaining code
            remaining_chunk = self._create_remaining_chunk(content, file_info, used_lines)
            if remaining_chunk:
                chunks.append(remaining_chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error creating chunks for {file_info.path}: {e}")
            return chunks
    
    def _has_jsx(self, content: str) -> bool:
        """Check if the content likely contains JSX."""
        # Look for JSX patterns like <Component> or
        jsx_pattern = r'<([A-Z][a-zA-Z0-9_]*|[a-z]+)[^>]*>.*?</\1>'
        return bool(re.search(jsx_pattern, content, re.DOTALL))

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from JavaScript/TypeScript."""
        imports = []
        
        for match in re.finditer(self.IMPORT_PATTERN, content):
            named_imports, default_import, module_path = match.groups()
            
            if named_imports:
                # Handle named imports like import { Component1, Component2 } from 'module'
                for named_item in named_imports.split(','):
                    named_item = named_item.strip()
                    if named_item:
                        # Handle "as" syntax
                        if ' as ' in named_item:
                            original, alias = named_item.split(' as ')
                            imports.append(alias.strip())
                        else:
                            imports.append(named_item)
            
            if default_import:
                # Handle default imports like import React from 'react'
                imports.append(default_import.strip())
        
        return imports

    def _extract_symbols(self, content: str, file_path: Path) -> Dict[str, Dict]:
        """Extract React components and hooks from JavaScript/TypeScript."""
        symbols = {}
        lines = content.split('\n')
        
        # Find function components
        for pattern in self.FUNC_COMPONENT_PATTERNS:
            for match in re.finditer(pattern, content):
                component_name = match.group(1)
                if component_name:
                    # Get the start position of the match
                    start_pos = match.start()
                    
                    # Find the line number for this position
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Approximate the end by finding the closing brace
                    end_line = self._find_function_end(lines, start_line)
                    
                    # Extract props from the function signature
                    props = self._extract_props_from_function(match.group(0))
                    
                    # Find hooks used in this component
                    hooks = self._find_hooks_in_range(content, start_line, end_line)
                    
                    # Function component information
                    symbols[component_name] = {
                        'type': 'function_component',
                        'start_line': start_line,
                        'end_line': end_line,
                        'props': props,
                        'hooks': hooks,
                        'references': []  # Will populate in a later pass
                    }
        
        # Find class components
        for pattern in self.CLASS_COMPONENT_PATTERNS:
            for match in re.finditer(pattern, content):
                component_name = match.group(1)
                if component_name:
                    # Get the start position of the match
                    start_pos = match.start()
                    
                    # Find the line number for this position
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Approximate the end of the class
                    end_line = self._find_class_end(lines, start_line)
                    
                    # Find methods in the class
                    methods = self._find_class_methods(content, start_line, end_line)
                    
                    # Class component information
                    symbols[component_name] = {
                        'type': 'class_component',
                        'start_line': start_line,
                        'end_line': end_line,
                        'methods': methods,
                        'references': []
                    }
        
        # Find custom hooks (not inside components)
        for match in re.finditer(self.HOOK_PATTERNS[-1], content):
            hook_name = match.group(1)
            if hook_name and hook_name.startswith('use') and hook_name[3].isupper():
                # Avoid hooks that we've already found in components
                already_defined = False
                for comp_info in symbols.values():
                    if hook_name in comp_info.get('hooks', []):
                        already_defined = True
                        break
                
                if not already_defined:
                    # Get the start position of the match
                    start_pos = match.start()
                    
                    # Find the line number for this position
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Approximate the end by finding the closing brace
                    end_line = self._find_function_end(lines, start_line)
                    
                    # Hook information
                    symbols[hook_name] = {
                        'type': 'custom_hook',
                        'start_line': start_line,
                        'end_line': end_line,
                        'references': []
                    }
        
        return symbols

    def _extract_props_from_function(self, func_signature: str) -> List[str]:
        """Extract props from a function component signature."""
        props = []
        
        # Look for props in the function parameters
        params_match = re.search(r'\(\s*{([^}]*)}', func_signature)
        if params_match:
            params_str = params_match.group(1)
            # Extract prop names, handling destructuring
            for prop in re.findall(r'([a-zA-Z0-9_]+)(?::\s*[a-zA-Z<>[\]|]+)?(?:\s*=\s*[^,]+)?', params_str):
                props.append(prop)
        
        return props

    def _find_hooks_in_range(self, content: str, start_line: int, end_line: int) -> List[str]:
        """Find React hooks used within a component."""
        hooks = []
        
        # Get the content within the line range
        lines = content.split('\n')
        range_content = '\n'.join(lines[start_line-1:end_line])
        
        # Look for useState hooks
        for match in re.finditer(r'const\s+\[([a-zA-Z0-9_]+),\s*set([A-Z][a-zA-Z0-9_]+)\]\s*=\s*(?:React\.)?useState', range_content):
            state_var = match.group(1)
            hooks.append(f"useState({state_var})")
        
        # Look for other standard hooks
        standard_hooks = [
            (r'(?:React\.)?useEffect\(\s*\(\s*\)\s*=>\s*{', 'useEffect'),
            (r'(?:React\.)?useContext\(\s*([a-zA-Z0-9_]+)', 'useContext'),
            (r'(?:React\.)?useRef\(\s*([a-zA-Z0-9_]+)?', 'useRef'),
            (r'(?:React\.)?useCallback\(\s*\(', 'useCallback'),
            (r'(?:React\.)?useMemo\(\s*\(\s*\)\s*=>', 'useMemo'),
            (r'(?:React\.)?useReducer\(\s*([a-zA-Z0-9_]+)', 'useReducer')
        ]
        
        for pattern, hook_name in standard_hooks:
            if re.search(pattern, range_content):
                hooks.append(hook_name)
        
        # Look for custom hooks
        for match in re.finditer(r'(?:const|let|var)\s+[a-zA-Z0-9_]+\s*=\s*(use[A-Z][a-zA-Z0-9_]*)\(', range_content):
            hooks.append(match.group(1))
        
        return hooks

    def _find_class_methods(self, content: str, start_line: int, end_line: int) -> List[str]:
        """Find methods in a class component."""
        methods = []
        
        # Get the content within the line range
        lines = content.split('\n')
        range_content = '\n'.join(lines[start_line-1:end_line])
        
        # Look for class methods
        method_pattern = r'(?:async\s+)?([a-zA-Z0-9_]+)\s*\([^)]*\)\s*{'
        for match in re.finditer(method_pattern, range_content):
            method_name = match.group(1)
            # Skip constructor
            if method_name != 'constructor' and not method_name.startswith('_'):
                methods.append(method_name)
        
        # Also look for lifecycle methods
        lifecycle_methods = [
            'componentDidMount', 'componentDidUpdate', 'componentWillUnmount',
            'shouldComponentUpdate', 'getSnapshotBeforeUpdate', 'componentDidCatch',
            'render'
        ]
        
        for method in lifecycle_methods:
            if re.search(rf'{method}\s*\(', range_content):
                methods.append(method)
        
        return methods

    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a function based on balanced braces."""
        brace_count = 0
        in_function = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            # Count opening and closing braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1
            
            # If we've found the matching closing brace
            if in_function and brace_count <= 0:
                return i + 1
        
        # If we couldn't find the end, make a reasonable estimate
        return min(start_line + 50, len(lines))

    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a class based on balanced braces."""
        # Same implementation as _find_function_end for now
        return self._find_function_end(lines, start_line)

    def _extract_content_lines(self, content: str, start_line: int, end_line: int) -> str:
        """Extract specific line range from content."""
        lines = content.split('\n')
        # Convert from 1-indexed to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        return '\n'.join(lines[start_idx:end_idx])

    def _create_file_level_chunk(self, file_info: FileInfo, content: str) -> Optional[Chunk]:
        """Create a chunk for file-level code (imports, etc.)."""
        # Extract the imports section
        lines = content.split('\n')
        import_lines = []
        
        for i, line in enumerate(lines):
            if re.match(self.IMPORT_PATTERN, line):
                import_lines.append(line)
            elif import_lines and not line.strip():
                # Include blank lines between imports
                import_lines.append(line)
            elif import_lines and not line.startswith('import'):
                # Stop when we reach non-import code after imports started
                break
        
        if import_lines:
            import_content = '\n'.join(import_lines)
            
            language = file_info.language
            lang_prefix = "js" if language == "javascript" else "ts" if language == "typescript" else "jsx"
            
            chunk_id = f"{lang_prefix}_{uuid.uuid4().hex[:8]}_imports_{Path(file_info.path).stem}"
            
            return Chunk(
                id=chunk_id,
                content=import_content,
                metadata={
                    "file_path": file_info.path,
                    "language": language,
                    "type": "imports",
                    "name": "Imports",
                    "start_line": 1,
                    "end_line": len(import_lines),
                    "imports": file_info.imports,
                }
            )
        
        return None

    def _create_remaining_chunk(self, content: str, file_info: FileInfo, used_lines: Set[int]) -> Optional[Chunk]:
        """Create a chunk for code not covered by component chunks."""
        lines = content.split('\n')
        remaining_lines = []
        
        for i, line in enumerate(lines):
            line_num = i + 1  # Convert to 1-indexed
            if line_num not in used_lines and line.strip():
                remaining_lines.append(line)
        
        if remaining_lines:
            remaining_content = '\n'.join(remaining_lines)
            
            language = file_info.language
            lang_prefix = "js" if language == "javascript" else "ts" if language == "typescript" else "jsx"
            
            chunk_id = f"{lang_prefix}_{uuid.uuid4().hex[:8]}_other_{Path(file_info.path).stem}"
            
            return Chunk(
                id=chunk_id,
                content=remaining_content,
                metadata={
                    "file_path": file_info.path,
                    "language": language,
                    "type": "other",
                    "name": "Other code",
                }
            )
        
        return None