from typing import Dict, Type, Optional
from .language_parsers.tree_sitter_parser import TreeSitterParser
from .language_parsers.tree_sitter_python_parser import TreeSitterPythonParser
from .language_parsers.tree_sitter_javascript_parser import TreeSitterJavaScriptParser
from .language_parsers.tree_sitter_go_parser import TreeSitterGoParser

class TreeSitterFactory:
    """
    Factory for creating Tree-sitter parsers for different languages.
    """
    
    # Map of language names to parser classes
    PARSER_CLASSES: Dict[str, Type[TreeSitterParser]] = {
        'python': TreeSitterPythonParser,
        'javascript': TreeSitterJavaScriptParser,
        'typescript': TreeSitterJavaScriptParser,
        'jsx': TreeSitterJavaScriptParser,
        'tsx': TreeSitterJavaScriptParser,
        'go': TreeSitterGoParser,
    }
    
    # Singleton parsers
    _parsers: Dict[str, TreeSitterParser] = {}
    
    @classmethod
    def get_parser(cls, language: str) -> Optional[TreeSitterParser]:
        """
        Get a Tree-sitter parser for the specified language.
        
        Args:
            language: Name of the language
            
        Returns:
            TreeSitterParser instance or None if language is not supported
        """
        # Check if we already have a parser for this language
        if language in cls._parsers:
            return cls._parsers[language]
            
        # Check if we have a parser class for this language
        parser_class = cls.PARSER_CLASSES.get(language)
        if not parser_class:
            return None
            
        # Create and cache a new parser
        parser = parser_class()
        cls._parsers[language] = parser
        
        return parser
    
    @classmethod
    def supports_language(cls, language: str) -> bool:
        """Check if a language is supported by Tree-sitter."""
        return language in cls.PARSER_CLASSES