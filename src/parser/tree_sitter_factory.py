import logging
from typing import Dict, Type, Optional
from pathlib import Path

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
    _logger = logging.getLogger(__name__)
    
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
            cls._logger.warning(f"No Tree-sitter parser class for language: {language}")
            return None
            
        # Create and cache a new parser
        try:
            parser = parser_class()
            cls._parsers[language] = parser
            return parser
        except Exception as e:
            cls._logger.error(f"Error creating Tree-sitter parser for {language}: {str(e)}")
            return None
    
    @classmethod
    def supports_language(cls, language: str) -> bool:
        """Check if a language is supported by Tree-sitter."""
        return language in cls.PARSER_CLASSES
        
    @classmethod
    def ensure_query_files_exist(cls) -> None:
        """Ensure that query files exist for all supported languages."""
        # Create the language_queries directory if it doesn't exist
        queries_dir = Path(__file__).parent / "language_queries"
        queries_dir.mkdir(exist_ok=True)
        
        # Check each supported language
        for language in cls.PARSER_CLASSES.keys():
            query_file = queries_dir / f"{language}_queries.scm"
            
            # Create empty query file if it doesn't exist
            if not query_file.exists():
                cls._logger.warning(f"Creating empty query file for {language}")
                with open(query_file, 'w') as f:
                    f.write(f"; Tree-sitter queries for {language}\n")