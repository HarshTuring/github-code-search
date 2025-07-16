from .tree_sitter_parser import TreeSitterParser

class TreeSitterJavaScriptParser(TreeSitterParser):
    """JavaScript/TypeScript parser using Tree-sitter."""
    
    def __init__(self):
        super().__init__('javascript')