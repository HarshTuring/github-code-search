from .tree_sitter_parser import TreeSitterParser

class TreeSitterGoParser(TreeSitterParser):
    """Go parser using Tree-sitter."""
    
    def __init__(self):
        super().__init__('go')