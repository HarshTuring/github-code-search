import os
import streamlit as st
from datetime import datetime

def get_file_size_str(size_bytes):
    """Convert file size in bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def render_file_explorer():
    """Render the file explorer component."""
    if "repo_path" not in st.session_state or not st.session_state.repo_path:
        st.warning("Repository path not found.")
        return
    
    st.subheader("File Explorer")
    repo_path = st.session_state.repo_path
    
    # Check if repo_path exists
    if not os.path.exists(repo_path):
        st.error(f"Repository path does not exist: {repo_path}")
        return
    
    # Initialize expanded directories if needed
    if "file_tree_expanded_dirs" not in st.session_state:
        st.session_state.file_tree_expanded_dirs = set([""])  # Root is expanded by default
    
    # Build a complete file tree structure for flat rendering
    file_tree = _build_file_tree(repo_path)
    
    # Render the file tree in a flat structure with visual indentation
    _render_flat_file_tree(file_tree, repo_path)

def _build_file_tree(repo_path):
    """
    Build a tree structure representing the repository contents.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        dict: A nested dictionary representing the file tree
    """
    tree = {}
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Get the relative path from the repo root
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            rel_path = ''
        
        # Skip hidden files
        files = [f for f in files if not f.startswith('.')]
        
        # Find current position in the tree
        current = tree
        if rel_path:
            parts = rel_path.split(os.path.sep)
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Add files (marked with None as value)
        for file in sorted(files):
            current[file] = None
    
    return tree

def _render_flat_file_tree(tree, repo_path, prefix="", path=""):
    """
    Render the file tree as a flat structure with indentation.
    
    Args:
        tree: The file tree dictionary
        repo_path: Base repository path
        prefix: Current indentation prefix
        path: Current path in the repository
    """
    # Sort items to show directories first, then files
    items = sorted(tree.items(), key=lambda x: (x[1] is None, x[0]))
    
    for name, subtree in items:
        full_path = os.path.join(path, name)
        is_dir = subtree is not None
        
        # Calculate indentation
        indent = "│   " * (prefix.count("│") + (1 if prefix and not prefix.endswith("└─ ") else 0))
        
        # Create a unique key for this item
        item_id = f"tree_item_{full_path.replace('/', '_').replace(' ', '_').replace('.', '_')}"
        
        # Display directory or file with indentation
        cols = st.columns([3, 1, 1])
        
        # Icon and indentation for file/directory
        icon = "📁" if is_dir else "📄"
        display_name = f"{prefix}{icon} {name}"
        
        with cols[0]:
            # For directories, add a toggle button
            if is_dir:
                # Check if directory is expanded
                is_expanded = full_path in st.session_state.file_tree_expanded_dirs
                
                # Create a button that toggles expansion
                if st.button(
                    display_name + (" 🔽" if is_expanded else " ▶️"), 
                    key=item_id
                ):
                    # Toggle directory expansion
                    if full_path in st.session_state.file_tree_expanded_dirs:
                        st.session_state.file_tree_expanded_dirs.remove(full_path)
                    else:
                        st.session_state.file_tree_expanded_dirs.add(full_path)
                    st.rerun()
            else:
                # For files, create a selectable button
                if st.button(display_name, key=item_id):
                    st.session_state.selected_file = full_path
                    st.rerun()
        
        # Only show metadata for files
        if not is_dir:
            file_path = os.path.join(repo_path, full_path)
            try:
                file_size = os.path.getsize(file_path)
                mod_time = os.path.getmtime(file_path)
                with cols[1]:
                    st.write(get_file_size_str(file_size))
                with cols[2]:
                    st.write(datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d"))
            except Exception:
                with cols[1]:
                    st.write("N/A")
                with cols[2]:
                    st.write("N/A")
        
        # If this is a directory and it's expanded, show its contents
        if is_dir and full_path in st.session_state.file_tree_expanded_dirs:
            # Adjust prefix for the next level of items
            if prefix:
                new_prefix = prefix.replace("└─ ", "    ").replace("├─ ", "│   ")
            else:
                new_prefix = ""
                
            # Recursively render the subtree
            _render_flat_file_tree(
                subtree, 
                repo_path, 
                new_prefix + "├─ " if subtree else new_prefix + "└─ ",
                full_path
            )