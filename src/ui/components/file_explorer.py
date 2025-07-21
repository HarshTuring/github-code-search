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
    
    # Display the file tree starting from root
    _render_directory("", repo_path)

def _render_directory(rel_dir_path, repo_base_path):
    """Recursively render directory contents with lazy loading.
    
    Args:
        rel_dir_path: Relative path from repository root
        repo_base_path: Absolute base path of the repository
    """
    abs_dir_path = os.path.join(repo_base_path, rel_dir_path) if rel_dir_path else repo_base_path
    
    try:
        # Get directory entries
        entries = os.listdir(abs_dir_path)
        
        # Split into directories and files, and sort both
        dirs = sorted([e for e in entries if os.path.isdir(os.path.join(abs_dir_path, e)) and not e.startswith('.')])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(abs_dir_path, e)) and not e.startswith('.')])
        
        # Process directories
        for dirname in dirs:
            dir_rel_path = os.path.join(rel_dir_path, dirname) if rel_dir_path else dirname
            dir_id = f"dir_{dir_rel_path.replace(os.path.sep, '_').replace(' ', '_').replace('.', '_')}"
            
            # Check if directory is expanded in session state
            is_expanded = dir_rel_path in st.session_state.file_tree_expanded_dirs
            
            # Create expander for directory
            with st.expander(f"📁 {dirname}", expanded=is_expanded):
                # Mark as expanded when user opens it
                if not is_expanded:
                    st.session_state.file_tree_expanded_dirs.add(dir_rel_path)
                
                # Only render contents if expanded
                _render_directory(dir_rel_path, repo_base_path)
        
        # Process files with metadata in a more structured layout
        for filename in files:
            file_rel_path = os.path.join(rel_dir_path, filename) if rel_dir_path else filename
            file_abs_path = os.path.join(abs_dir_path, filename)
            file_id = f"file_{file_rel_path.replace(os.path.sep, '_').replace(' ', '_').replace('.', '_')}"
            
            # Get file metadata
            try:
                file_size = os.path.getsize(file_abs_path)
                mod_time = os.path.getmtime(file_abs_path)
                size_str = get_file_size_str(file_size)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size_str = "N/A"
                mod_time_str = "N/A"
            
            # Create a container with hover effect for each file
            with st.container():
                # Highlight the selected file
                is_selected = st.session_state.get('selected_file') == file_rel_path
                bg_color = "#f0f7ff" if is_selected else "transparent"
                
                # Create columns with proper alignment and spacing
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    if st.button(f"📄 {filename}", key=file_id):
                        st.session_state.selected_file = file_rel_path
                        st.rerun()
                with cols[1]:
                    st.text(size_str)
                with cols[2]:
                    st.text(mod_time_str)
                
                # Add a subtle divider after each file
                st.markdown('<hr style="margin: 0.1em 0; border: none; height: 1px; background-color: #f0f0f0;">', unsafe_allow_html=True)
    
    except PermissionError:
        st.warning(f"Permission denied: {rel_dir_path}")
    except Exception as e:
        st.error(f"Error reading directory: {str(e)}")