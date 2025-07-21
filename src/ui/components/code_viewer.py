import os
import streamlit as st
import mimetypes

def determine_language(file_path):
    """Determine the programming language based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.md': 'markdown',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.xml': 'xml',
        '.sql': 'sql',
        '.sh': 'bash',
    }
    
    return language_map.get(ext, '')

def is_binary_file(file_path, sample_size=1024):
    """Check if a file is binary by looking at a sample of its content."""
    # Check by mimetype first
    mime, _ = mimetypes.guess_type(file_path)
    if mime and not mime.startswith('text/'):
        # Check for common text mimetypes that don't start with 'text/'
        text_mimes = ['application/json', 'application/javascript', 'application/xml']
        if not any(mime.startswith(tm) for tm in text_mimes):
            return True
    
    # Check common binary extensions
    binary_extensions = {'.exe', '.bin', '.obj', '.o', '.dll', '.so', '.pyc',
                        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
                        '.zip', '.gz', '.tar', '.pdf', '.doc', '.docx', '.ppt'}
    if os.path.splitext(file_path)[1].lower() in binary_extensions:
        return True
    
    # Check content
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        return b'\0' in sample  # Binary files often contain null bytes
    except:
        # If we can't read the file, assume it's not binary
        return False

def render_code_viewer():
    """Render the code viewer component."""
    if "selected_file" not in st.session_state or not st.session_state.selected_file:
        st.info("Select a file from the file explorer to view its contents.")
        return None
        
    if "repo_path" not in st.session_state or not st.session_state.repo_path:
        st.warning("Repository path not found.")
        return None
        
    file_rel_path = st.session_state.selected_file
    file_abs_path = os.path.join(st.session_state.repo_path, file_rel_path)
    
    # Display file path
    st.subheader(f"File: {file_rel_path}")
    
    # Display file metadata
    try:
        file_size = os.path.getsize(file_abs_path)
        mod_time = os.path.getmtime(file_abs_path)
        
        from src.ui.components.file_explorer import get_file_size_str
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Size: {get_file_size_str(file_size)}")
        with col2:
            from datetime import datetime
            st.info(f"Last modified: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.warning(f"Could not get file metadata: {str(e)}")
    
    # Check if file is binary
    if is_binary_file(file_abs_path):
        st.warning("This appears to be a binary file and cannot be displayed.")
        return None
    
    # Read and display file content
    try:
        with open(file_abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Determine language for syntax highlighting
        language = determine_language(file_rel_path)
        
        # Display code with syntax highlighting
        st.code(content, language=language)
        
        # Return content for potential use by summary component
        return content
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None