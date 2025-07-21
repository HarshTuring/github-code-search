import os
import re
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.ui.components.code_viewer import is_binary_file

def render_search_bar():
    """Render search bar for repository searching."""
    st.write("### Search Repository")
    
    # Search input
    search_query = st.text_input("Search for files or content:", 
                                placeholder="Enter search term",
                                key="search_query")
    
    # Search options
    col1, col2, col3 = st.columns(3)
    with col1:
        search_filenames = st.checkbox("Filenames", value=True, key="search_filenames")
    with col2:
        search_content = st.checkbox("File content", value=True, key="search_content")
    with col3:
        case_sensitive = st.checkbox("Case sensitive", value=False, key="search_case_sensitive")
    
    # Search button
    search_submitted = st.button("Search", use_container_width=True, key="search_button")
    
    if search_submitted and search_query and (search_filenames or search_content):
        with st.spinner(f"Searching repository for '{search_query}'..."):
            results = search_repository(
                search_query,
                st.session_state.repo_path,
                search_filenames=search_filenames,
                search_content=search_content,
                case_sensitive=case_sensitive
            )
            st.session_state.search_results = results
            st.session_state.search_performed = True
            st.session_state.search_query_display = search_query
            st.rerun()

def render_search_results():
    """Render search results if a search has been performed."""
    if not st.session_state.get("search_performed"):
        return
    
    results = st.session_state.get("search_results", {})
    query = st.session_state.get("search_query_display", "")
    
    filename_results = results.get("filename_matches", [])
    content_results = results.get("content_matches", [])
    
    # Display search summary
    total_results = len(filename_results) + sum(len(matches) for _, matches in content_results)
    st.write(f"### Search Results for '{query}'")
    st.write(f"Found {total_results} matches")
    
    # Clear search results button
    if st.button("Clear Search Results", key="clear_search"):
        st.session_state.search_performed = False
        if "search_results" in st.session_state:
            del st.session_state.search_results
        st.rerun()
    
    # Display filename matches
    if filename_results:
        with st.expander(f"Filename Matches ({len(filename_results)})", expanded=True):
            for file_path in filename_results:
                # Create a clickable button for each file
                if st.button(f"📄 {file_path}", key=f"result_file_{file_path.replace('/', '_')}"):
                    st.session_state.selected_file = file_path
                    st.session_state.search_performed = False  # Clear search results
                    st.rerun()
    
    # Display content matches
    if content_results:
        total_content_matches = sum(len(matches) for _, matches in content_results)
        st.subheader(f"Content Matches ({total_content_matches})")
        
        # Create a selectbox to choose which file's matches to view
        file_options = [f"{path} ({len(matches)} matches)" for path, matches in content_results]
        selected_file_idx = 0
        if len(file_options) > 1:
            selected_file_idx = st.selectbox(
                "Select file to view matches:", 
                range(len(file_options)),
                format_func=lambda i: file_options[i]
            )
        
        # Display matches for the selected file
        file_path, matches = content_results[selected_file_idx]
        
        st.write(f"### 📄 {file_path} ({len(matches)} matches)")
        
        # Display each match with context
        for i, (line_num, line_text, context) in enumerate(matches):
            st.code(f"{context}", language='text')
            
            # Add buttons for each match
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button(f"Open at line {line_num}", key=f"goto_{file_path}_{line_num}_{i}"):
                    st.session_state.selected_file = file_path
                    st.session_state.goto_line = line_num
                    st.session_state.search_performed = False
                    st.rerun()
            
        # Add a single button to open the full file at the bottom
        if st.button(f"Open full file: {file_path}", key=f"open_{file_path}"):
            st.session_state.selected_file = file_path
            st.session_state.search_performed = False
            st.rerun()

def search_repository(query, repo_path, search_filenames=True, search_content=True, case_sensitive=False):
    """
    Search repository for files matching query in name or content.
    
    Args:
        query (str): Search query
        repo_path (str): Path to repository
        search_filenames (bool): Whether to search filenames
        search_content (bool): Whether to search file content
        case_sensitive (bool): Whether search should be case sensitive
    
    Returns:
        dict: Dictionary with search results
    """
    results = {
        "filename_matches": [],
        "content_matches": []
    }
    
    if not case_sensitive:
        query = query.lower()
    
    # Compile regex for content search
    try:
        if case_sensitive:
            regex = re.compile(re.escape(query))
        else:
            regex = re.compile(re.escape(query), re.IGNORECASE)
    except:
        # Fallback if regex compilation fails
        regex = None
    
    # Walk through repository
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Process each file
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            # Search filenames
            if search_filenames:
                filename = file if case_sensitive else file.lower()
                if query in filename:
                    results["filename_matches"].append(rel_path)
            
            # Search content (skip binary files)
            if search_content and not is_binary_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        matches = []
                        lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            line_num = i + 1
                            line_text = line if case_sensitive else line.lower()
                            
                            if query in line_text:
                                # Get context (lines before and after)
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                
                                # Format context with line numbers and highlighting
                                context_lines = []
                                for j in range(start, end):
                                    prefix = f"{j+1}: " 
                                    if j == i:
                                        # This is the matching line - highlight it
                                        context_lines.append(f"{prefix}→ {lines[j].rstrip()}")
                                    else:
                                        context_lines.append(f"{prefix}  {lines[j].rstrip()}")
                                
                                # Join context lines
                                context = "\n".join(context_lines)
                                matches.append((line_num, line.strip(), context))
                        
                        if matches:
                            results["content_matches"].append((rel_path, matches))
                except Exception as e:
                    # Skip files we can't read
                    continue
    
    return results

def render_repository_search():
    """Render the repository search component."""
    # Initialize search state if needed
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    
    # Render search bar
    render_search_bar()
    
    # Render search results if search was performed
    render_search_results()