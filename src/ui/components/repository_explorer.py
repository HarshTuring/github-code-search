import streamlit as st
import os
from pathlib import Path
from src.ui.state import navigate_to
from src.embeddings.vector_store import VectorStoreManager
from src.ui.components.file_explorer import render_file_explorer
from src.ui.components.code_viewer import render_code_viewer
from src.ui.components.file_summary import render_file_summary
from src.ui.components.repository_search import render_repository_search


def get_available_repositories():
    """
    Get a list of repositories that have been processed and are ready to explore.
    
    Returns:
        list: List of repository information dictionaries with name and collection_name
    """
    try:
        # Use VectorStoreManager to list repositories
        vector_store_manager = VectorStoreManager()
        repositories = vector_store_manager.list_repositories()
        
        # Format into a more useful structure
        formatted_repos = []
        for repo_name in repositories:
            # Format the display name (replace underscores with slashes for readability)
            display_name = repo_name
            if "_" in repo_name:
                # This might be a username_repo format, try to make it more readable
                parts = repo_name.split("_", 1)
                if len(parts) == 2:
                    display_name = f"{parts[0]}/{parts[1]}"
            
            formatted_repos.append({
                "name": display_name,
                "collection_name": repo_name
            })
        
        return formatted_repos
    except Exception as e:
        st.error(f"Error loading available repositories: {e}")
        return []

def render_repository_list():
    """Render a list of available repositories that can be explored."""
    st.title("Repository Explorer")
    st.write("Select a repository to explore its contents.")
    
    # Get available repositories
    repositories = get_available_repositories()
    
    if not repositories:
        st.warning("No processed repositories found. Please analyze a repository first.")
        
        # Add a button to go back to the home page
        if st.button("Analyze a Repository", use_container_width=True):
            navigate_to("home")
        
        st.info("To analyze a repository, go to the Home page and enter a GitHub repository URL.")
        return
    
    # Display repositories as a list of cards
    st.subheader("Available Repositories:")
    
    # Use columns to create a grid layout
    cols = st.columns(3)
    
    for i, repo in enumerate(repositories):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {repo['name']}")
                
                # Calculate repository directory path to check if files exist
                repo_name_short = repo['name'].split('/')[-1] if '/' in repo['name'] else repo['name']
                repo_dir = Path(f"./data/repos/{repo_name_short}")
                
                # Show additional information if available
                if repo_dir.exists():
                    # Count files in repository
                    file_count = sum(1 for _ in repo_dir.glob('**/*') if _.is_file())
                    st.write(f"Files: {file_count}")
                    
                    # Show repository size
                    size_mb = sum(_.stat().st_size for _ in repo_dir.glob('**/*') if _.is_file()) / (1024 * 1024)
                    st.write(f"Size: {size_mb:.1f} MB")
                
                # Explore button for this repository
                if st.button("Explore", key=f"explore_{i}", use_container_width=True):
                    # Store the selected repository in session state
                    st.session_state.selected_repository = repo
                    
                    # Determine and store the repository path
                    repo_name = repo['name'].split('/')[-1] if '/' in repo['name'] else repo['name']
                    repo_path = os.path.join("./data/repos", repo_name)
                    st.session_state.repo_path = repo_path
                    st.session_state.repo_name = repo['name']
                    
                    # Set flag to indicate we're now exploring a repository
                    st.session_state.exploring_repository = True
                    
                    # Reset file selection
                    if 'selected_file' in st.session_state:
                        st.session_state.selected_file = None
                    
                    st.rerun()

def render_file_explorer_interface():
    """Render the file explorer interface for a selected repository."""
    # Back button to return to repository list
    if st.button("← Back to Repository List", use_container_width=True):
        st.session_state.exploring_repository = False
        if 'selected_file' in st.session_state:
            st.session_state.selected_file = None
        st.rerun()
    
    st.title(f"Exploring: {st.session_state.repo_name}")
    
    # Add tabs for file explorer and search
    tab1, tab2 = st.tabs(["File Explorer", "Search"])
    
    with tab1:
        # Create a two-column layout
        col1, col2 = st.columns([1, 2])
        
        # Left column: File Explorer
        with col1:
            render_file_explorer()
        
        # Right column: Code Viewer and Summary
        with col2:
            # Render code viewer, which returns file content if successful
            file_content = render_code_viewer()
            
            # Render file summary component if we have file content
            if file_content is not None:
                render_file_summary(file_content)
    
    with tab2:
        # Render search interface
        render_repository_search()
        
        # If a file is selected from search, show it
        if st.session_state.get("selected_file"):
            st.markdown("---")
            file_content = render_code_viewer()
            if file_content is not None:
                render_file_summary(file_content)

def render_repository_explorer():
    """
    Main repository explorer component.
    Shows either repository list or file explorer based on state.
    """
    # Check if we're exploring a specific repository
    if not st.session_state.get('exploring_repository', False):
        render_repository_list()
    else:
        render_file_explorer_interface()