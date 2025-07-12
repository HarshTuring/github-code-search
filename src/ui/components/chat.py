import streamlit as st
import os
from pathlib import Path
from src.ui.state import navigate_to
from src.embeddings.vector_store import VectorStoreManager

def get_available_repositories():
    """
    Get a list of repositories that have been processed and are ready to be queried.
    
    Returns:
        list: List of repository information dictionaries with name and collection_name
    """
    try:
        # Use VectorStoreManager from the correct module
        vector_store_manager = VectorStoreManager()
        repositories = vector_store_manager.list_repositories()
        
        # Format into a more useful structure if needed
        formatted_repos = []
        for repo_name in repositories:
            # Format the display name (replace underscores with slashes for readability if needed)
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
    """
    Render a list of available repositories that can be queried.
    """
    st.title("Chat with Repository")
    st.write("Select a repository to start chatting with.")
    
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
                
                # Chat button for this repository
                if st.button("Chat", key=f"chat_{i}", use_container_width=True):
                    # Store the selected repository in session state
                    st.session_state.selected_repository = repo
                    # Navigate to chat interface
                    navigate_to("repository_chat")

def render_chat():
    """
    Render the chat interface component.
    Will either show the repository list or the actual chat interface
    depending on the state.
    """
    # Check if we need to show the repository list or the actual chat interface
    if "selected_repository" not in st.session_state or st.session_state.current_page == "chat":
        render_repository_list()
    else:
        # In a future implementation, this will render the actual chat interface
        st.title(f"Chat with {st.session_state.selected_repository['name']}")
        st.write("Chat interface will be implemented in the next step.")
        
        # Add a back button
        if st.button("← Back to Repository List", use_container_width=True):
            # Clear the selected repository and go back to the list
            if "selected_repository" in st.session_state:
                del st.session_state.selected_repository
            navigate_to("chat")