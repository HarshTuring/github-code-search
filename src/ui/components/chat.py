import streamlit as st
import os
import time
from pathlib import Path
from src.ui.state import navigate_to
from src.embeddings.vector_store import VectorStore, VectorStoreManager
from src.embeddings.embedding_generator import EmbeddingGenerator

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
                    print(f"Chat button clicked for repo: {repo}")
                    # Store the selected repository in session state
                    st.session_state.selected_repository = repo
                    print(f"Session state after setting repo: {st.session_state}")
                    # Navigate to chat interface
                    print("Navigating to chat...")
                    navigate_to("chat")
                    print("Navigation complete")

def render_chat():
    """
    Render the chat interface component.
    Shows either the repository list or the chat interface based on state.
    """
    print("Rendering chat interface...")
    print(f"Current page: {st.session_state.current_page}")
    print(f"Selected repository: {st.session_state.get('selected_repository')}")
    
    # Show repository list if no repository is selected
    if "selected_repository" not in st.session_state or not st.session_state.selected_repository:
        print("No repository selected, showing repository list")
        render_repository_list()
        return
        
    # If we get here, we have a selected repository and should show the chat interface
    print("Repository selected, showing chat interface")
    
    # Get the selected repository
    repo = st.session_state.selected_repository
    print(f"Selected repository: {repo}")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        print("Initializing chat messages")
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ask me anything about the repository: {repo['name']}"}
        ]
    
    # Debug: Print current messages
    print(f"Current messages: {st.session_state.messages}")
    
    # Back button to return to repository list
    if st.button("← Back to Repository List", use_container_width=True):
        if "selected_repository" in st.session_state:
            del st.session_state.selected_repository
        if "messages" in st.session_state:
            del st.session_state.messages
        navigate_to("chat")
        return
    
    # Display chat messages
    st.title(f"Chat with {repo['name']}")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about this repository..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show thinking message with dots animation
                with st.status("Thinking...", expanded=True) as status:
                    # Show initial message
                    status.write("Analyzing your question")
                    
                    # Import the analyzer
                    from src.main import GitHubCodeAnalyzer
                    
                    # Initialize the analyzer
                    analyzer = GitHubCodeAnalyzer()
                    
                    # The repository name in the vector store uses underscores instead of slashes
                    repo_name = repo['name'].replace('/', '_')
                    
                    # Update status
                    status.write("Searching the codebase...")
                    
                    # Query the repository
                    response_data = analyzer.query_repository(
                        repo_name=repo_name,
                        query_text=prompt,
                        top_k=5  # Get top 5 most relevant chunks
                    )
                    
                    # Clear the status when done
                    status.update(label="Analysis complete", state="complete", expanded=False)
                
                if response_data.get('success', False):
                    # Format the response from the analyzer
                    response = response_data.get('response', 'No response generated')
                    
                    # Add sources if available
                    if 'sources' in response_data and response_data['sources']:
                        response += "\n\nSources:"
                        for i, source in enumerate(response_data['sources'], 1):
                            file_path = source.get('file_path', 'Unknown')
                            response += f"\n{i}. {file_path}"
                else:
                    error_msg = response_data.get('error', 'Unknown error')
                    response = f"Error processing your query: {error_msg}"
                
                # Simulate streaming the response
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.02)  # Faster typing effect
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"Error querying the repository: {str(e)}"
                print(error_msg)
                full_response = error_msg
                message_placeholder.error(error_msg)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})