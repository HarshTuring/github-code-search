import streamlit as st
import os
import time
import re
from src.ui.state import set_repo_processing, update_repo_status
from src.fetcher.repository_fetcher import RepositoryFetcher
from src.parser.repository_parser import RepositoryParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore

def is_valid_github_url(url):
    """
    Check if the URL is a valid GitHub repository URL.
    
    Args:
        url (str): The URL to check.
        
    Returns:
        bool: True if URL is valid, False otherwise.
    """
    pattern = r'^https?://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+/?$'
    return bool(re.match(pattern, url))

def extract_repo_name(url):
    """
    Extract repository name from GitHub URL.
    
    Args:
        url (str): GitHub repository URL.
        
    Returns:
        str: Repository name.
    """
    # Extract username/reponame part from the URL
    match = re.search(r'github\.com/([a-zA-Z0-9-]+/[a-zA-Z0-9_.-]+)', url)
    if match:
        return match.group(1)
    return "unknown-repo"

def process_repository(url, target_dir="./data/repos"):
    """
    Process a repository: fetch, parse, and create embeddings.
    
    Args:
        url (str): GitHub repository URL.
        target_dir (str): Base directory to store repositories.
        
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    try:
        repo_name = extract_repo_name(url)
        st.session_state.repo_name = repo_name
        
        # Status tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Step 1: Fetch repository
        with progress_placeholder.container():
            st.write("Fetching repository...")
            progress_bar = st.progress(0)
        
        status_placeholder.info("Cloning repository from GitHub...")
        
        # Initialize the fetcher with the target directory
        fetcher = RepositoryFetcher(base_storage_path=target_dir)
        
        # Perform the fetch operation - let RepositoryFetcher handle the path
        repo_path = fetcher.fetch(url, repo_name.split('/')[-1])
        if not repo_path:
            status_placeholder.error("Failed to fetch repository.")
            return False
        
        st.session_state.repo_path = str(repo_path)
        progress_bar.progress(25)
        update_repo_status(fetch=True)
        status_placeholder.success("Repository fetched successfully!")
        time.sleep(0.5)  # Short pause for UX
        
        # Step 2: Parse repository
        status_placeholder.info("Parsing repository code...")
        progress_bar.progress(35)
        
        # Use the path returned by the fetcher
        parser = RepositoryParser(repo_path=repo_path)
        try:
            # First parse the repository to build the structure
            repo_structure = parser.parse()
            
            # Then create chunks from the parsed repository
            chunks = parser.create_chunks()
            
            if not chunks:
                status_placeholder.error("Failed to parse repository or no code chunks found.")
                return False
            
            progress_bar.progress(60)
            update_repo_status(parse=True)
            status_placeholder.success(f"Repository parsed successfully! Found {len(chunks)} code chunks.")
            time.sleep(0.5)  # Short pause for UX
            
            # Step 3: Generate embeddings and store in vector database
            status_placeholder.info("Generating embeddings and storing in vector database...")
            progress_bar.progress(70)
            
            # Initialize embedding generator with API key from environment
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                status_placeholder.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                return False
                
            embedding_generator = EmbeddingGenerator(api_key=openai_api_key)
            # Use the repository name for the vector store
            collection_name = repo_name.replace('/', '_')  # Replace / with _ for collection name
            vector_store = VectorStore(repo_name=collection_name)
            
            # Process chunks in batches to show progress
            batch_size = max(1, len(chunks) // 10)  # Split into ~10 batches for progress updates
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Generate embeddings for the batch
                embedded_chunks = embedding_generator.generate_embeddings(batch)
                
                # Store in vector database
                vector_store.add_chunks(embedded_chunks)
                
                # Update progress
                progress = 70 + (i // batch_size) * 3
                progress_bar.progress(min(95, progress))
            
            # Final progress update
            progress_bar.progress(100)
            update_repo_status(embed=True)
            status_placeholder.success("Repository processing complete! You can now chat with the repository.")
            
            return True
            
        except Exception as e:
            status_placeholder.error(f"Error processing repository: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def render_home():
    """
    Render the home page with repository input form.
    """
    st.title("GitHub Repository Analyzer")
    st.write("Enter a GitHub repository URL to analyze its codebase.")
    
    # Debug button to show current state
    if st.button("Debug: Show State"):
        st.write("### Current Session State")
        st.json({k: v for k, v in st.session_state.items() if k != 'chat_input_buffer'})
    
    # Repository input form
    with st.form(key="repo_form"):
        repo_url = st.text_input(
            "GitHub Repository URL", 
            placeholder="https://github.com/username/repository",
            value=st.session_state.repo_url
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button(
                label="Analyze Repository", 
                use_container_width=True,
                disabled=st.session_state.repo_processing
            )
        with col2:
            force_reload = st.checkbox("Force reload", value=False, 
                                       help="If checked, repository will be re-downloaded even if already processed")
    
    # Process repository
    if submit_button:
        # Validate URL
        if not repo_url:
            st.error("Please enter a GitHub repository URL.")
            return
        
        if not is_valid_github_url(repo_url):
            st.error("Please enter a valid GitHub repository URL (e.g., https://github.com/username/repository).")
            return
        
        # Store URL in session state
        st.session_state.repo_url = repo_url
        
        # Check if repository is already processed
        if not force_reload and st.session_state.repo_loaded and extract_repo_name(repo_url) == st.session_state.repo_name:
            st.success("Repository is already processed and ready for chat!")
            return
        
        # Reset processing status
        update_repo_status(fetch=False, parse=False, embed=False)
        st.session_state.repo_loaded = False
        set_repo_processing(True)
        
        # Process repository
        success = process_repository(repo_url)
        set_repo_processing(False)
        
        if success:
            st.success("Repository processing complete! You can now chat with the repository.")
        else:
            st.error("Failed to process repository. Please check the URL and try again.")
    
    # Show additional information if no repository is loaded
    if not st.session_state.repo_loaded and not st.session_state.repo_processing:
        st.markdown("---")
        st.subheader("How it works")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 1. Input Repository")
            st.write("Provide a GitHub repository URL to analyze.")
        
        with col2:
            st.markdown("### 2. Process Code")
            st.write("We'll fetch, parse, and create embeddings for the code.")
        
        with col3:
            st.markdown("### 3. Chat with Repository")
            st.write("Ask questions about the codebase in natural language.")