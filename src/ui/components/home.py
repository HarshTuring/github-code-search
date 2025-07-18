import streamlit as st
import os
import time
import re
import requests
from urllib.parse import urlencode
from src.ui.state import set_repo_processing, update_repo_status
from src.fetcher.repository_fetcher import RepositoryFetcher
from src.parser.repository_parser import RepositoryParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_REPOS_URL = "https://api.github.com/user/repos"
REDIRECT_URI = "http://localhost:8501"

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

def get_github_auth_url():
    """Generate GitHub OAuth URL"""
    params = {
        'client_id': GITHUB_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'repo,user',
        'allow_signup': 'true'
    }
    return f"{GITHUB_AUTH_URL}?{urlencode(params)}"

def get_github_user(access_token):
    """Get GitHub user info using access token"""
    headers = {'Authorization': f'token {access_token}'}
    response = requests.get(GITHUB_USER_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def get_user_repos(access_token):
    """Get all repositories for the authenticated user (including private ones)"""
    headers = {'Authorization': f'token {access_token}'}
    all_repos = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API
    
    while True:
        params = {
            'per_page': per_page,
            'page': page,
            'affiliation': 'owner,collaborator',
            'sort': 'updated',
            'direction': 'desc'
        }
        
        response = requests.get(GITHUB_REPOS_URL, headers=headers, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            break
            
        all_repos.extend(repos)
        
        # Check if we've reached the last page
        if len(repos) < per_page:
            break
            
        page += 1
    
    return all_repos

def render_github_connect():
    """Render GitHub OAuth connection tab"""
    st.write("Connect your GitHub account to access your repositories.")
    
    if 'github_token' in st.session_state and 'github_user' in st.session_state:
        try:
            user = st.session_state.github_user
            username = user.get('login', 'GitHub User')
            display_name = user.get('name', username)
            
            # Display user info in a nicer format
            col1, col2 = st.columns([1, 3])
            with col1:
                if user.get('avatar_url'):
                    st.image(user['avatar_url'], width=60)
            with col2:
                st.markdown(f"""
                **Connected as**  
                {display_name}  
                `{username}`
                """)
            
            # Add a button to reload repositories
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("↻ Refresh Repositories"):
                    if 'repos' in st.session_state:
                        del st.session_state.repos
                        st.rerun()
            with col2:
                if st.button("Disconnect GitHub"):
                    if 'repos' in st.session_state:
                        del st.session_state.repos
                    del st.session_state.github_token
                    st.rerun()
            
            # Show loading state while fetching repositories
            if 'repos' not in st.session_state:
                with st.spinner('Loading your repositories...'):
                    try:
                        st.session_state.repos = get_user_repos(st.session_state.github_token)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load repositories: {str(e)}")
                        if st.button("Retry Loading Repositories"):
                            if 'repos' in st.session_state:
                                del st.session_state.repos
                            st.rerun()
            else:
                # Display repositories
                st.subheader("Your Repositories")
                st.caption("Click on a repository to analyze it")
                
                # Add search filter
                search_term = st.text_input("Search repositories...", "", 
                                         placeholder="Type to filter repositories")
                
                # Filter and display repositories
                filtered_repos = [
                    repo for repo in st.session_state.repos 
                    if search_term.lower() in repo['full_name'].lower()
                ]
                
                # Show repository list with pagination
                items_per_page = 10
                if 'page' not in st.session_state:
                    st.session_state.page = 0
                
                total_pages = (len(filtered_repos) - 1) // items_per_page + 1
                if total_pages > 1:
                    st.caption(f"Page {st.session_state.page + 1} of {total_pages}")
                
                # Pagination controls
                col1, col2, _ = st.columns([1, 1, 2])
                with col1:
                    if st.session_state.page > 0 and st.button("← Previous"):
                        st.session_state.page -= 1
                        st.rerun()
                with col2:
                    if (st.session_state.page + 1) * items_per_page < len(filtered_repos) and st.button("Next →"):
                        st.session_state.page += 1
                        st.rerun()
                
                # Display current page of repositories
                start_idx = st.session_state.page * items_per_page
                end_idx = min((st.session_state.page + 1) * items_per_page, len(filtered_repos))
                
                for repo in filtered_repos[start_idx:end_idx]:
                    with st.expander(f"{repo['name']} {'🔒' if repo['private'] else ''}"):
                        st.markdown(f"""
                        **Description:** {repo['description'] or 'No description'}
                        
                        **Language:** {repo['language'] or 'Not specified'}
                        
                        **Last Updated:** {repo['updated_at'].split('T')[0]}
                        
                        **Size:** {repo['size']} KB
                        
                        **URL:** [{repo['html_url']}]({repo['html_url']})
                        """)
                        
                        if st.button(f"Analyze {repo['name']}", key=f"analyze_{repo['id']}"):
                            st.session_state.repo_url = repo['html_url']
                            st.session_state.repo_loaded = False
                            st.session_state.analyze_from_github = True  # Flag to indicate analysis from GitHub tab
                            st.rerun()
                
        except Exception as e:
            st.error(f"Error fetching user info: {str(e)}")
            st.button("Retry Connection")
            return
    else:
        auth_url = get_github_auth_url()
        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="background-color: #24292e; color: white; border: none; 
                         padding: 10px 20px; text-align: center; 
                         text-decoration: none; display: inline-block; 
                         font-size: 16px; margin: 4px 2px; 
                         cursor: pointer; border-radius: 5px;">
                <i class="fab fa-github"></i> Sign in with GitHub
            </button>
        </a>
        """, unsafe_allow_html=True)

def render_repo_input():
    """Render repository URL input form"""
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
            value=st.session_state.get('repo_url', '')
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button(
                label="Analyze Repository", 
                use_container_width=True,
                disabled=st.session_state.get('repo_processing', False)
            )
        with col2:
            force_reload = st.checkbox("Force reload", value=False, 
                                     help="If checked, repository will be re-downloaded even if already processed")
    
    return repo_url, submit_button, force_reload

def handle_github_callback():
    """Handle GitHub OAuth callback"""
    # Check for OAuth code in URL
    if 'code' in st.query_params and 'github_token' not in st.session_state:
        code = st.query_params['code']
        try:
            # Exchange code for access token
            response = requests.post(
                GITHUB_TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    'client_id': GITHUB_CLIENT_ID,
                    'client_secret': GITHUB_CLIENT_SECRET,
                    'code': code,
                    'redirect_uri': REDIRECT_URI
                }
            )
            response.raise_for_status()
            token_data = response.json()
            
            # Store token and user info in session state
            if 'access_token' in token_data:
                st.session_state.github_token = token_data['access_token']
                
                # Get and store user info
                headers = {'Authorization': f'token {st.session_state.github_token}'}
                user_response = requests.get(GITHUB_USER_URL, headers=headers)
                user_response.raise_for_status()
                st.session_state.github_user = user_response.json()
                
                # Clear the code from URL and reload
                st.query_params.clear()
                st.rerun()
            
        except Exception as e:
            st.error(f"Error during GitHub authentication: {str(e)}")
            if 'github_token' in st.session_state:
                del st.session_state.github_token
            if 'github_user' in st.session_state:
                del st.session_state.github_user

def render_home():
    """
    Render the home page with repository input form and GitHub OAuth.
    """
    st.title("GitHub Repository Analyzer")
    
    # Handle GitHub OAuth callback
    handle_github_callback()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Enter Repository URL", "Connect to GitHub"])
    
    # Track active tab using URL parameters
    if 'tab' not in st.query_params:
        st.query_params['tab'] = 'repo'  # Default to first tab
    
    # Handle tab switching
    if tab1:
        st.query_params['tab'] = 'repo'
    if tab2:
        st.query_params['tab'] = 'github'
    
    with tab1:
        repo_url, submit_button, force_reload = render_repo_input()
    
    # Process repository when form is submitted or when analyzing from GitHub tab
    process_repo = False
    repo_to_process = None
    
    # Check if we're processing from the form
    if submit_button and st.query_params.get('tab') == 'repo':
        if not repo_url:
            st.error("Please enter a GitHub repository URL.")
            return
        
        if not is_valid_github_url(repo_url):
            st.error("Please enter a valid GitHub repository URL (e.g., https://github.com/username/repository).")
            return
        
        repo_to_process = repo_url
        process_repo = True
    # Check if we're processing from GitHub tab
    elif st.session_state.get('analyze_from_github'):
        repo_to_process = st.session_state.get('repo_url')
        process_repo = True
        # Clear the flag to prevent reprocessing on rerun
        st.session_state.analyze_from_github = False
    
    if process_repo and repo_to_process:
        # Store URL in session state
        st.session_state.repo_url = repo_to_process
        
        # Check if repository is already processed
        if not force_reload and st.session_state.get('repo_loaded') and extract_repo_name(repo_to_process) == st.session_state.get('repo_name'):
            st.success("Repository is already processed and ready for chat!")
            return
        
        # Reset processing status
        update_repo_status(fetch=False, parse=False, embed=False)
        st.session_state.repo_loaded = False
        set_repo_processing(True)
        
        # Show processing message
        with st.spinner(f"Processing repository: {repo_to_process}"):
            # Process repository
            if process_repository(repo_to_process):
                st.session_state.repo_loaded = True
                st.session_state.repo_name = extract_repo_name(repo_to_process)
                st.success("Repository processed successfully!")
                # Switch to the first tab to show the chat interface
                st.query_params['tab'] = 'repo'
                st.rerun()
            else:
                st.error("Failed to process repository. Please check the URL and try again.")
        
        set_repo_processing(False)
    
    with tab2:
        render_github_connect()
    
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