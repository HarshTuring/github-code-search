import streamlit as st

def initialize_state():
    """
    Initialize the Streamlit session state with default values.
    This ensures all required state variables are available.
    """
    # General app state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    # Repository state
    if "repo_url" not in st.session_state:
        st.session_state.repo_url = ""
    if "repo_loaded" not in st.session_state:
        st.session_state.repo_loaded = False
    if "repo_processing" not in st.session_state:
        st.session_state.repo_processing = False
    if "repo_name" not in st.session_state:
        st.session_state.repo_name = ""
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = ""
    
    if "selected_repository" not in st.session_state:
        st.session_state.selected_repository = None
    
    # Processing state
    if "fetch_complete" not in st.session_state:
        st.session_state.fetch_complete = False
    if "parse_complete" not in st.session_state:
        st.session_state.parse_complete = False
    if "embed_complete" not in st.session_state:
        st.session_state.embed_complete = False
    
    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    # Repository explorer state
    if "exploring_repository" not in st.session_state:
        st.session_state.exploring_repository = False
    if "file_tree_expanded_dirs" not in st.session_state:
        st.session_state.file_tree_expanded_dirs = set([""])  # Root is expanded by default
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "file_summaries" not in st.session_state:
        st.session_state.file_summaries = {}  # Cache for generated summaries

    # Search state
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    if "search_results" not in st.session_state:
        st.session_state.search_results = {}
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "goto_line" not in st.session_state:
        st.session_state.goto_line = None

def navigate_to(page):
    """
    Change the current page.
    
    Args:
        page (str): The page to navigate to.
    """
    st.session_state.current_page = page
    st.rerun()
    st.session_state.current_page = page
    
def set_repo_processing(processing):
    """
    Set the repository processing state.
    
    Args:
        processing (bool): Whether the repository is being processed.
    """
    st.session_state.repo_processing = processing

def update_repo_status(fetch=None, parse=None, embed=None):
    """
    Update the status of repository processing steps.
    
    Args:
        fetch (bool, optional): Status of fetch step.
        parse (bool, optional): Status of parse step.
        embed (bool, optional): Status of embed step.
    """
    if fetch is not None:
        st.session_state.fetch_complete = fetch
    if parse is not None:
        st.session_state.parse_complete = parse
    if embed is not None:
        st.session_state.embed_complete = embed
    
    # If all steps are complete, mark repo as loaded
    if st.session_state.fetch_complete and st.session_state.parse_complete and st.session_state.embed_complete:
        st.session_state.repo_loaded = True
        st.session_state.repo_processing = False