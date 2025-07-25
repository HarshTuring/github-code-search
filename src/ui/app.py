import streamlit as st
from src.ui.state import initialize_state
from src.ui.components.home import render_home
from src.ui.components.sidebar import render_sidebar
from src.ui.components.chat import render_chat
from src.ui.components.repository_explorer import render_repository_explorer

def run_app():
    """
    Main function to run the Streamlit application.
    Manages the overall app flow and state transitions.
    """
    # Initialize session state if not already done
    initialize_state()
    
    # Render the sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.current_page == "home":
        render_home()
    elif st.session_state.current_page == "chat":
        render_chat()
    elif st.session_state.current_page == "explorer":  # New page
        render_repository_explorer()
    else:
        # Fallback to home page
        render_home()