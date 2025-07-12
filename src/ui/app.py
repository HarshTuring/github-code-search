import streamlit as st
from src.ui.state import initialize_state
from src.ui.components.home import render_home
from src.ui.components.sidebar import render_sidebar

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
        # We'll implement this in the next step
        st.write("Chat interface will be displayed here")
    else:
        # Fallback to home page
        render_home()