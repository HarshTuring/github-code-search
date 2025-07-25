import streamlit as st
from src.ui.state import navigate_to

def render_sidebar():
    """
    Render the application sidebar with navigation and settings.
    """
    with st.sidebar:
        st.title("GitHub Repository Chat")
        
        # Application navigation
        st.header("Navigation")
        
        if st.button("Home", use_container_width=True):
            navigate_to("home")
            
        # Check if there are any repositories available
        from src.embeddings.vector_store import VectorStoreManager
        vector_store_manager = VectorStoreManager()
        repositories = vector_store_manager.list_repositories()
        has_repositories = len(repositories) > 0

        button_disabled = not (st.session_state.repo_loaded or has_repositories)

        
        # Enable chat button if a repo is loaded or if repositories exist
        chat_btn = st.button(
            "Chat with Repository", 
            disabled=not (st.session_state.repo_loaded or has_repositories),
            use_container_width=True
        )
        if chat_btn:
            navigate_to("chat")
        
        # Add button for repository explorer
        explorer_btn = st.button(
            "Repository Explorer",
            disabled=button_disabled,
            use_container_width=True
        )
        if explorer_btn:
            navigate_to("explorer")
        
        # Repository information
        if st.session_state.repo_loaded:
            st.header("Current Repository")
            st.write(f"**Name:** {st.session_state.repo_name}")
            
            # Status indicators
            st.subheader("Processing Status")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.fetch_complete:
                    st.success("Fetched", icon="✅")
                else:
                    st.info("Fetch", icon="⏳")
            with col2:
                if st.session_state.parse_complete:
                    st.success("Parsed", icon="✅")
                else:
                    st.info("Parse", icon="⏳")
            with col3:
                if st.session_state.embed_complete:
                    st.success("Embedded", icon="✅")
                else:
                    st.info("Embed", icon="⏳")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.caption("GitHub Repository Chat v1.0")