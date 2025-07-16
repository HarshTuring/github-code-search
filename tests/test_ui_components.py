"""Unit tests for Streamlit UI components."""
import pytest
import streamlit as st
import sys
from unittest.mock import MagicMock, patch, PropertyMock, ANY
from src.ui.state import initialize_state
from src.ui.components.home import render_home
from src.ui.components.chat import render_chat
from src.ui.components.sidebar import render_sidebar

# Create a custom dict that allows attribute access for session state
class SessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# Mock the GitHubCodeAnalyzer class
class MockGitHubCodeAnalyzer:
    def __init__(self):
        self.query_repository = MagicMock(return_value={
            'success': True,
            'response': 'Test response',
            'sources': [{'file': 'test.py', 'content': 'test content'}]
        })

# Create a mock instance that will be used as the return value
mock_analyzer_instance = MockGitHubCodeAnalyzer()

# Mock the Streamlit module for testing
@pytest.fixture
def mock_streamlit():
    # Create a session state that supports both dict and attribute access
    session_state = SessionState()
    
    # Create the mock for the module where GitHubCodeAnalyzer is imported
    mock_src_main = MagicMock()
    mock_src_main.GitHubCodeAnalyzer.return_value = mock_analyzer_instance
    
    with patch('streamlit.container'), \
         patch('streamlit.title'), \
         patch('streamlit.header'), \
         patch('streamlit.subheader'), \
         patch('streamlit.text'), \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.progress'), \
         patch('streamlit.error'), \
         patch('streamlit.success'), \
         patch('streamlit.session_state', session_state), \
         patch('streamlit.session_state', new_callable=PropertyMock(return_value=session_state)), \
         patch.dict('sys.modules', {'src.main': mock_src_main}):
        
        yield {
            'button': mock_button,
            'text_input': mock_text_input,
            'selectbox': mock_selectbox,
            'state': session_state,
            'mock_analyzer': mock_analyzer_instance
        }

def test_initialize_state(mock_streamlit):
    """Test that initialize_state sets up the session state correctly."""
    # Call the function
    initialize_state()
    
    # Check if session state is initialized with expected keys
    assert 'current_page' in st.session_state
    assert 'repo_loaded' in st.session_state
    assert 'fetch_complete' in st.session_state
    assert 'parse_complete' in st.session_state
    assert 'embed_complete' in st.session_state
    assert 'messages' in st.session_state

def test_render_home_initial_state(mock_streamlit):
    """Test the home page renders correctly in initial state."""
    # Set up initial state
    initialize_state()
    
    # Call the function
    render_home()
    
    # Verify UI elements are rendered
    mock_streamlit['text_input'].assert_called()  # Should have repo URL input
    mock_streamlit['button'].assert_called()      # Should have submit button

def test_render_sidebar(mock_streamlit):
    """Test the sidebar renders navigation buttons."""
    # Set up initial state
    initialize_state()
    
    # Mock the repository list to return at least one repo
    with patch('src.ui.components.chat.get_available_repositories') as mock_get_repos:
        mock_get_repos.return_value = [
            {'name': 'test/repo', 'collection_name': 'test_repo'}
        ]
        
        # Call the function
        render_sidebar()
        
        # Verify the buttons are rendered with correct disabled state
        mock_streamlit['button'].assert_any_call(
            "Home",
            use_container_width=True
        )
        mock_streamlit['button'].assert_any_call(
            "Chat with Repository",
            use_container_width=True,
            disabled=False  # Should be enabled when repos are available
        )