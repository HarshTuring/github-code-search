import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import os
import sys
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ui.components.file_chat import render_file_chat, generate_file_chat_response

class TestFileChat(unittest.TestCase):
    def setUp(self):
        # Reset Streamlit's session state before each test
        st.session_state.clear()
        
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, 'repo')
        os.makedirs(self.repo_path, exist_ok=True)
        
        # Set up test file path and content
        self.test_file = "test_file.py"
        self.test_content = "def hello_world():\n    return 'Hello, World!'"
        
        # Create a test file in the temporary directory
        self.test_file_path = os.path.join(self.repo_path, self.test_file)
        with open(self.test_file_path, 'w') as f:
            f.write(self.test_content)
        
        # Set up session state
        st.session_state.selected_file = self.test_file
        st.session_state.repo_path = self.repo_path
    
    def tearDown(self):
        # Clean up the temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Stop the patcher if it was started
        if hasattr(self, 'patcher'):
            self.patcher.stop()
    
    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    @patch('streamlit.empty')
    @patch('builtins.open', create=True)
    def test_render_file_chat_initial_state(self, mock_open, mock_empty, mock_chat_message, mock_chat_input):
        # Test initial rendering without user input
        mock_chat_input.return_value = None  # Simulate no user input
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = self.test_content
        mock_open.return_value = mock_file
        
        # Call the function
        render_file_chat()
        
        # Verify initial message is added to session state
        chat_key = f"file_chat_{self.test_file.replace('/', '_')}"
        self.assertIn(chat_key, st.session_state)
        self.assertEqual(len(st.session_state[chat_key]), 1)
        self.assertEqual(st.session_state[chat_key][0]["role"], "assistant")
        self.assertIn(f"Ask me anything about the file `{self.test_file}`", 
                     st.session_state[chat_key][0]["content"])

    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    @patch('streamlit.empty')
    @patch('src.ui.components.file_chat.generate_file_chat_response')
    @patch('src.ui.components.file_chat.determine_language')
    def test_render_file_chat_with_user_input(self, mock_determine_language, mock_generate, mock_empty, mock_chat_message, mock_chat_input):
        # Setup test data
        mock_determine_language.return_value = "python"
        test_prompt = "What does this file do?"
        test_response = "This file contains a function that returns 'Hello, World!'"
        
        # Mock chat input to return our test prompt
        mock_chat_input.return_value = test_prompt
        
        # Mock the response generation
        mock_generate.return_value = test_response
        
        # Create a mock for the empty() return value
        mock_message_placeholder = MagicMock()
        mock_empty.return_value = mock_message_placeholder
        
        # Mock file open
        with patch('builtins.open', unittest.mock.mock_open(read_data=self.test_content)) as mock_file:
            # Call the function
            render_file_chat()
        
        # Verify the message was added to the chat history
        chat_key = f"file_chat_{self.test_file.replace('/', '_')}"
        self.assertEqual(len(st.session_state[chat_key]), 3)  # Initial message + user + assistant
        self.assertEqual(st.session_state[chat_key][1]["role"], "user")
        self.assertEqual(st.session_state[chat_key][1]["content"], test_prompt)
        
        # Verify the response was added to the chat history with the exact content
        self.assertEqual(st.session_state[chat_key][2]["content"].strip(), test_response.strip())
        
        # Verify the response was generated with the correct arguments
        mock_generate.assert_called_once()
        
        # Get the actual call arguments
        args, _ = mock_generate.call_args
        
        # Check positional arguments
        self.assertEqual(args[0], test_prompt)  # prompt
        self.assertEqual(args[1], self.test_file)  # file_path
        self.assertEqual(args[2], self.test_content)  # file_content
        self.assertEqual(args[3], "python")  # language
        
        # Check conversation history
        self.assertEqual(len(args[4]), 3)  # Initial message + user prompt + assistant response
        self.assertEqual(args[4][0]["role"], "assistant")
        self.assertEqual(args[4][1]["role"], "user")
        self.assertEqual(args[4][1]["content"], test_prompt)
        self.assertEqual(args[4][2]["role"], "assistant")

    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    @patch('streamlit.empty')
    @patch('builtins.open', create=True)
    def test_chat_input_box_present(self, mock_open, mock_empty, mock_chat_message, mock_chat_input):
        """Test that the chat input box is present in the file chat."""
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = self.test_content
        mock_open.return_value = mock_file
        
        # Mock chat input to return None (no user input yet)
        mock_chat_input.return_value = None
        
        # Call the function
        render_file_chat()
        
        # Verify chat_input was called with the correct prompt and key
        expected_key = f"file_chat_input_{self.test_file.replace('/', '_')}"
        mock_chat_input.assert_called_once_with("Ask about this file...", key=expected_key)

if __name__ == '__main__':
    unittest.main()
