import streamlit as st
import os
import time
from src.llm.client import get_llm_client
from src.ui.components.code_viewer import determine_language

def render_file_chat():
    """Render chat interface for conversing about a specific file."""
    if "selected_file" not in st.session_state or not st.session_state.selected_file:
        return
    
    file_path = st.session_state.selected_file
    chat_key = f"file_chat_{file_path.replace('/', '_')}"
    
    # Initialize file chat messages if not exist for this file
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {"role": "assistant", "content": f"Ask me anything about the file `{os.path.basename(file_path)}`"}
        ]
    
    # Create a container for the chat header
    header = st.container()
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Create a container for the input at the bottom
    input_container = st.container()
    
    # Display header
    with header:
        st.markdown("---")
        st.markdown("## Chat with this File")
        st.caption("Ask questions specifically about this file's code and functionality.")
    
    # Display chat messages in the chat container
    with chat_container:
        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Handle user input in the input container at the bottom
    with input_container:
        if prompt := st.chat_input("Ask about this file...", key=f"file_chat_input_{file_path.replace('/', '_')}"):
            # Add the user's message to the chat
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            
            # Display the user's message in the chat container
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display the assistant's response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    try:
                        with st.spinner("Generating response..."):
                            # Load file content
                            file_abs_path = os.path.join(st.session_state.repo_path, file_path)
                            with open(file_abs_path, 'r', encoding='utf-8', errors='replace') as f:
                                file_content = f.read()
                            
                            # Determine language for better context
                            language = determine_language(file_path)
                            
                            # Generate a response about the specific file
                            response = generate_file_chat_response(prompt, file_path, file_content, language, st.session_state[chat_key])
                            
                            # Display the response with a typing effect
                            full_response = ""
                            for chunk in response.split():
                                full_response += chunk + " "
                                time.sleep(0.01)  # Fast typing effect
                                message_placeholder.markdown(full_response + "▌")
                            
                            message_placeholder.markdown(full_response)
                            
                            # Add the response to the chat history
                            st.session_state[chat_key].append({"role": "assistant", "content": full_response})
                            
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        message_placeholder.error(error_message)
                        st.session_state[chat_key].append({"role": "assistant", "content": error_message})

def generate_file_chat_response(prompt, file_path, file_content, language, conversation_history):
    """
    Generate a response to a question about a specific file.
    
    Args:
        prompt (str): The user's question
        file_path (str): The path to the file being discussed
        file_content (str): The content of the file
        language (str): The programming language of the file
        conversation_history (list): Previous messages in the conversation
        
    Returns:
        str: The generated response
    """
    # Truncate file content if too large
    max_content_length = 6000  # Adjust based on your LLM's context window
    is_truncated = False
    if len(file_content) > max_content_length:
        file_content = file_content[:max_content_length]
        is_truncated = True
    
    # Format conversation history for context
    formatted_history = ""
    # Skip the first assistant greeting and include only the last 6 messages for context
    for message in conversation_history[1:7] if len(conversation_history) > 1 else []:
        formatted_history += f"{message['role'].upper()}: {message['content']}\n\n"
    
    # Create system prompt that focuses the LLM on this specific file
    system_prompt = f"""
    You are a code assistant analyzing a specific file from a software repository. Your task is to focus exclusively on this file and provide insightful, accurate responses about it.
    
    FILE INFORMATION:
    - Filename: {os.path.basename(file_path)}
    - Full path: {file_path}
    - Language: {language.upper()}
    {f"- Note: File was truncated due to size limitations. Only the first {max_content_length} characters are shown." if is_truncated else ""}
    
    Guidelines:
    1. Analyze the file thoroughly before answering
    2. Provide specific references to line numbers and code sections when relevant
    3. Explain code patterns, functions, and logic in detail
    4. If you're uncertain about something, acknowledge it clearly
    5. Focus your answers only on this specific file
    6. When appropriate, suggest improvements to the code
    7. If asked about other files or the broader system, politely redirect focus to this file
    
    The file content is provided below:
    
    ```{language}
    {file_content}
    ```
    
    PREVIOUS CONVERSATION:
    {formatted_history}
    """
    
    # Get LLM client
    llm_client = get_llm_client()
    
    # Create the user message with the question
    user_message = f"USER QUESTION ABOUT THIS FILE: {prompt}"
    
    # Generate the response
    return llm_client.generate_with_system_prompt(system_prompt, user_message)