import streamlit as st
from src.llm.client import get_llm_client
import os

def generate_file_summary(file_path, file_content, language):
    """Generate a comprehensive summary of a file using the LLM.
    
    Args:
        file_path: Path to the file
        file_content: Content of the file
        language: Programming language of the file
        
    Returns:
        str: Summary of the file
    """
    # Check content length and truncate if necessary
    max_length = 8000  # Adjust based on your LLM's context limit
    is_truncated = False
    if len(file_content) > max_length:
        file_content = file_content[:max_length] 
        is_truncated = True
    
    # Enhanced prompt for detailed file analysis
    prompt = f"""
    # CODE ANALYSIS TASK

    As a senior software architect specializing in code analysis, please analyze this code file and provide a thorough explanation of its functionality, structure, and purpose.

    ## File Details
    - **Filename**: {os.path.basename(file_path)}
    - **Language**: {language.upper()}
    - **File Path**: {file_path}
    {f"- **NOTE**: This file was truncated due to size limitations. Analysis is based on the first {max_length} characters only." if is_truncated else ""}

    ## Analysis Requirements
    Please structure your analysis as follows:

    ### 1. Primary Purpose
    Provide a clear, concise explanation of what this file does and its role in the overall system.

    ### 2. Key Components
    Identify and explain:
    - Main classes/functions/methods and their purposes
    - Important data structures
    - Key algorithms or business logic
    - Any API endpoints or interfaces exposed

    ### 3. Dependencies and Relationships
    Analyze:
    - External libraries/frameworks used and why
    - Dependencies on other system components
    - How this file fits into the larger application architecture

    ### 4. Design Patterns and Architecture
    Identify:
    - Any design patterns implemented
    - Architectural principles followed
    - Code organization approaches

    ### 5. Technical Details
    Note any:
    - Interesting implementation details
    - Performance considerations
    - Error handling approaches
    - Security considerations (if apparent)

    Write your analysis in a professional but accessible style, using markdown formatting for readability.
    Focus on providing insights that would help a developer understand this file's role and functionality quickly.

    ## CODE TO ANALYZE
    ```{language}
    {file_content}
    ```
    """
    
    # Get LLM client and generate response
    llm_client = get_llm_client()
    response = llm_client.generate(prompt)
    
    return response.strip()

def render_file_summary(file_content=None):
    """Render the file summary component.
    
    Args:
        file_content: Content of the file (optional)
    """
    if "selected_file" not in st.session_state or not st.session_state.selected_file:
        return
        
    file_rel_path = st.session_state.selected_file
    
    # Initialize file summaries cache if not exists
    if "file_summaries" not in st.session_state:
        st.session_state.file_summaries = {}
    
    # Check if summary is in cache
    if file_rel_path in st.session_state.file_summaries:
        summary = st.session_state.file_summaries[file_rel_path]
        
        # Display with a divider for separation
        st.markdown("---")
        st.markdown("## File Summary")
        st.markdown(summary)
        
        # Add refresh button
        if st.button("Regenerate Summary", key=f"refresh_{file_rel_path}"):
            del st.session_state.file_summaries[file_rel_path]
            st.rerun()
    else:
        # Generate summary button
        st.markdown("---")
        if st.button("Generate File Summary", key=f"gen_summary_{file_rel_path}", use_container_width=True):
            with st.spinner("Generating file summary..."):
                try:
                    if file_content is None:
                        # Read content if not provided
                        file_abs_path = os.path.join(st.session_state.repo_path, file_rel_path)
                        with open(file_abs_path, 'r', encoding='utf-8', errors='replace') as f:
                            file_content = f.read()
                    
                    # Determine language
                    from src.ui.components.code_viewer import determine_language
                    language = determine_language(file_rel_path)
                    
                    # Generate summary
                    summary = generate_file_summary(file_rel_path, file_content, language)
                    
                    # Cache the summary
                    st.session_state.file_summaries[file_rel_path] = summary
                    
                    # Force UI refresh
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    # Cache the error message
                    st.session_state.file_summaries[file_rel_path] = f"⚠️ Error generating summary: {str(e)}"
                    st.rerun()