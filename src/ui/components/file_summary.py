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
    
    # Create enhanced prompt for LLM
    prompt = f"""
    You are a senior software engineer tasked with analyzing and explaining code files.
    
    Please analyze the following {language.upper()} file and provide a comprehensive summary that includes:

    1. MAIN PURPOSE: Explain the primary purpose and responsibility of this file within the codebase.
    
    2. KEY COMPONENTS:
       - Identify and explain important classes, functions, or methods
       - Highlight any interfaces, enums, or data structures defined
       - Describe the relationships between these components
    
    3. DEPENDENCIES:
       - List major external libraries or modules imported
       - Explain how these dependencies are used
       - Identify dependencies on other parts of the codebase
    
    4. ARCHITECTURAL PATTERNS:
       - Identify any design patterns or architectural approaches used
       - Explain how the code is structured and why
       - Note any notable algorithmic techniques
    
    5. POTENTIAL ISSUES OR IMPROVEMENTS:
       - Identify any potential code smells, performance concerns, or security issues
       - Suggest possible improvements (optional)

    Format your response in Markdown with clear headings and bullet points where appropriate.
    Focus on being thorough but concise, emphasizing the most important aspects of the file.
    
    FILE NAME: {os.path.basename(file_path)}
    FILE LANGUAGE: {language}
    {f"NOTE: This file was truncated due to size limitations. Analysis is based on the first {max_length} characters only." if is_truncated else ""}
    
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
        st.button("Generate File Summary", 
                key=f"gen_summary_{file_rel_path}", 
                on_click=lambda: _generate_and_cache_summary(file_rel_path, file_content),
                use_container_width=True)

def _generate_and_cache_summary(file_rel_path, file_content=None):
    """Generate and cache a file summary.
    
    Helper function to be used with on_click.
    
    Args:
        file_rel_path: Relative path to the file
        file_content: File content if already loaded
    """
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
        
    except Exception as e:
        # Store error message in session state
        st.session_state.file_summaries[file_rel_path] = f"Error generating summary: {str(e)}"