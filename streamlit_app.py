import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from src.ui.app import run_app

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # If .env doesn't exist in the same directory, try one level up (for development)
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env)

# Configure the Streamlit page
st.set_page_config(
    page_title="GitHub Repository Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the main application
if __name__ == "__main__":
    run_app()