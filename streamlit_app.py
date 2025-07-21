import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file as early as possible
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Loaded .env from: {env_path}")
else:
    # If .env doesn't exist in the same directory, try one level up (for development)
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(dotenv_path=parent_env, override=True)
        print(f"Loaded .env from: {parent_env}")
    else:
        print("Warning: No .env file found")

# Now that environment is loaded, import other modules
import streamlit as st
from src.ui.app import run_app

# Debug: Print environment variables for verification
print("Environment variables loaded:")
print(f"CLIENT_ID: {'*' * 8 + os.getenv('CLIENT_ID', 'Not found')[-4:] if os.getenv('CLIENT_ID') else 'Not found'}")
print(f"CLIENT_SECRET: {'*' * 8 + os.getenv('CLIENT_SECRET', 'Not found')[-4:] if os.getenv('CLIENT_SECRET') else 'Not found'}")

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