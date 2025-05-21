"""
Streamlit entry point for the Cricket Image Chatbot
This file is used as the entry point for Streamlit Cloud deployment.
"""

import os
import sys
import streamlit as st
import nltk
from pathlib import Path

st.set_page_config(
        page_title="Cricket Image Chatbot",
        page_icon="🏏",
        layout="wide"
    )

# Set this to avoid potential memory issues with transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Create necessary directories
def setup_directories():
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist (should already exist in repo)
    data_dir = Path("data")
    os.makedirs(data_dir, exist_ok=True)

# Initialize the app
def initialize():
    
    
    # Download NLTK data
    download_nltk_data()
    
    # Setup directories
    setup_directories()
    
    # Check if database needs initialization
    try:
        import db_store
        if not db_store.database_exists():
            with st.spinner("Database not initialized. Initializing now..."):
                import init_db
                init_db.main()
                st.success("Database initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        st.info("Please check your database connection settings in Streamlit secrets.")
        if st.button("Retry Database Initialization"):
            try:
                import init_db
                init_db.main()
                st.success("Database initialized successfully!")
                st.experimental_rerun()
            except Exception as e2:
                st.error(f"Error during retry: {str(e2)}")

# Run the main app
def main():
    # Show a loading screen
    with st.spinner("Initializing application..."):
        initialize()
    
    # Import app.py and run it
    try:
        import app_full_stremlite
        app_full_stremlite.main()  
    except Exception as e:
        st.error(f"Error loading main application: {str(e)}")
        st.info("Try refreshing the page or check the application logs.")
        
        # Display system information for debugging
        with st.expander("System Information (for debugging)"):
            st.write("Python version:", sys.version)
            st.write("Current directory:", os.getcwd())
            
            # List installed packages
            st.write("Installed packages:")
            try:
                from importlib.metadata import distributions
            except ImportError:
                from importlib_metadata import distributions  # For Python < 3.8

            packages = sorted([f"{d.metadata['Name']}=={d.version}" for d in distributions()])
            st.write("Installed packages:", packages)

            
            # List files in data directory
            data_dir = Path("data")
            if data_dir.exists():
                st.write("Files in data directory:")
                st.write([f.name for f in data_dir.iterdir() if f.is_file()])
            else:
                st.warning("Data directory not found!")
    
if __name__ == "__main__":
    main()