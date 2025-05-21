"""
Configuration settings for the Cricket Image Chatbot
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# Data files
CSV_FILE = DATA_DIR / "finalTaggedData.csv"

# Function to get configuration from Streamlit secrets or environment variables
def get_config(key, default=None):
    """Get configuration from Streamlit secrets or environment variables"""
    # Try to get from Streamlit secrets first
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    # Then try environment variables
    return os.environ.get(key, default)

# Embedding model
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# PostgreSQL database settings
DB_NAME = get_config("DB_NAME", "jsk1_data")
DB_USER = get_config("DB_USER", "postgres")
DB_PASSWORD = get_config("DB_PASSWORD", "Skd6397@@")
DB_HOST = get_config("DB_HOST", "localhost")
DB_PORT = get_config("DB_PORT", "5432")

# LLaMA API settings (deprecated)
LLAMA_API_URL = get_config("LLAMA_API_URL", "https://api.llama-api.com")
LLAMA_API_KEY = get_config("LLAMA_API_KEY", "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")

# Groq API settings
GROQ_API_URL = get_config("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = get_config("GROQ_API_KEY", "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")
GROQ_MODEL = get_config("GROQ_MODEL", "llama-3.3-70b-versatile")

# Streamlit settings
STREAMLIT_TITLE = "Joburg Super Kings Image Finder"
STREAMLIT_DESCRIPTION = "Ask questions about Joburg Super Kings cricket images and get relevant image links."
