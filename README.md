# Cricket Image Chatbot

A Streamlit-based chatbot application that uses AI to analyze cricket images and provide insights.

## Deployment on Streamlit Cloud

### Optimized Project Structure

The project has been optimized for Streamlit Cloud deployment:

```
cricket-image-chatbot/
├── .streamlit/
│   └── secrets.toml       # Contains all secrets (don't commit this!)
├── data/                  # Contains all CSV data files
│   ├── Action.csv
│   ├── Event.csv
│   ├── finalTaggedData.csv
│   ├── Mood.csv
│   ├── Players.csv
│   └── Sublocation.csv
├── app.py                 # Main application
├── streamlit_app.py       # Entry point for Streamlit Cloud
├── pyproject.toml         # Python dependencies (using unversioned dependencies)
├── packages.txt           # System dependencies
├── setup.py               # Package setup
├── __init__.py            # Package initialization
└── ... (other application files)
```

### Deployment Steps

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Set the main file path to `streamlit_app.py`
4. Configure your secrets in Streamlit Cloud
5. Deploy your app

### Secrets Configuration

In Streamlit Cloud, add the following secrets:

```toml
# Database credentials
DB_NAME = "defaultdb"
DB_USER = "avnadmin"
DB_PASSWORD = "your_password"
DB_HOST = "your_host.aivencloud.com"
DB_PORT = "24832"

# API settings
GROQ_API_KEY = "your_groq_api_key"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Troubleshooting Deployment Issues

If you encounter issues during deployment:

1. Check the Streamlit Cloud logs for specific error messages
2. Verify that all secrets are correctly configured
3. Ensure your database is accessible from Streamlit Cloud
4. The streamlit_app.py includes debugging information that will be displayed if there are issues

## Local Development

1. Clone the repository
2. Create a `.streamlit/secrets.toml` file with your secrets
3. Install dependencies: `pip install -e .`
4. Run the app: `streamlit run streamlit_app.py`