"""
Flask API for the Cricket Image Chatbot
"""

from flask import Flask, request, jsonify
import re
import requests
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse
import datetime
import json
import os
from flask_cors import CORS

# Import from your existing modules
import config
from llm_service import query_images
from vector_store import get_or_create_vector_store
from langchain.docstore.document import Document
import db_store
from auth import save_user_query, get_user_queries
from db_store import is_player_query, get_player_names_in_query, get_document_id_from_url

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize vector store at app startup
vector_store = get_or_create_vector_store()

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and accessible

    Args:
        url (str): URL to check

    Returns:
        bool: True if the URL is valid and accessible, False otherwise
    """
    try:
        # Check if the URL has a valid format
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        # Try to make a HEAD request to check if the URL is accessible
        # Use a timeout to avoid hanging
        response = requests.head(url, timeout=2)
        return response.status_code < 400
    except Exception:
        return False

def convert_google_drive_url(url: str) -> str:
    """
    Convert a Google Drive sharing URL to a direct image URL

    Args:
        url (str): Google Drive sharing URL

    Returns:
        str: Direct image URL or original URL if not a Google Drive URL
    """
    # Check if it's a Google Drive URL
    if "drive.google.com/file/d/" in url:
        try:
            # Extract the file ID
            file_id = url.split("/d/")[1].split("/")[0]

            # Return the large thumbnail URL as it's more likely to work with restricted files
            return f"https://drive.google.com/thumbnail?id={file_id}&sz=w2000"
        except Exception:
            # If there's any error in extraction, return the original URL
            return url
    return url

def extract_urls_from_response(response: str) -> List[Tuple[str, str]]:
    """
    Extract image URLs and descriptions from the LLM response

    Args:
        response (str): The LLM response

    Returns:
        List[Tuple[str, str]]: List of (url, description) tuples or empty list if no URLs found
    """
    # Check if response contains any URLs
    if not response or "http" not in response:
        return []

    # Simple regex to extract URLs
    url_pattern = r'https?://[^\s)"]+'
    urls = re.findall(url_pattern, response)

    # If no URLs found, return empty list
    if not urls:
        return []

    # Extract descriptions (this is a simple approach, might need refinement)
    descriptions = []
    for url in urls:
        # Find the text around the URL
        url_index = response.find(url)

        # Look for the end of the sentence or paragraph
        end_index = response.find('\n', url_index)
        if end_index == -1:
            end_index = len(response)

        # Look for the beginning of the sentence or paragraph
        start_index = response.rfind('\n', 0, url_index)
        if start_index == -1:
            start_index = 0
        else:
            start_index += 1  # Skip the newline

        # Extract the description
        description = response[start_index:end_index].strip()
        description = description.replace(url, '').strip()
        descriptions.append(description)

    # Pair URLs with descriptions
    return list(zip(urls, descriptions))

def serialize_document(doc: Document, score: float) -> Dict[str, Any]:
    """
    Serialize a Document object to JSON for API responses
    
    Args:
        doc (Document): The document object
        score (float): Similarity score
        
    Returns:
        Dict[str, Any]: Serialized document
    """
    return {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    }

# ----- Flask Routes -----

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route('/api/query_images', methods=['POST'])
def api_query_images():
    """
    API endpoint to query images based on user input
    
    Expects JSON with:
    - query: str (required)
    - force_similarity: bool (optional)
    - user_id: int (optional)
    
    Returns:
    - response_text: str
    - similar_images: List[Dict]
    - used_similarity: bool
    """
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing required parameter 'query'"}), 400
        
    query_text = data['query']
    force_similarity = data.get('force_similarity', False)
    user_id = data.get('user_id')
    
    # Save the query if user ID is provided
    if user_id:
        save_user_query(user_id, query_text)
    
    try:
        # Query images using existing function
        response_text, similar_images, used_similarity = query_images(query_text, force_similarity)
        
        # Serialize documents for JSON response
        serialized_images = [serialize_document(doc, score) for doc, score in similar_images]
        
        return jsonify({
            "response_text": response_text,
            "similar_images": serialized_images,
            "used_similarity": used_similarity
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document_id', methods=['GET'])
def get_document_id():
    """
    Get document ID based on image URL
    
    Query parameters:
    - url: str (required)
    
    Returns:
    - document_id: int or null
    """
    url = request.args.get('url')
    
    if not url:
        return jsonify({"error": "Missing required parameter 'url'"}), 400
    
    try:
        doc_id = get_document_id_from_url(url)
        return jsonify({"document_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def store_user_feedback():
    """
    Store user feedback for an image
    
    Expects JSON with:
    - doc_id: int (required)
    - query: str (required)
    - image_url: str (required)
    - rating: int (required, 1 for positive, -1 for negative)
    - user_id: int (optional)
    
    Returns:
    - success: bool
    - message: str
    """
    data = request.json
    
    if not data or not all(key in data for key in ['doc_id', 'query', 'image_url', 'rating']):
        return jsonify({"error": "Missing required parameters"}), 400
    
    doc_id = data['doc_id']
    query = data['query']
    image_url = data['image_url']
    rating = data['rating']
    
    try:
        success = db_store.store_feedback(doc_id, query, image_url, rating)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Feedback stored successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to store feedback"
            }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user_queries', methods=['GET'])
def get_user_query_history():
    """
    Get user query history
    
    Query parameters:
    - user_id: int (required)
    
    Returns:
    - queries: List[Dict]
    """
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({"error": "Missing required parameter 'user_id'"}), 400
    
    try:
        # Get user's query history
        queries = get_user_queries(int(user_id))
        
        # Format for JSON response
        formatted_queries = [
            {
                "query": query,
                "timestamp": timestamp.isoformat()
            }
            for query, timestamp in queries
        ]
        
        return jsonify({"queries": formatted_queries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_query', methods=['POST'])
def save_query():
    """
    Save a user query
    
    Expects JSON with:
    - user_id: int (required)
    - query: str (required)
    
    Returns:
    - success: bool
    - message: str
    """
    data = request.json
    
    if not data or not all(key in data for key in ['user_id', 'query']):
        return jsonify({"error": "Missing required parameters"}), 400
    
    user_id = data['user_id']
    query = data['query']
    
    try:
        save_user_query(user_id, query)
        return jsonify({
            "success": True,
            "message": "Query saved successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/is_player_query', methods=['GET'])
def check_player_query():
    """
    Check if a query contains player names
    
    Query parameters:
    - query: str (required)
    
    Returns:
    - is_player_query: bool
    - player_names: List[str] (if applicable)
    """
    query_text = request.args.get('query')
    
    if not query_text:
        return jsonify({"error": "Missing required parameter 'query'"}), 400
    
    try:
        result = is_player_query(query_text.lower())
        
        response = {
            "is_player_query": result
        }
        
        # If it's a player query, get the player names
        if result:
            player_names = get_player_names_in_query(query_text.lower())
            response["player_names"] = player_names
            
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """
    User login endpoint
    
    Expects JSON with:
    - email: str (required)
    - password: str (required)
    
    Returns:
    - user: Dict or null
    - token: str (if successful)
    - message: str
    """
    data = request.json
    
    if not data or not all(key in data for key in ['email', 'password']):
        return jsonify({"error": "Missing email or password"}), 400
    
    email = data['email']
    password = data['password']
    
    try:
        # Use your existing authentication logic
        from login import authenticate_user
        user = authenticate_user(email, password)
        
        if user:
            # Generate a simple token (in production, use a proper JWT)
            token = f"token_{user['id']}_{hash(datetime.datetime.now().isoformat())}"
            
            return jsonify({
                "success": True,
                "user": user,
                "token": token,
                "message": "Login successful"
            })
        else:
            return jsonify({
                "success": False,
                "user": None,
                "message": "Invalid email or password"
            }), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure NLTK resources are available before starting
    try:
        import nltk
        
        # List of required resources
        resources = [
            'punkt',
            'wordnet',
            'omw-1.4',  # Open Multilingual WordNet
            'averaged_perceptron_tagger'
        ]

        for resource in resources:
            try:
                # Check if resource exists
                if resource == 'punkt':
                    nltk.data.find(f'tokenizers/{resource}')
                elif resource == 'wordnet' or resource == 'omw-1.4':
                    nltk.data.find(f'corpora/{resource}')
                else:
                    nltk.data.find(f'taggers/{resource}')
                print(f"NLTK resource '{resource}' is already available.")
            except LookupError:
                # Download if not found
                print(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource)
                print(f"Downloaded NLTK resource '{resource}'.")
    except ImportError:
        print("NLTK not available. Some natural language processing features may be limited.")
        
    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)