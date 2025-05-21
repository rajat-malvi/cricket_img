"""
Flask API for the Cricket Image Chatbot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import requests
import datetime
from typing import List, Tuple
from urllib.parse import urlparse
import nltk
import os

# Import required modules
from llm_service import query_images
from vector_store import get_or_create_vector_store
from langchain.docstore.document import Document
import db_store
# from auth import initialize_auth_session_state, save_user_query, get_user_queries
from auth import save_user_query, get_user_queries, verify_login, register_user  # need to change verify_login

# Ensure all required NLTK resources are downloaded
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',
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

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Initialize vector store on startup
def initialize_vector_store():
    """Initialize vector store at app startup"""
    print("Initializing vector store...")
    get_or_create_vector_store()
    print("Vector store initialized")

@app.before_first_request
def before_first_request():
    """Run these tasks before the first request"""
    ensure_nltk_resources()
    initialize_vector_store()

# Helper functions
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

    # Extract descriptions
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

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required'}), 400
    
    user = verify_login(email, password)
    
    if user:
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email']
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

@app.route('/api/register', methods=['POST'])
def register():
    """Register endpoint"""
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Name, email, and password are required'}), 400
    
    success, message = register_user(name, email, password)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400

@app.route('/api/query', methods=['POST'])
def query():
    """Query endpoint for cricket images"""
    data = request.json
    prompt = data.get('prompt')
    user_id = data.get('user_id')
    force_similarity = data.get('force_similarity', False)
    
    if not prompt:
        return jsonify({'success': False, 'message': 'Prompt is required'}), 400
    
    # Save the user query if user_id is provided
    if user_id:
        save_user_query(user_id, prompt)
    
    # Generate response
    response_text, similar_images, used_similarity = query_images(prompt, force_similarity=force_similarity)
    
    # Convert similar images to serializable format
    serializable_images = []
    for doc, score in similar_images:
        # Check if metadata is available
        if hasattr(doc, 'metadata'):
            # Convert metadata to serializable format
            metadata = {k: str(v) if k != 'document_id' else v for k, v in doc.metadata.items() if k != 'embedding'}
            
            # Add the image URL and similarity score
            image_data = {
                'url': metadata.get('url') or metadata.get('image_url') or '',
                'similarity_score': 1.0 - score,  # Convert distance to similarity
                'metadata': metadata,
                'page_content': doc.page_content if hasattr(doc, 'page_content') else ''
            }
            serializable_images.append(image_data)
    
    # Extract URLs from the response text
    url_desc_pairs = extract_urls_from_response(response_text)
    
    return jsonify({
        'success': True,
        'response_text': response_text,
        'similar_images': serializable_images,
        'used_similarity': used_similarity,
        'extracted_urls': [{'url': url, 'description': desc} for url, desc in url_desc_pairs]
    })

@app.route('/api/user/queries', methods=['GET'])
def user_queries():
    """Get user's query history"""
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'success': False, 'message': 'User ID is required'}), 400
    
    queries = get_user_queries(user_id)
    
    # Convert queries to serializable format
    serializable_queries = [{'query': query, 'timestamp': timestamp.isoformat()} for query, timestamp in queries]
    
    return jsonify({
        'success': True,
        'queries': serializable_queries
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Store user feedback on image relevance"""
    data = request.json
    doc_id = data.get('doc_id')
    query = data.get('query')
    image_url = data.get('image_url')
    rating = data.get('rating')
    
    if not all([doc_id, query, image_url, rating is not None]):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    success = db_store.store_feedback(doc_id, query, image_url, rating)
    
    if success:
        return jsonify({'success': True, 'message': 'Feedback stored successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to store feedback'}), 500

@app.route('/api/document/id', methods=['GET'])
def get_document_id():
    """Get document ID from URL"""
    url = request.args.get('url')
    
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'}), 400
    
    doc_id = db_store.get_document_id_from_url(url)
    
    if doc_id:
        return jsonify({
            'success': True,
            'document_id': doc_id
        })
    else:
        return jsonify({'success': False, 'message': 'Document not found'}), 404

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)