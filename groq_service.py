"""
Groq API service module for the Cricket Image Chatbot
"""

import os
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

class GroqAPI:
    """
    Groq API client

    This is a wrapper around the Groq API for generating text responses.
    """

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY", config.GROQ_API_KEY)
        self.api_url = config.GROQ_API_URL
        self.model = config.GROQ_MODEL

        if not self.api_key:
            print("Warning: GROQ_API_KEY not available")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Groq API

        Args:
            prompt (str): The prompt to send to the API
            **kwargs: Additional parameters for the API

        Returns:
            str: The generated text
        """
        # For development/testing, we'll use a fallback if Groq API is not available
        if not self.api_key:
            return self._fallback_generate(prompt)

        # Set up the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Data format for Groq API (OpenAI-compatible)
        data = {
            "model": kwargs.get("model", self.model),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that helps users find cricket images."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 1.0),
        }

        try:
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30  # Add a timeout to avoid hanging
            )

            # Check if the request was successful
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Return the generated text
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""

        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        print("Using fallback response generation")

        # Extract the question from the prompt
        import re
        question_match = re.search(r'User Question: (.*?)(\n|$)', prompt)
        question = question_match.group(1) if question_match else "your query"

        # Generate a simple, concise response
        return f"""Here are some cricket images related to {question}. The most relevant matches are displayed below."""
