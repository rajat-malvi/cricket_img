"""
Authentication module for the Cricket Image Chatbot
"""

import re
import hashlib
import psycopg2
import streamlit as st
from typing import Tuple, Optional, Dict, Any

import config

def get_db_connection():
    """
    Get a connection to the PostgreSQL database
    """
    # Check if we're connecting to Aiven PostgreSQL (based on host)
    if 'aivencloud.com' in config.DB_HOST:
        # Use SSL for Aiven PostgreSQL
        return psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
    else:
        # Use standard connection for local PostgreSQL
        return psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT
        )

def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256

    Args:
        password: The password to hash

    Returns:
        The hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email: str) -> bool:
    """
    Check if an email is valid

    Args:
        email: The email to check

    Returns:
        True if the email is valid, False otherwise
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def is_valid_password(password: str) -> bool:
    """
    Check if a password is valid (at least 8 characters)

    Args:
        password: The password to check

    Returns:
        True if the password is valid, False otherwise
    """
    return len(password) >= 8

def register_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user

    Args:
        name: The user's name
        email: The user's email
        password: The user's password

    Returns:
        A tuple of (success, message)
    """
    # Validate inputs
    if not name:
        return False, "Name is required"

    if not is_valid_email(email):
        return False, "Invalid email format"

    if not is_valid_password(password):
        return False, "Password must be at least 8 characters long"

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False, "Email already registered"

        # Hash the password
        hashed_password = hash_password(password)

        # Insert the new user
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s) RETURNING id",
            (name, email, hashed_password)
        )

        user_id = cursor.fetchone()[0]
        conn.commit()

        cursor.close()
        conn.close()

        return True, f"User registered successfully with ID {user_id}"

    except Exception as e:
        return False, f"Error registering user: {str(e)}"

def login_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Login a user

    Args:
        email: The user's email
        password: The user's password

    Returns:
        A tuple of (success, message, user_data)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Hash the password
        hashed_password = hash_password(password)

        # Check if user exists with this email and password
        cursor.execute(
            "SELECT id, name, email FROM users WHERE email = %s AND password = %s",
            (email, hashed_password)
        )

        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            user_data = {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
            return True, "Login successful", user_data
        else:
            return False, "Invalid email or password", None

    except Exception as e:
        return False, f"Error logging in: {str(e)}", None

def save_user_query(user_id: int, query: str) -> bool:
    """
    Save a user query to the database

    Args:
        user_id: The user's ID
        query: The query text

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO user_queries (user_id, query) VALUES (%s, %s)",
            (user_id, query)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"Error saving user query: {str(e)}")
        return False

def get_user_queries(user_id: int, limit: int = 10) -> list:
    """
    Get a user's query history

    Args:
        user_id: The user's ID
        limit: Maximum number of queries to return

    Returns:
        A list of queries
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT query, timestamp FROM user_queries WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s",
            (user_id, limit)
        )

        queries = cursor.fetchall()
        cursor.close()
        conn.close()

        return queries

    except Exception as e:
        print(f"Error getting user queries: {str(e)}")
        return []

def initialize_auth_session_state():
    """Initialize authentication-related session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None

    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False


# def verify_login(email,password):
#     return True