"""
Script to initialize the database for Render deployment with Aiven PostgreSQL
"""

import os
import sys
import time
import psycopg2
import nltk
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from init_aiven_db import setup_pgvector, create_tables

def download_nltk_resources():
    """Download required NLTK resources"""
    print("Downloading NLTK resources...")
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]

    for resource in resources:
        try:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)
            print(f"Downloaded NLTK resource '{resource}'.")
        except Exception as e:
            print(f"Error downloading NLTK resource '{resource}': {e}")
            print("Continuing anyway...")

def check_database_connection():
    """
    Check if we can connect to the Aiven PostgreSQL database
    """
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to connect to Aiven PostgreSQL...")
            conn = psycopg2.connect(
                dbname=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT,
                sslmode='require',
                connect_timeout=10
            )
            conn.close()
            print("Successfully connected to Aiven PostgreSQL database")
            return True
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    print("Failed to connect to Aiven PostgreSQL database after multiple attempts")
    return False

def create_users_table():
    """
    Create the users and user_queries tables if they don't exist
    """
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(64) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create user_queries table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_queries (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        cursor.close()
        conn.close()

        print("Users and user_queries tables created successfully")
    except Exception as e:
        print(f"Error creating users tables: {e}")
        raise

def main():
    """
    Main function to set up the database
    """
    print("Starting database setup for Render deployment with Aiven PostgreSQL...")

    # Download NLTK resources first
    download_nltk_resources()

    # Check database connection
    if not check_database_connection():
        print("Cannot proceed with database setup due to connection issues")
        print("Continuing anyway to ensure NLTK resources are downloaded...")
        return

    # Set up pgvector extension
    try:
        setup_pgvector()
    except Exception as e:
        print(f"Error setting up pgvector: {e}")
        print("Continuing with setup...")

    # Create tables
    try:
        create_tables()
    except Exception as e:
        print(f"Error creating tables: {e}")
        print("Continuing with setup...")

    # Create users tables
    try:
        create_users_table()
    except Exception as e:
        print(f"Error creating users tables: {e}")
        print("Continuing with setup...")

    print("Database setup for Render deployment with Aiven PostgreSQL complete!")

if __name__ == "__main__":
    main()
