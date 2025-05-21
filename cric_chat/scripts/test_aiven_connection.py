"""
Script to test the connection to Aiven PostgreSQL database
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Load environment variables
load_dotenv()

def test_connection():
    """
    Test the connection to Aiven PostgreSQL database
    """
    print("Testing connection to Aiven PostgreSQL database...")
    print(f"Host: {config.DB_HOST}")
    print(f"Port: {config.DB_PORT}")
    print(f"Database: {config.DB_NAME}")
    print(f"User: {config.DB_USER}")
    
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require',
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        
        # Get PostgreSQL version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"Successfully connected to Aiven PostgreSQL database")
        print(f"PostgreSQL version: {version}")
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        extension = cursor.fetchone()
        
        if extension:
            print("pgvector extension is installed")
        else:
            print("pgvector extension is NOT installed")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
