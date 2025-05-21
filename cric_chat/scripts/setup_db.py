"""
Script to initialize the database for deployment
"""

import os
import sys
import time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from init_db import create_database_if_not_exists, initialize_tables_and_data

def wait_for_db(max_attempts=30, delay=2):
    """
    Wait for the database to be available
    
    Args:
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print(f"Waiting for database at {config.DB_HOST}:{config.DB_PORT}...")
    
    for attempt in range(max_attempts):
        try:
            # Try to connect to PostgreSQL server
            conn = psycopg2.connect(
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT
            )
            conn.close()
            print("Database is available!")
            return True
        except psycopg2.OperationalError as e:
            print(f"Attempt {attempt+1}/{max_attempts}: Database not available yet. Error: {e}")
            time.sleep(delay)
    
    print("Failed to connect to database after maximum attempts")
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
            port=config.DB_PORT
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

def main():
    """
    Main function to set up the database
    """
    print("Starting database setup...")
    
    # Wait for the database to be available
    if not wait_for_db():
        print("Database not available. Exiting.")
        sys.exit(1)
    
    # Create database if it doesn't exist
    create_database_if_not_exists()
    
    # Initialize tables and data
    initialize_tables_and_data()
    
    # Create users tables
    create_users_table()
    
    print("Database setup complete!")

if __name__ == "__main__":
    main()
