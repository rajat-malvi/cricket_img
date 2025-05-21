"""
Script to create a new PostgreSQL database for the Cricket Image Chatbot
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import config

def create_database():
    """
    Create a new PostgreSQL database
    """
    print(f"Creating new database '{config.DB_NAME}'...")
    
    # Connect to PostgreSQL server (not to a specific database)
    conn = psycopg2.connect(
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
        # Connect to 'postgres' database to create a new database
        dbname='postgres'
    )
    
    # Set isolation level to AUTOCOMMIT
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{config.DB_NAME}'")
    exists = cursor.fetchone()
    
    if not exists:
        # Create the database
        cursor.execute(f"CREATE DATABASE {config.DB_NAME}")
        print(f"Database '{config.DB_NAME}' created successfully.")
    else:
        print(f"Database '{config.DB_NAME}' already exists.")
    
    # Close connection
    cursor.close()
    conn.close()
    
    # Connect to the new database to create pgvector extension
    conn = psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )
    
    cursor = conn.cursor()
    
    # Create pgvector extension
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("pgvector extension created successfully.")
    except Exception as e:
        print(f"Error creating pgvector extension: {e}")
        print("Vector similarity search may not work properly.")
    
    # Close connection
    cursor.close()
    conn.close()
    
    print("Database setup complete.")

if __name__ == "__main__":
    create_database()
