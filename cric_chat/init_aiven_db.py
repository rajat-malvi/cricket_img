"""
Initialize the Aiven PostgreSQL database for the Cricket Image Chatbot
"""

import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import config

def setup_pgvector():
    """
    Set up the pgvector extension in the Aiven PostgreSQL database
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Create the pgvector extension if it doesn't exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        print("pgvector extension created successfully")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up pgvector extension: {e}")
        raise

def create_tables():
    """
    Create the necessary tables in the Aiven PostgreSQL database
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
        cursor = conn.cursor()

        # Create players table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL
        )
        """)

        # Create action table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS action (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL
        )
        """)

        # Create event table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS event (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL
        )
        """)

        # Create mood table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL
        )
        """)

        # Create sublocation table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sublocation (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL
        )
        """)

        # Create cricket_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cricket_data (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            url TEXT,
            player_id INTEGER REFERENCES players(id),
            action_id INTEGER REFERENCES action(id),
            event_id INTEGER REFERENCES event(id),
            mood_id INTEGER REFERENCES mood(id),
            sublocation_id INTEGER REFERENCES sublocation(id),
            datetime_original TIMESTAMP,
            time_of_day VARCHAR(50),
            shot_type VARCHAR(50),
            focus VARCHAR(50),
            no_of_faces INTEGER,
            caption TEXT
        )
        """)

        # Create documents table for storing document content
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB
        )
        """)

        # Create embeddings table with vector support
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            embedding vector(768)
        )
        """)

        # Create feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            image_url TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        cursor.close()
        conn.close()

        print("Tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise

def main():
    """
    Main function to initialize the database
    """
    print("Initializing Aiven PostgreSQL database...")
    
    # Set up pgvector extension
    setup_pgvector()
    
    # Create tables
    create_tables()
    
    print("Aiven PostgreSQL database initialization complete!")

if __name__ == "__main__":
    main()
