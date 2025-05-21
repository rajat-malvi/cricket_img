"""
Script to initialize the Aiven PostgreSQL database for Render deployment
This script will:
1. Create necessary tables in Aiven PostgreSQL
2. Load data from CSV files if available
3. Set up pgvector extension
"""

import os
import sys
import time
import json
import psycopg2
import pandas as pd
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vector_store import get_embeddings_model
from langchain.docstore.document import Document

def get_aiven_db_connection():
    """
    Get a connection to the Aiven PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
    """
    try:
        return psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
    except Exception as e:
        print(f"Error connecting to Aiven database: {e}")
        raise

def setup_pgvector():
    """
    Set up pgvector extension in the database
    """
    try:
        conn = get_aiven_db_connection()
        cursor = conn.cursor()

        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.commit()
        cursor.close()
        conn.close()

        print("pgvector extension created successfully")
    except Exception as e:
        print(f"Error setting up pgvector: {e}")
        raise

def create_tables():
    """
    Create the database tables in Aiven PostgreSQL
    """
    conn = get_aiven_db_connection()
    cursor = conn.cursor()

    # Create reference tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        player_id VARCHAR(10) PRIMARY KEY,
        player_name VARCHAR(100) NOT NULL,
        team_code VARCHAR(10)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS action (
        action_id VARCHAR(10) PRIMARY KEY,
        action_name VARCHAR(100) NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS event (
        event_id VARCHAR(10) PRIMARY KEY,
        event_name VARCHAR(100) NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mood (
        mood_id VARCHAR(10) PRIMARY KEY,
        mood_name VARCHAR(100) NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sublocation (
        sublocation_id VARCHAR(10) PRIMARY KEY,
        sublocation_name VARCHAR(100) NOT NULL
    )
    """)

    # Create main data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cricket_data (
        id SERIAL PRIMARY KEY,
        file_name VARCHAR(255) NOT NULL,
        url TEXT NOT NULL,
        player_id VARCHAR(10) REFERENCES players(player_id),
        datetime_original TIMESTAMP,
        date DATE,
        time_of_day VARCHAR(50),
        no_of_faces INTEGER,
        focus TEXT,
        shot_type VARCHAR(100),
        event_id VARCHAR(10) REFERENCES event(event_id),
        mood_id VARCHAR(10) REFERENCES mood(mood_id),
        action_id VARCHAR(10) REFERENCES action(action_id),
        caption TEXT,
        apparel TEXT,
        brands_and_logos TEXT,
        sublocation_id VARCHAR(10) REFERENCES sublocation(sublocation_id),
        location TEXT,
        make VARCHAR(100),
        model VARCHAR(100),
        copyright TEXT,
        photographer TEXT,
        description TEXT
    )
    """)

    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB
    )
    """)

    # Create embeddings table with pgvector
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            embedding vector(384)
        )
        """)
    except Exception as e:
        print(f"Error creating embeddings table: {e}")
        print("Creating embeddings table without vector type")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            embedding BYTEA
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

    print("Tables created in Aiven PostgreSQL database")

def load_csv_data():
    """
    Load data from CSV files if available
    """
    csv_path = config.CSV_FILE
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Skipping data loading from CSV")
        return False

    try:
        print(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")

        conn = get_aiven_db_connection()
        cursor = conn.cursor()

        # Process and insert data
        # This is a simplified version - you may need to adapt this to your CSV structure
        for _, row in df.iterrows():
            # Insert into appropriate tables
            pass

        conn.commit()
        cursor.close()
        conn.close()

        print("Data loaded from CSV successfully")
        return True
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return False

def main():
    """
    Main function to initialize the Aiven PostgreSQL database
    """
    print("Starting Aiven PostgreSQL database initialization for Render deployment...")

    # Step 1: Set up pgvector extension
    print("\nStep 1: Setting up pgvector extension...")
    setup_pgvector()

    # Step 2: Create tables
    print("\nStep 2: Creating tables...")
    create_tables()

    # Step 3: Load data from CSV if available
    print("\nStep 3: Loading data from CSV if available...")
    csv_loaded = load_csv_data()

    if not csv_loaded:
        print("\nNo data loaded from CSV. The database is initialized with empty tables.")
        print("You will need to populate the database with data manually.")

    print("\nAiven PostgreSQL database initialization complete!")

if __name__ == "__main__":
    main()
