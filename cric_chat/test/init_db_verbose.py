"""
Verbose script to initialize the PostgreSQL database for the Cricket Image Chatbot
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time

import config
import db_store
from vector_store import get_embeddings_model

def create_database_if_not_exists():
    """
    Create the PostgreSQL database if it doesn't exist
    """
    print("Connecting to PostgreSQL server...")
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT
    )
    
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    print(f"Checking if database '{config.DB_NAME}' exists...")
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{config.DB_NAME}'")
    exists = cursor.fetchone()
    
    if not exists:
        print(f"Creating database '{config.DB_NAME}'...")
        cursor.execute(f"CREATE DATABASE {config.DB_NAME}")
        print(f"Database '{config.DB_NAME}' created successfully.")
    else:
        print(f"Database '{config.DB_NAME}' already exists.")
    
    cursor.close()
    conn.close()
    print("Database connection closed.")

def initialize_tables_and_data():
    """
    Initialize database tables and load data
    """
    # Create tables
    print("Creating database tables...")
    db_store.create_tables()
    print("Database tables created successfully.")
    
    # Load reference data from CSV files
    print("Loading reference data from CSV files...")
    try:
        db_store.load_all_reference_data()
        print("Reference data loaded successfully.")
    except Exception as e:
        print(f"Error loading reference data: {e}")
        raise
    
    # Generate documents from database
    print("Generating documents from database...")
    try:
        documents = db_store.generate_documents_from_db()
        print(f"Generated {len(documents)} documents.")
    except Exception as e:
        print(f"Error generating documents: {e}")
        raise
    
    # Clear existing vector store data
    print("Clearing existing vector store data...")
    try:
        db_store.clear_database()
        print("Vector store data cleared successfully.")
    except Exception as e:
        print(f"Error clearing vector store data: {e}")
        raise
    
    # Generate embeddings and store in database
    print("Generating embeddings and storing in database...")
    try:
        embeddings_model = get_embeddings_model()
        print("Embeddings model loaded successfully.")
        
        texts = [doc.page_content for doc in documents]
        print(f"Generating embeddings for {len(texts)} documents...")
        
        # Process in batches to avoid memory issues
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"Batch {i//batch_size + 1} processed successfully.")
            # Sleep briefly to avoid overwhelming the system
            time.sleep(1)
        
        print(f"Generated {len(all_embeddings)} embeddings.")
        
        print("Storing documents and embeddings in database...")
        db_store.insert_documents(documents, all_embeddings)
        print("Documents and embeddings stored successfully.")
    except Exception as e:
        print(f"Error generating and storing embeddings: {e}")
        raise
    
    print("Database initialization complete.")

def main():
    """
    Main function to initialize the database
    """
    print("Starting database initialization...")
    
    try:
        # Create database if it doesn't exist
        create_database_if_not_exists()
        
        # Initialize tables and data
        initialize_tables_and_data()
        
        print("Database setup complete!")
    except Exception as e:
        print(f"Error during database initialization: {e}")
        print("Database setup failed.")

if __name__ == "__main__":
    main()
