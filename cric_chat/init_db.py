"""
Script to initialize the PostgreSQL database for the Cricket Image Chatbot
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import config
import db_store
from vector_store import get_embeddings_model

def create_database_if_not_exists():
    """
    Create the PostgreSQL database if it doesn't exist
    """
    # Check if we're using Aiven PostgreSQL
    if 'aivencloud.com' in config.DB_HOST:
        print(f"Using Aiven PostgreSQL database '{config.DB_NAME}'")
        print("Skipping database creation as Aiven databases are pre-created")
        return

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

def initialize_tables_and_data():
    """
    Initialize database tables and load data
    """
    # Create tables
    print("Creating database tables...")
    db_store.create_tables()

    # Load reference data from CSV files
    print("Loading reference data from CSV files...")
    db_store.load_all_reference_data()

    # Generate documents from database
    print("Generating documents from database...")
    documents = db_store.generate_documents_from_db()

    # Clear existing vector store data
    print("Clearing existing vector store data...")
    db_store.clear_database()

    # Generate embeddings and store in database
    print("Generating embeddings and storing in database...")
    embeddings_model = get_embeddings_model()
    texts = [doc.page_content for doc in documents]
    embeddings = embeddings_model.embed_documents(texts)
    db_store.insert_documents(documents, embeddings)

    print("Database initialization complete.")

def main():
    """
    Main function to initialize the database
    """
    print("Starting database initialization...")

    # Create database if it doesn't exist
    create_database_if_not_exists()

    # Initialize tables and data
    initialize_tables_and_data()

    print("Database setup complete!")

if __name__ == "__main__":
    main()
