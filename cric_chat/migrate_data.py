"""
Script to migrate data to the new PostgreSQL database for the Cricket Image Chatbot
"""

import os
import time
import psycopg2
import pandas as pd
from typing import List

import config
import db_store
from vector_store import get_embeddings_model
from langchain.docstore.document import Document

def migrate_data():
    """
    Migrate data to the new database
    """
    print(f"Starting data migration to '{config.DB_NAME}'...")
    
    # Step 1: Create tables
    print("Creating database tables...")
    db_store.create_tables()
    print("Database tables created successfully.")
    
    # Step 2: Load reference data
    print("Loading reference data from CSV files...")
    try:
        db_store.load_all_reference_data()
        print("Reference data loaded successfully.")
    except Exception as e:
        print(f"Error loading reference data: {e}")
        raise
    
    # Step 3: Generate documents from database
    print("Generating documents from database...")
    try:
        documents = db_store.generate_documents_from_db()
        print(f"Generated {len(documents)} documents.")
    except Exception as e:
        print(f"Error generating documents: {e}")
        raise
    
    # Step 4: Generate embeddings and store in database
    print("Generating embeddings and storing in database...")
    try:
        # Get embeddings model
        embeddings_model = get_embeddings_model()
        
        # Generate embeddings for documents
        texts = [doc.page_content for doc in documents]
        print(f"Generating embeddings for {len(texts)} documents...")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            batch_embeddings = embeddings_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        # Store documents and embeddings in database
        print("Storing documents and embeddings in database...")
        db_store.insert_documents(documents, all_embeddings)
        print("Documents and embeddings stored successfully.")
    except Exception as e:
        print(f"Error generating and storing embeddings: {e}")
        raise
    
    print("Data migration complete!")

def check_migration():
    """
    Check that the migration was successful
    """
    print("Checking migration results...")
    
    # Connect to the database
    conn = db_store.get_db_connection()
    cursor = conn.cursor()
    
    # Check tables and row counts
    tables = [
        "players",
        "action",
        "event",
        "mood",
        "sublocation",
        "cricket_data",
        "documents",
        "embeddings"
    ]
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' has {count} rows.")
    
    # Close connection
    cursor.close()
    conn.close()
    
    print("Migration check complete.")

def migrate_local_to_aiven():
    """
    Migrates data (cricket_data, documents, embeddings) from a local PostgreSQL database (using fallback credentials) into the Aiven PostgreSQL database (using credentials from config).
    """
    import psycopg2
    import config
    import json
    import sys

    # Fallback (local) DB credentials
    local_db_name = "jsk1_data"
    local_db_user = "postgres"
    local_db_password = "Skd6397@@"
    local_db_host = "localhost"
    local_db_port = "5432"

    # Aiven DB credentials (from config)
    aiven_db_name = config.DB_NAME
    aiven_db_user = config.DB_USER
    aiven_db_password = config.DB_PASSWORD
    aiven_db_host = config.DB_HOST
    aiven_db_port = config.DB_PORT

    try:
        # Connect to local DB (using fallback credentials)
        conn_local = psycopg2.connect(dbname=local_db_name, user=local_db_user, password=local_db_password, host=local_db_host, port=local_db_port)
        cur_local = conn_local.cursor()

        # Connect to Aiven DB (using credentials from config)
        conn_aiven = psycopg2.connect(dbname=aiven_db_name, user=aiven_db_user, password=aiven_db_password, host=aiven_db_host, port=aiven_db_port, sslmode="require")
        cur_aiven = conn_aiven.cursor()

        # Migrate cricket_data table (assumed to have the same schema)
        cur_local.execute("SELECT * FROM cricket_data;")
        cricket_rows = cur_local.fetchall()
        if cricket_rows:
            cur_aiven.execute("TRUNCATE cricket_data;")
            cur_aiven.executemany("INSERT INTO cricket_data (id, file_name, url, player_id, datetime_original, date, time_of_day, no_of_faces, focus, shot_type, event_id, mood_id, action_id, caption, apparel, brands_and_logos, sublocation_id, location, make, model, copyright, photographer, description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", cricket_rows)
            print("Migrated cricket_data table.")

        # Migrate documents table (assumed to have columns (id, content, metadata))
        cur_local.execute("SELECT id, content, metadata FROM documents;")
        doc_rows = cur_local.fetchall()
        if doc_rows:
            cur_aiven.execute("TRUNCATE documents;")
            cur_aiven.executemany("INSERT INTO documents (id, content, metadata) VALUES (%s, %s, %s);", doc_rows)
            print("Migrated documents table.")

        # Migrate embeddings table (assumed to have columns (id, embedding, query_text))
        cur_local.execute("SELECT id, embedding, query_text FROM embeddings;")
        emb_rows = cur_local.fetchall()
        if emb_rows:
            cur_aiven.execute("TRUNCATE embeddings;")
            cur_aiven.executemany("INSERT INTO embeddings (id, embedding, query_text) VALUES (%s, %s, %s);", emb_rows)
            print("Migrated embeddings table.")

        conn_aiven.commit()
        print("Migration completed successfully.")
    except Exception as e:
        print("Error migrating data:", e, file=sys.stderr)
        if conn_aiven:
            conn_aiven.rollback()
    finally:
        if cur_local:
            cur_local.close()
        if conn_local:
            conn_local.close()
        if cur_aiven:
            cur_aiven.close()
        if conn_aiven:
            conn_aiven.close()

def main():
    """
    Main function
    """
    try:
        # Migrate data
        migrate_data()
        
        # Check migration
        check_migration()
        
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Error during migration: {e}")
        print("Migration failed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate data (from local DB to Aiven DB).")
    parser.add_argument("--migrate", action="store_true", help="Migrate local data into Aiven DB.")
    args = parser.parse_args()
    if args.migrate:
        migrate_local_to_aiven()
    else:
        migrate_data()
