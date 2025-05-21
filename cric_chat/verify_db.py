"""
Script to verify the database setup for the Cricket Image Chatbot
"""

import psycopg2
import json
from typing import List, Tuple
from langchain.docstore.document import Document

import config
from vector_store import get_embeddings_model

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

def check_tables():
    """
    Check the tables in the database
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cursor.fetchall()

    print("Database tables:")
    for table in tables:
        print(f"- {table[0]}")

        # Get row count for each table
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")

    cursor.close()
    conn.close()

def check_reference_data():
    """
    Check the reference data in the database
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check players
    cursor.execute("SELECT player_id, player_name, team_code FROM players LIMIT 5")
    players = cursor.fetchall()

    print("\nPlayers (sample):")
    for player in players:
        print(f"  {player[0]}: {player[1]} ({player[2]})")

    # Check action
    cursor.execute("SELECT action_id, action_name FROM action LIMIT 5")
    actions = cursor.fetchall()

    print("\nActions (sample):")
    for action in actions:
        print(f"  {action[0]}: {action[1]}")

    # Check event
    cursor.execute("SELECT event_id, event_name FROM event LIMIT 5")
    events = cursor.fetchall()

    print("\nEvents (sample):")
    for event in events:
        print(f"  {event[0]}: {event[1]}")

    cursor.close()
    conn.close()

def check_cricket_data():
    """
    Check the cricket data in the database
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check cricket data
    cursor.execute("""
    SELECT c.id, c.file_name, c.url, p.player_name, e.event_name
    FROM cricket_data c
    LEFT JOIN players p ON c.player_id = p.player_id
    LEFT JOIN event e ON c.event_id = e.event_id
    LIMIT 5
    """)
    data = cursor.fetchall()

    print("\nCricket Data (sample):")
    for row in data:
        print(f"  ID: {row[0]}")
        print(f"  File: {row[1]}")
        print(f"  URL: {row[2]}")
        print(f"  Player: {row[3]}")
        print(f"  Event: {row[4]}")
        print()

    cursor.close()
    conn.close()

def check_documents():
    """
    Check the documents in the database
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check documents
    cursor.execute("SELECT id, content, metadata FROM documents LIMIT 3")
    documents = cursor.fetchall()

    print("\nDocuments (sample):")
    for doc in documents:
        print(f"  ID: {doc[0]}")
        print(f"  Content: {doc[1][:100]}...")

        # Parse metadata
        metadata = doc[2]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        print(f"  Metadata: {list(metadata.keys())}")
        print()

    cursor.close()
    conn.close()

def test_similarity_search():
    """
    Test the similarity search functionality
    """
    print("\nTesting similarity search...")

    # Get embeddings model
    embeddings_model = get_embeddings_model()

    # Generate a test query embedding
    test_query = "cricket player batting"
    print(f"Test query: '{test_query}'")

    query_embedding = embeddings_model.embed_query(test_query)

    # Perform similarity search
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Use pgvector for similarity search
        cursor.execute("""
        SELECT d.id, d.content, 1 - (e.embedding <=> %s) as similarity
        FROM embeddings e
        JOIN documents d ON e.document_id = d.id
        ORDER BY e.embedding <=> %s
        LIMIT 3
        """, (query_embedding, query_embedding))

        results = cursor.fetchall()

        print(f"Found {len(results)} results:")
        for doc_id, content, similarity in results:
            print(f"  Document {doc_id}: Similarity {similarity:.4f}")
            print(f"  Content: {content[:100]}...")
            print()
    except Exception as e:
        print(f"Error in similarity search: {e}")

    cursor.close()
    conn.close()

def main():
    """
    Main function
    """
    print(f"Verifying database '{config.DB_NAME}'...")

    try:
        # Check tables
        check_tables()

        # Check reference data
        check_reference_data()

        # Check cricket data
        check_cricket_data()

        # Check documents
        check_documents()

        # Test similarity search
        test_similarity_search()

        print("\nDatabase verification complete!")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    main()
