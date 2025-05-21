"""
Script to migrate data from local PostgreSQL database to Aiven PostgreSQL database
"""

import os
import sys
import time
import json
import psycopg2
import pandas as pd
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vector_store import get_embeddings_model
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Local database connection parameters
LOCAL_DB_NAME = "jsk1_data"
LOCAL_DB_USER = "postgres"
LOCAL_DB_PASSWORD = "Skd6397@@"
LOCAL_DB_HOST = "localhost"
LOCAL_DB_PORT = "5432"

# Aiven database connection parameters
AIVEN_DB_NAME = os.environ.get("DB_NAME", config.DB_NAME)
AIVEN_DB_USER = os.environ.get("DB_USER", config.DB_USER)
AIVEN_DB_PASSWORD = os.environ.get("DB_PASSWORD", config.DB_PASSWORD)
AIVEN_DB_HOST = os.environ.get("DB_HOST", config.DB_HOST)
AIVEN_DB_PORT = os.environ.get("DB_PORT", config.DB_PORT)

def get_local_db_connection():
    """
    Get a connection to the local PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
    """
    try:
        return psycopg2.connect(
            dbname=LOCAL_DB_NAME,
            user=LOCAL_DB_USER,
            password=LOCAL_DB_PASSWORD,
            host=LOCAL_DB_HOST,
            port=LOCAL_DB_PORT
        )
    except Exception as e:
        print(f"Error connecting to local database: {e}")
        raise

def get_aiven_db_connection():
    """
    Get a connection to the Aiven PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
    """
    try:
        return psycopg2.connect(
            dbname=AIVEN_DB_NAME,
            user=AIVEN_DB_USER,
            password=AIVEN_DB_PASSWORD,
            host=AIVEN_DB_HOST,
            port=AIVEN_DB_PORT,
            sslmode='require'
        )
    except Exception as e:
        print(f"Error connecting to Aiven database: {e}")
        raise

def create_tables_in_aiven():
    """
    Create the database tables in Aiven PostgreSQL
    """
    conn = get_aiven_db_connection()
    cursor = conn.cursor()

    # Create pgvector extension if it doesn't exist
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("pgvector extension created or already exists")
    except Exception as e:
        print(f"Warning: Could not create pgvector extension: {e}")
        print("Vector similarity search may not work properly")

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

def migrate_reference_data():
    """
    Migrate reference data from local PostgreSQL to Aiven PostgreSQL
    """
    try:
        local_conn = get_local_db_connection()
        aiven_conn = get_aiven_db_connection()

        local_cursor = local_conn.cursor()
        aiven_cursor = aiven_conn.cursor()

        # Migrate players
        print("Migrating players data...")
        local_cursor.execute("SELECT player_id, player_name, team_code FROM players")
        players = local_cursor.fetchall()

        for player in players:
            aiven_cursor.execute(
                "INSERT INTO players (player_id, player_name, team_code) VALUES (%s, %s, %s) ON CONFLICT (player_id) DO UPDATE SET player_name = EXCLUDED.player_name, team_code = EXCLUDED.team_code",
                player
            )

        # Migrate action
        print("Migrating action data...")
        local_cursor.execute("SELECT action_id, action_name FROM action")
        actions = local_cursor.fetchall()

        for action in actions:
            aiven_cursor.execute(
                "INSERT INTO action (action_id, action_name) VALUES (%s, %s) ON CONFLICT (action_id) DO UPDATE SET action_name = EXCLUDED.action_name",
                action
            )

        # Migrate event
        print("Migrating event data...")
        local_cursor.execute("SELECT event_id, event_name FROM event")
        events = local_cursor.fetchall()

        for event in events:
            aiven_cursor.execute(
                "INSERT INTO event (event_id, event_name) VALUES (%s, %s) ON CONFLICT (event_id) DO UPDATE SET event_name = EXCLUDED.event_name",
                event
            )

        # Migrate mood
        print("Migrating mood data...")
        local_cursor.execute("SELECT mood_id, mood_name FROM mood")
        moods = local_cursor.fetchall()

        for mood in moods:
            aiven_cursor.execute(
                "INSERT INTO mood (mood_id, mood_name) VALUES (%s, %s) ON CONFLICT (mood_id) DO UPDATE SET mood_name = EXCLUDED.mood_name",
                mood
            )

        # Migrate sublocation
        print("Migrating sublocation data...")
        local_cursor.execute("SELECT sublocation_id, sublocation_name FROM sublocation")
        sublocations = local_cursor.fetchall()

        for sublocation in sublocations:
            aiven_cursor.execute(
                "INSERT INTO sublocation (sublocation_id, sublocation_name) VALUES (%s, %s) ON CONFLICT (sublocation_id) DO UPDATE SET sublocation_name = EXCLUDED.sublocation_name",
                sublocation
            )

        aiven_conn.commit()
        local_cursor.close()
        aiven_cursor.close()
        local_conn.close()
        aiven_conn.close()

        print("Reference data migrated successfully")
    except Exception as e:
        print(f"Error migrating reference data: {e}")
        raise

def migrate_cricket_data():
    """
    Migrate cricket data from local PostgreSQL to Aiven PostgreSQL
    """
    try:
        local_conn = get_local_db_connection()
        aiven_conn = get_aiven_db_connection()

        local_cursor = local_conn.cursor()
        aiven_cursor = aiven_conn.cursor()

        print("Migrating cricket data...")
        local_cursor.execute("""
        SELECT
            file_name, url, player_id, datetime_original, date, time_of_day, no_of_faces,
            focus, shot_type, event_id, mood_id, action_id, caption, apparel,
            brands_and_logos, sublocation_id, location, make, model, copyright, photographer, description
        FROM cricket_data
        """)

        cricket_data = local_cursor.fetchall()

        for row in cricket_data:
            aiven_cursor.execute("""
            INSERT INTO cricket_data (
                file_name, url, player_id, datetime_original, date, time_of_day, no_of_faces,
                focus, shot_type, event_id, mood_id, action_id, caption, apparel,
                brands_and_logos, sublocation_id, location, make, model, copyright, photographer, description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, row)

        aiven_conn.commit()
        local_cursor.close()
        aiven_cursor.close()
        local_conn.close()
        aiven_conn.close()

        print(f"Cricket data migrated successfully ({len(cricket_data)} rows)")
    except Exception as e:
        print(f"Error migrating cricket data: {e}")
        raise

def migrate_users_data():
    """
    Migrate users and user_queries data from local PostgreSQL to Aiven PostgreSQL
    """
    try:
        local_conn = get_local_db_connection()
        aiven_conn = get_aiven_db_connection()

        local_cursor = local_conn.cursor()
        aiven_cursor = aiven_conn.cursor()

        # Check if users table exists in local database
        local_cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'users'
        )
        """)
        users_table_exists = local_cursor.fetchone()[0]

        if users_table_exists:
            print("Migrating users data...")
            local_cursor.execute("SELECT id, name, email, password, created_at FROM users")
            users = local_cursor.fetchall()

            for user in users:
                aiven_cursor.execute("""
                INSERT INTO users (id, name, email, password, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                    name = EXCLUDED.name,
                    password = EXCLUDED.password,
                    created_at = EXCLUDED.created_at
                """, user)

            print(f"Users data migrated successfully ({len(users)} rows)")

            # Migrate user_queries
            print("Migrating user_queries data...")
            local_cursor.execute("SELECT user_id, query, timestamp FROM user_queries")
            user_queries = local_cursor.fetchall()

            for query in user_queries:
                aiven_cursor.execute("""
                INSERT INTO user_queries (user_id, query, timestamp)
                VALUES (%s, %s, %s)
                """, query)

            print(f"User queries data migrated successfully ({len(user_queries)} rows)")
        else:
            print("Users table does not exist in local database. Skipping users data migration.")

        aiven_conn.commit()
        local_cursor.close()
        aiven_cursor.close()
        local_conn.close()
        aiven_conn.close()
    except Exception as e:
        print(f"Error migrating users data: {e}")

def generate_and_store_embeddings():
    """
    Generate documents from cricket data and store with embeddings in Aiven PostgreSQL
    """
    try:
        aiven_conn = get_aiven_db_connection()
        aiven_cursor = aiven_conn.cursor()

        print("Generating documents from cricket data...")

        # Join cricket_data with reference tables to get names instead of IDs
        aiven_cursor.execute("""
        SELECT
            c.id, c.file_name, c.url,
            p.player_name, p.team_code,
            c.datetime_original, c.date, c.time_of_day, c.no_of_faces, c.focus, c.shot_type,
            e.event_name,
            m.mood_name,
            a.action_name,
            c.caption, c.apparel, c.brands_and_logos,
            s.sublocation_name,
            c.location, c.make, c.model, c.copyright, c.photographer
        FROM cricket_data c
        LEFT JOIN players p ON c.player_id = p.player_id
        LEFT JOIN event e ON c.event_id = e.event_id
        LEFT JOIN mood m ON c.mood_id = m.mood_id
        LEFT JOIN action a ON c.action_id = a.action_id
        LEFT JOIN sublocation s ON c.sublocation_id = s.sublocation_id
        """)

        rows = aiven_cursor.fetchall()
        documents = []

        for row in rows:
            # Create metadata dictionary
            metadata = {
                "id": row[0],
                "file_name": row[1],
                "url": row[2],
                "player_name": row[3],
                "team_code": row[4],
                "datetime_original": str(row[5]) if row[5] else None,
                "date": str(row[6]) if row[6] else None,
                "time_of_day": row[7],
                "no_of_faces": row[8],
                "focus": row[9],
                "shot_type": row[10],
                "event_name": row[11],
                "mood_name": row[12],
                "action_name": row[13],
                "caption": row[14],
                "apparel": row[15],
                "brands_and_logos": row[16],
                "sublocation_name": row[17],
                "location": row[18],
                "make": row[19],
                "model": row[20],
                "copyright": row[21],
                "photographer": row[22]
            }

            # Create a concise description in the format shown in the example
            content = f"{row[14] or 'Cricket image'} Action: {row[13] or 'Unknown'}. Event: {row[11] or 'Unknown'}. Mood: {row[12] or 'Unknown'}. Location: {row[17] or 'Unknown'}. Time of day: {row[7] or 'Unknown'}. Focus: {row[9] or 'Unknown'}. Shot type: {row[10] or 'Unknown'}. Apparel: {row[15] or 'Unknown'}. Brands and logos: {row[16] or 'None'}. Number of faces: {row[8] or '0'}"

            # Create document
            doc = Document(page_content=content.strip(), metadata=metadata)
            documents.append(doc)

        print(f"Generated {len(documents)} documents")

        # Clear existing documents and embeddings
        print("Clearing existing documents and embeddings...")
        aiven_cursor.execute("DELETE FROM embeddings")
        aiven_cursor.execute("DELETE FROM documents")
        aiven_conn.commit()

        # Generate embeddings
        print("Generating embeddings...")
        embeddings_model = get_embeddings_model()
        texts = [doc.page_content for doc in documents]

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

        # Store documents and embeddings
        print("Storing documents and embeddings...")
        for i, doc in enumerate(documents):
            # Insert document
            aiven_cursor.execute(
                "INSERT INTO documents (content, metadata) VALUES (%s, %s) RETURNING id",
                (doc.page_content, json.dumps(doc.metadata))
            )
            doc_id = aiven_cursor.fetchone()[0]

            # Update metadata with document ID
            doc.metadata["document_id"] = doc_id

            # Insert embedding - convert to PostgreSQL vector format
            vector_str = f"[{','.join(str(x) for x in all_embeddings[i])}]"
            try:
                aiven_cursor.execute(
                    "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s::vector)",
                    (doc_id, vector_str)
                )
            except Exception as e:
                print(f"Error inserting embedding: {e}")
                # Try alternative method with explicit array syntax
                try:
                    aiven_cursor.execute(
                        "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s::vector)",
                        (doc_id, f"[{','.join(str(float(x)) for x in all_embeddings[i])}]")
                    )
                except Exception as e2:
                    print(f"Error inserting embedding (alternative method): {e2}")
                    # Last resort: store as bytea
                    import pickle
                    try:
                        aiven_cursor.execute(
                            "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s)",
                            (doc_id, pickle.dumps(all_embeddings[i]))
                        )
                    except Exception as e3:
                        print(f"Error inserting embedding (bytea method): {e3}")

        aiven_conn.commit()
        aiven_cursor.close()
        aiven_conn.close()

        print("Documents and embeddings stored successfully")
    except Exception as e:
        print(f"Error generating and storing embeddings: {e}")
        raise

def main():
    """
    Main function to migrate data from local PostgreSQL to Aiven PostgreSQL
    """
    print("Starting migration from local PostgreSQL to Aiven PostgreSQL...")

    try:
        # Step 1: Create tables in Aiven PostgreSQL
        print("\nStep 1: Creating tables in Aiven PostgreSQL...")
        create_tables_in_aiven()

        # Step 2: Migrate reference data
        print("\nStep 2: Migrating reference data...")
        migrate_reference_data()

        # Step 3: Migrate cricket data
        print("\nStep 3: Migrating cricket data...")
        migrate_cricket_data()

        # Step 4: Migrate users data
        print("\nStep 4: Migrating users data...")
        migrate_users_data()

        # Step 5: Generate and store embeddings
        print("\nStep 5: Generating and storing embeddings...")
        generate_and_store_embeddings()

        print("\nMigration complete!")
    except Exception as e:
        print(f"\nError during migration: {e}")
        print("Migration failed. Please check the error message and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
