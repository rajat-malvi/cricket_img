"""
Database operations module for the Cricket Image Chatbot
"""

import os
import re
import json
import psycopg2
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from langchain.docstore.document import Document

import config

def get_db_connection():
    """
    Get a connection to the PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
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

def database_exists() -> bool:
    """
    Check if the database has been initialized with documents

    Returns:
        bool: True if the database exists and has documents, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if documents table exists and has rows
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'documents'
        )
        """)
        table_exists = cursor.fetchone()[0]

        if table_exists:
            # Check if there are any documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return count > 0

        cursor.close()
        conn.close()

        return False
    except Exception as e:
        print(f"Error checking if database exists: {e}")
        return False

def reference_data_exists() -> bool:
    """
    Check if reference data exists in the database

    Returns:
        bool: True if reference data exists, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if players table exists and has rows
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'players'
        )
        """)
        table_exists = cursor.fetchone()[0]

        if table_exists:
            # Check if there are any players
            cursor.execute("SELECT COUNT(*) FROM players")
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return count > 0

        cursor.close()
        conn.close()

        return False
    except Exception as e:
        print(f"Error checking if reference data exists: {e}")
        return False

def create_tables():
    """
    Create the database tables
    """
    conn = get_db_connection()
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

    conn.commit()
    cursor.close()
    conn.close()

def load_all_reference_data():
    """
    Load all reference data from CSV files
    """
    # Load players
    players_df = pd.read_csv(os.path.join(config.DATA_DIR, "Players.csv"))
    load_reference_data_players(players_df)

    # Load action
    action_df = pd.read_csv(os.path.join(config.DATA_DIR, "Action.csv"))
    load_reference_data(action_df, "action", "action_id", "action_name")

    # Load event
    event_df = pd.read_csv(os.path.join(config.DATA_DIR, "Event.csv"))
    load_reference_data(event_df, "event", "event_id", "event_name")

    # Load mood
    mood_df = pd.read_csv(os.path.join(config.DATA_DIR, "Mood.csv"))
    load_reference_data(mood_df, "mood", "mood_id", "mood_name")

    # Load sublocation
    sublocation_df = pd.read_csv(os.path.join(config.DATA_DIR, "Sublocation.csv"))
    load_reference_data(sublocation_df, "sublocation", "sublocation_id", "sublocation_name")

    # Load cricket data
    load_cricket_data()

def load_reference_data_players(df):
    """
    Load players data from a DataFrame into the players table

    Args:
        df (pd.DataFrame): DataFrame containing the players data
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if table has data
    cursor.execute("SELECT COUNT(*) FROM players")
    count = cursor.fetchone()[0]

    if count == 0:
        # If table is empty, just insert new data
        for _, row in df.iterrows():
            # Check if team_code column exists in the DataFrame
            team_code = row['team_code'] if 'team_code' in df.columns and pd.notna(row['team_code']) else None

            cursor.execute(
                "INSERT INTO players (player_id, player_name, team_code) VALUES (%s, %s, %s)",
                (row['player_id'], row['Player Name'], team_code)
            )
    else:
        # If table has data, update existing records
        for _, row in df.iterrows():
            # Check if team_code column exists in the DataFrame
            team_code = row['team_code'] if 'team_code' in df.columns and pd.notna(row['team_code']) else None

            cursor.execute(
                "UPDATE players SET player_name = %s, team_code = %s WHERE player_id = %s",
                (row['Player Name'], team_code, row['player_id'])
            )

            # If record doesn't exist, insert it
            if cursor.rowcount == 0:
                cursor.execute(
                    "INSERT INTO players (player_id, player_name, team_code) VALUES (%s, %s, %s)",
                    (row['player_id'], row['Player Name'], team_code)
                )

    conn.commit()
    cursor.close()
    conn.close()

def load_reference_data(df, table_name, id_col, name_col):
    """
    Load reference data from a DataFrame into a database table

    Args:
        df (pd.DataFrame): DataFrame containing the reference data
        table_name (str): Name of the database table
        id_col (str): Name of the ID column
        name_col (str): Name of the name column
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert name_col to a database-friendly format
    db_name_col = name_col.lower().replace(' ', '_')

    # Check if table has data
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]

    if count == 0:
        # If table is empty, just insert new data
        for _, row in df.iterrows():
            cursor.execute(
                f"INSERT INTO {table_name} ({id_col}, {db_name_col}) VALUES (%s, %s)",
                (row[id_col], row[name_col])
            )
    else:
        # If table has data, update existing records
        for _, row in df.iterrows():
            cursor.execute(
                f"UPDATE {table_name} SET {db_name_col} = %s WHERE {id_col} = %s",
                (row[name_col], row[id_col])
            )

            # If record doesn't exist, insert it
            if cursor.rowcount == 0:
                cursor.execute(
                    f"INSERT INTO {table_name} ({id_col}, {db_name_col}) VALUES (%s, %s)",
                    (row[id_col], row[name_col])
                )

    conn.commit()
    cursor.close()
    conn.close()

def load_cricket_data():
    """
    Load cricket data from CSV file into the database
    """
    # Load the CSV file
    df = pd.read_csv(config.CSV_FILE)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if table has data
    cursor.execute("SELECT COUNT(*) FROM cricket_data")
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"Cricket data table already has {count} rows. Skipping data load.")
        cursor.close()
        conn.close()
        return

    # Insert new data
    for _, row in df.iterrows():
        # Handle multiple player IDs
        player_id = row['player_id'].split(',')[0] if pd.notna(row['player_id']) else None

        cursor.execute("""
        INSERT INTO cricket_data (
            file_name, url, player_id, datetime_original, date, time_of_day, no_of_faces,
            focus, shot_type, event_id, mood_id, action_id, caption, apparel,
            brands_and_logos, sublocation_id, location, make, model, copyright, photographer
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['File Name'],
            row['URL'],
            player_id,
            row['DateTimeOriginal'] if pd.notna(row['DateTimeOriginal']) else None,
            row['Date'] if pd.notna(row['Date']) else None,
            row['TimeOfDay'] if pd.notna(row['TimeOfDay']) else None,
            row['NoOfFaces'] if pd.notna(row['NoOfFaces']) else None,
            row['Focus'] if pd.notna(row['Focus']) else None,
            row['Shot Type'] if pd.notna(row['Shot Type']) else None,
            row['event_id'] if pd.notna(row['event_id']) else None,
            row['mood_id'] if pd.notna(row['mood_id']) else None,
            row['action_id'] if pd.notna(row['action_id']) else None,
            row['caption'] if pd.notna(row['caption']) else None,
            row['apparel'] if pd.notna(row['apparel']) else None,
            row['brands_and_logos'] if pd.notna(row['brands_and_logos']) else None,
            row['sublocation_id'] if pd.notna(row['sublocation_id']) else None,
            row['Location'] if pd.notna(row['Location']) else None,
            row['Make'] if pd.notna(row['Make']) else None,
            row['Model'] if pd.notna(row['Model']) else None,
            row['Copyright'] if pd.notna(row['Copyright']) else None,
            row['Photographer'] if pd.notna(row['Photographer']) else None
        ))

    conn.commit()
    cursor.close()
    conn.close()

def generate_documents_from_db() -> List[Document]:
    """
    Generate documents from the database for vector storage

    Returns:
        List[Document]: List of documents
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Join cricket_data with reference tables to get names instead of IDs
    cursor.execute("""
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

    documents = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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
        # This format combines the caption with key attributes in a structured way
        content = f"{row[14] or 'Cricket image'} Action: {row[13] or 'Unknown'}. Event: {row[11] or 'Unknown'}. Mood: {row[12] or 'Unknown'}. Location: {row[17] or 'Unknown'}. Time of day: {row[7] or 'Unknown'}. Focus: {row[9] or 'Unknown'}. Shot type: {row[10] or 'Unknown'}. Apparel: {row[15] or 'Unknown'}. Brands and logos: {row[16] or 'None'}. Number of faces: {row[8] or '0'}"

        # Create document
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)

    cursor.close()
    conn.close()

    return documents

def clear_database():
    """
    Clear the vector store database (documents and embeddings)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Delete all embeddings and documents
    cursor.execute("DELETE FROM embeddings")
    cursor.execute("DELETE FROM documents")

    conn.commit()
    cursor.close()
    conn.close()

def insert_documents(documents: List[Document], embeddings: List[List[float]]):
    """
    Insert documents and embeddings into the database

    Args:
        documents (List[Document]): List of documents
        embeddings (List[List[float]]): List of embeddings
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert documents and embeddings
    for i, doc in enumerate(documents):
        # Insert document
        cursor.execute(
            "INSERT INTO documents (content, metadata) VALUES (%s, %s) RETURNING id",
            (doc.page_content, json.dumps(doc.metadata))
        )
        doc_id = cursor.fetchone()[0]

        # Update metadata with document ID
        doc.metadata["document_id"] = doc_id

        # Insert embedding - convert to PostgreSQL vector format
        vector_str = f"[{','.join(str(x) for x in embeddings[i])}]"
        try:
            cursor.execute(
                "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s::vector)",
                (doc_id, vector_str)
            )
        except Exception as e:
            print(f"Error inserting embedding: {e}")
            # Try alternative method with explicit array syntax
            try:
                cursor.execute(
                    "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s::vector)",
                    (doc_id, f"[{','.join(str(float(x)) for x in embeddings[i])}]")
                )
            except Exception as e2:
                print(f"Error inserting embedding (alternative method): {e2}")
                # Last resort: store as bytea
                import pickle
                try:
                    cursor.execute(
                        "INSERT INTO embeddings (document_id, embedding) VALUES (%s, %s)",
                        (doc_id, pickle.dumps(embeddings[i]))
                    )
                except Exception as e3:
                    print(f"Error inserting embedding (bytea method): {e3}")

    conn.commit()
    cursor.close()
    conn.close()

def similarity_search(query_embedding: List[float], k: int = 0, query_text: str = "", similarity_threshold: float = 0.0) -> List[Tuple[Document, float]]:
    """
    Perform a similarity search in the database

    Args:
        query_embedding (List[float]): Query embedding
        k (int): Number of results to return (default: 0, which means return all results)
        query_text (str): Original query text for feedback-based adjustments
        similarity_threshold (float): Minimum similarity score (0.0-1.0) to include results (default: 0.0)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if we should use SQL query for player names
    if is_player_query(query_text):
        print(f"Detected player query: '{query_text}'")
        results = get_images_by_player_name(query_text, k)
        if results:
            print(f"Found {len(results)} results using direct SQL query")
            cursor.close()
            conn.close()
            return results

    # Check if we should use SQL query for press meet
    if is_press_meet_query(query_text):
        print(f"Detected press meet query: '{query_text}'")
        results = get_images_by_press_meet(k)
        if results:
            print(f"Found {len(results)} results using direct SQL query")
            cursor.close()
            conn.close()
            return results

    # Check if we should use SQL query for practice images
    if is_practice_query(query_text):
        print(f"Detected practice query: '{query_text}'")
        results = get_images_by_practice(k)
        if results:
            print(f"Found {len(results)} results using direct SQL query")
            cursor.close()
            conn.close()
            return results

    # Try vector similarity search
    try:
        # Adjust similarity threshold (convert from 0-1 to cosine distance)
        # Note: We're not using distance_threshold directly in the SQL query
        # but keeping this calculation for reference
        _ = 1.0 - similarity_threshold  # distance_threshold

        # Skip feedback adjustments for now
        # We'll implement this function later if needed

        # Convert Python list to PostgreSQL vector format
        vector_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Use pgvector for similarity search
        if k > 0:
            cursor.execute("""
            SELECT d.id, d.content, d.metadata, 1 - (e.embedding <=> %s::vector) as similarity
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """, (vector_str, vector_str, k))
        else:
            cursor.execute("""
            SELECT d.id, d.content, d.metadata, 1 - (e.embedding <=> %s::vector) as similarity
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            WHERE 1 - (e.embedding <=> %s::vector) >= %s
            ORDER BY e.embedding <=> %s::vector
            """, (vector_str, vector_str, similarity_threshold, vector_str))

        results = []
        for row in cursor.fetchall():
            # Unpack row values (doc_id is used for debugging if needed)
            _, content, metadata_json, similarity = row

            # Parse metadata
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            else:
                metadata = metadata_json

            # Ensure metadata has the correct format
            if "document_id" in metadata and "id" not in metadata:
                metadata["id"] = metadata["document_id"]

            if "image_url" not in metadata and "url" in metadata:
                metadata["image_url"] = metadata["url"]

            if "url" not in metadata and "image_url" in metadata:
                metadata["url"] = metadata["image_url"]

            # Create document
            doc = Document(page_content=content, metadata=metadata)

            # Add to results
            results.append((doc, 1.0 - similarity))

        cursor.close()
        conn.close()

        return results
    except Exception as e:
        print(f"Error in similarity search: {e}")
        cursor.close()
        conn.close()
        return []

def get_player_names_in_query(query: str) -> List[str]:
    """
    Get the player names mentioned in the query

    Args:
        query (str): Query text

    Returns:
        List[str]: List of player names found in the query
    """
    # Get all player names
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT player_name, team_code FROM players")
    player_data = cursor.fetchall()
    player_names = [row[0].lower() for row in player_data]
    cursor.close()
    conn.close()

    # Check if any player name is in the query
    query_lower = query.lower()
    found_players = []

    # Name variations for all players
    name_variations = {
        "beuran hendricks": ["beuran", "hendricks", "b hendricks", "b. hendricks"],
        "david wiese": ["david", "wiese", "d wiese", "d. wiese"],
        "donovan ferreira": ["donovan", "ferreira", "d ferreira", "d. ferreira"],
        "devon conway": ["devon", "conway", "d conway", "d. conway"],
        "doug bracewell": ["doug", "bracewell", "d bracewell", "d. bracewell"],
        "eric simons": ["eric", "simons", "e simons", "e. simons"],
        "evan jones": ["evan", "jones", "e jones", "e. jones"],
        "faf du plessis": ["faf", "du plessis", "faf duplessis", "duplessis", "f du plessis", "f. du plessis"],
        "gerald coetzee": ["gerald", "coetzee", "g coetzee", "g. coetzee"],
        "hardus viljoen": ["hardus", "viljoen", "h viljoen", "h. viljoen"],
        "imran tahir": ["imran", "tahir", "i tahir", "i. tahir"],
        "jonny bairstow": ["jonny", "bairstow", "j bairstow", "j. bairstow"],
        "jp king": ["j.p. king", "j p king", "j.p king", "jp", "king", "j. king"],
        "kasi viswanathan": ["kasi", "viswanathan", "k viswanathan", "k. viswanathan"],
        "lakshmi narayanan": ["lakshmi", "narayanan", "l narayanan", "l. narayanan"],
        "leus du plooy": ["leus", "du plooy", "leus duplooy", "l du plooy", "l. du plooy"],
        "lutho sipamla": ["lutho", "sipamla", "l sipamla", "l. sipamla"],
        "maheesh theekshana": ["maheesh", "theekshana", "m theekshana", "m. theekshana"],
        "matheesha pathirana": ["matheesha", "pathirana", "m pathirana", "m. pathirana"],
        "moeen ali": ["moen ali", "mo ali", "moeen", "moen", "m ali", "m. ali", "moeen", "moin ali", "moin", "mo"],
        "sanjay natarajan": ["sanjay", "natarajan", "s natarajan", "s. natarajan"],
        "sibonelo makhanya": ["sibonelo", "makhanya", "s makhanya", "s. makhanya"],
        "stephen fleming": ["fleming", "steve fleming", "stephen", "steve", "s fleming", "s. fleming", "coach", "coach fleming", "head coach"],
        "tabraiz shamsi": ["tabraiz", "shamsi", "t shamsi", "t. shamsi"],
        "tommy simsek": ["tommy", "simsek", "t simsek", "t. simsek"],
        "tshepo moreki": ["tshepo", "moreki", "t moreki", "t. moreki"],
        "wihan lubbe": ["wihan", "lubbe", "w lubbe", "w. lubbe"]
    }

    # First check for exact matches
    for name in player_names:
        if name.lower() in query_lower and name.lower() not in found_players:
            found_players.append(name.lower())

    # Check for name variations
    for standard_name, variations in name_variations.items():
        if standard_name in query_lower or any(var in query_lower for var in variations):
            if standard_name not in found_players:
                found_players.append(standard_name)

    # Check for partial matches (first name or last name)
    for name in player_names:
        name_parts = name.lower().split()
        if len(name_parts) > 1:  # Only for names with first and last name
            first_name, last_name = name_parts[0], name_parts[-1]
            # Check if first or last name appears as a whole word
            if (re.search(r'\b' + re.escape(first_name) + r'\b', query_lower) or
                re.search(r'\b' + re.escape(last_name) + r'\b', query_lower)) and name.lower() not in found_players:
                found_players.append(name.lower())

    return found_players

def is_player_query(query: str) -> bool:
    """
    Check if the query is asking for a specific player, with enhanced name matching

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for a specific player, False otherwise
    """
    # Use the get_player_names_in_query function to check if any player names are in the query
    return len(get_player_names_in_query(query)) > 0

def is_press_meet_query(query: str) -> bool:
    """
    Check if the query is asking for press meet images

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for press meet images, False otherwise
    """
    query_lower = query.lower()
    press_terms = ["press meet", "press conference", "media", "interview", "press"]

    return any(term in query_lower for term in press_terms)

def is_practice_query(query: str) -> bool:
    """
    Check if the query is asking for practice images

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for practice images, False otherwise
    """
    query_lower = query.lower()
    practice_terms = ["practice", "training", "net session", "nets"]

    return any(term in query_lower for term in practice_terms)

def get_images_by_player_name(query: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images for a specific player using SQL query, with enhanced name matching

    Args:
        query (str): Query text
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all player names
    cursor.execute("SELECT player_id, player_name, team_code FROM players")
    player_data = cursor.fetchall()
    players = {row[1].lower(): (row[0], row[2]) for row in player_data}

    # Check for multiple players in the query
    query_lower = query.lower()

    # Check if this is a query for multiple specific players
    if ("and" in query_lower or "&" in query_lower or "," in query_lower) and any(name.lower() in query_lower for name in players.keys()):
        return get_images_with_multiple_players(query, k)

    # Name variations mapping for all players
    name_variations = {
        "beuran hendricks": ["beuran", "hendricks", "b hendricks", "b. hendricks"],
        "david wiese": ["david", "wiese", "d wiese", "d. wiese"],
        "donovan ferreira": ["donovan", "ferreira", "d ferreira", "d. ferreira"],
        "devon conway": ["devon", "conway", "d conway", "d. conway"],
        "doug bracewell": ["doug", "bracewell", "d bracewell", "d. bracewell"],
        "eric simons": ["eric", "simons", "e simons", "e. simons"],
        "evan jones": ["evan", "jones", "e jones", "e. jones"],
        "faf du plessis": ["faf", "du plessis", "faf duplessis", "duplessis", "f du plessis", "f. du plessis"],
        "gerald coetzee": ["gerald", "coetzee", "g coetzee", "g. coetzee"],
        "hardus viljoen": ["hardus", "viljoen", "h viljoen", "h. viljoen"],
        "imran tahir": ["imran", "tahir", "i tahir", "i. tahir"],
        "jonny bairstow": ["jonny", "bairstow", "j bairstow", "j. bairstow"],
        "jp king": ["j.p. king", "j p king", "j.p king", "jp", "king", "j. king"],
        "kasi viswanathan": ["kasi", "viswanathan", "k viswanathan", "k. viswanathan"],
        "lakshmi narayanan": ["lakshmi", "narayanan", "l narayanan", "l. narayanan"],
        "leus du plooy": ["leus", "du plooy", "leus duplooy", "l du plooy", "l. du plooy"],
        "lutho sipamla": ["lutho", "sipamla", "l sipamla", "l. sipamla"],
        "maheesh theekshana": ["maheesh", "theekshana", "m theekshana", "m. theekshana"],
        "matheesha pathirana": ["matheesha", "pathirana", "m pathirana", "m. pathirana"],
        "moeen ali": ["moen ali", "mo ali", "moeen", "moen", "m ali", "m. ali", "moeen", "moin ali", "moin", "mo"],
        "sanjay natarajan": ["sanjay", "natarajan", "s natarajan", "s. natarajan"],
        "sibonelo makhanya": ["sibonelo", "makhanya", "s makhanya", "s. makhanya"],
        "stephen fleming": ["fleming", "steve fleming", "stephen", "steve", "s fleming", "s. fleming", "coach", "coach fleming", "head coach"],
        "tabraiz shamsi": ["tabraiz", "shamsi", "t shamsi", "t. shamsi"],
        "tommy simsek": ["tommy", "simsek", "t simsek", "t. simsek"],
        "tshepo moreki": ["tshepo", "moreki", "t moreki", "t. moreki"],
        "wihan lubbe": ["wihan", "lubbe", "w lubbe", "w. lubbe"]
    }

    # Find player ID using various matching techniques
    player_id = None

    # 1. Try exact name match
    for name, (pid, _) in players.items():
        if name.lower() in query_lower:
            player_id = pid
            break

    # 2. Try name variations if no exact match
    if not player_id:
        for standard_name, variations in name_variations.items():
            if standard_name in query_lower or any(var in query_lower for var in variations):
                # Find the player ID for this standard name
                for name, (pid, _) in players.items():
                    if name.lower() == standard_name or any(var == name.lower() for var in variations):
                        player_id = pid
                        break
                if player_id:
                    break

    # 3. Try partial name matching if still no match
    if not player_id:
        for name, (pid, _) in players.items():
            name_parts = name.lower().split()
            if len(name_parts) > 1:  # Only for names with first and last name
                first_name, last_name = name_parts[0], name_parts[-1]
                # Check if first or last name appears as a whole word
                if re.search(r'\b' + re.escape(first_name) + r'\b', query_lower) or \
                   re.search(r'\b' + re.escape(last_name) + r'\b', query_lower):
                    player_id = pid
                    break

    # If no player found, return empty list
    if not player_id:
        cursor.close()
        conn.close()
        return []

    # Check for specific action or location in the query
    action_clause = ""
    location_clause = ""

    # Check for action terms
    action_terms = ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]
    for action in action_terms:
        if action in query_lower:
            cursor.execute("SELECT action_id FROM action WHERE LOWER(action_name) LIKE %s", (f"%{action.lower()}%",))
            action_ids = cursor.fetchall()
            if action_ids:
                action_id_list = [f"'{row[0]}'" for row in action_ids]
                action_clause = f" AND c.action_id IN ({', '.join(action_id_list)})"
                break

    # Check for location terms
    location_terms = ["stadium", "field", "nets", "dressing room", "press room"]
    for location in location_terms:
        if location in query_lower:
            cursor.execute("SELECT sublocation_id FROM sublocation WHERE LOWER(sublocation_name) LIKE %s", (f"%{location.lower()}%",))
            sublocation_ids = cursor.fetchall()
            if sublocation_ids:
                sublocation_id_list = [f"'{row[0]}'" for row in sublocation_ids]
                location_clause = f" AND c.sublocation_id IN ({', '.join(sublocation_id_list)})"
                break

    # Check for solo/individual image request
    solo_clause = ""
    solo_terms = ["solo", "alone", "individual", "single", "by himself", "by herself"]
    if any(term in query_lower for term in solo_terms):
        solo_clause = " AND c.no_of_faces = 1"

    # Get images for this player with additional filters
    limit_clause = f"LIMIT {k}" if k > 0 else ""

    # Build the complete WHERE clause with all filters
    where_clause = f"WHERE c.player_id = %s{action_clause}{location_clause}{solo_clause}"

    cursor.execute(f"""
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
    {where_clause}
    {limit_clause}
    """, (player_id,))

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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
        # This format combines the caption with key attributes in a structured way
        content = f"{row[14] or 'Cricket image'} Action: {row[13] or 'Unknown'}. Event: {row[11] or 'Unknown'}. Mood: {row[12] or 'Unknown'}. Location: {row[17] or 'Unknown'}. Time of day: {row[7] or 'Unknown'}. Focus: {row[9] or 'Unknown'}. Shot type: {row[10] or 'Unknown'}. Apparel: {row[15] or 'Unknown'}. Brands and logos: {row[16] or 'None'}. Number of faces: {row[8] or '0'}"

        # Create document
        doc = Document(page_content=content.strip(), metadata=metadata)

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_by_press_meet(k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images from press meets using SQL query

    Args:
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get press meet images
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE e.event_name = 'Press Meet'
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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
        # This format combines the caption with key attributes in a structured way
        content = f"{row[14] or 'Cricket image'} Action: {row[13] or 'Unknown'}. Event: {row[11] or 'Unknown'}. Mood: {row[12] or 'Unknown'}. Location: {row[17] or 'Unknown'}. Time of day: {row[7] or 'Unknown'}. Focus: {row[9] or 'Unknown'}. Shot type: {row[10] or 'Unknown'}. Apparel: {row[15] or 'Unknown'}. Brands and logos: {row[16] or 'None'}. Number of faces: {row[8] or '0'}"

        # Create document
        doc = Document(page_content=content.strip(), metadata=metadata)

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def store_feedback(doc_id: int, query: str, image_url: str, rating: int) -> bool:
    """
    Store user feedback on image relevance

    Args:
        doc_id (int): Document ID
        query (str): Query text
        image_url (str): Image URL
        rating (int): User rating (1 for positive, -1 for negative)

    Returns:
        bool: True if feedback was stored successfully, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert feedback
        cursor.execute(
            "INSERT INTO feedback (document_id, query, image_url, rating) VALUES (%s, %s, %s, %s)",
            (doc_id, query, image_url, rating)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        print(f"Error storing feedback: {e}")
        return False

def get_document_id_from_url(url: str) -> Optional[int]:
    """
    Get document ID from image URL

    Args:
        url (str): Image URL

    Returns:
        Optional[int]: Document ID or None if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get document ID
        cursor.execute(
            "SELECT id FROM documents WHERE metadata->>'image_url' = %s OR metadata->>'url' = %s LIMIT 1",
            (url, url)
        )

        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return result[0] if result else None
    except Exception as e:
        print(f"Error getting document ID from URL: {e}")
        return None

def get_count_from_db(query_type: str) -> int:
    """
    Get count of images from the database based on query type

    Args:
        query_type (str): Type of query (e.g., 'press_meet', 'practice', 'total')

    Returns:
        int: Count of images
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if query_type == 'press_meet':
            cursor.execute("""
            SELECT COUNT(*)
            FROM cricket_data c
            JOIN event e ON c.event_id = e.event_id
            WHERE e.event_name = 'Press Meet'
            """)
        elif query_type == 'practice':
            cursor.execute("""
            SELECT COUNT(*)
            FROM cricket_data c
            JOIN event e ON c.event_id = e.event_id
            WHERE e.event_name = 'Practice'
            """)
        elif query_type == 'match':
            cursor.execute("""
            SELECT COUNT(*)
            FROM cricket_data c
            JOIN event e ON c.event_id = e.event_id
            WHERE e.event_name = 'Match'
            """)
        elif query_type == 'promotional':
            cursor.execute("""
            SELECT COUNT(*)
            FROM cricket_data c
            JOIN event e ON c.event_id = e.event_id
            WHERE e.event_name = 'Promotional Event'
            """)
        elif query_type == 'total':
            cursor.execute("SELECT COUNT(*) FROM cricket_data")
        else:
            cursor.close()
            conn.close()
            return 0

        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting count from database: {e}")
        cursor.close()
        conn.close()
        return 0

def get_images_by_practice(k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images from practice sessions using SQL query

    Args:
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get practice images
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE e.event_name = 'Practice'
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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
        # This format combines the caption with key attributes in a structured way
        content = f"{row[14] or 'Cricket image'} Action: {row[13] or 'Unknown'}. Event: {row[11] or 'Unknown'}. Mood: {row[12] or 'Unknown'}. Location: {row[17] or 'Unknown'}. Time of day: {row[7] or 'Unknown'}. Focus: {row[9] or 'Unknown'}. Shot type: {row[10] or 'Unknown'}. Apparel: {row[15] or 'Unknown'}. Brands and logos: {row[16] or 'None'}. Number of faces: {row[8] or '0'}"

        # Create document
        doc = Document(page_content=content.strip(), metadata=metadata)

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_by_action(action_name: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images for a specific action using SQL query

    Args:
        action_name (str): Action name to search for
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get action ID
    cursor.execute("SELECT action_id FROM action WHERE LOWER(action_name) LIKE %s", (f"%{action_name.lower()}%",))
    action_ids = cursor.fetchall()

    if not action_ids:
        cursor.close()
        conn.close()
        return []

    # Build query with all matching action IDs
    action_id_list = [f"'{row[0]}'" for row in action_ids]
    action_id_clause = f"c.action_id IN ({', '.join(action_id_list)})"

    # Get images for this action
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE {action_id_clause}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_by_mood(mood_name: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images for a specific mood using SQL query

    Args:
        mood_name (str): Mood name to search for
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get mood ID
    cursor.execute("SELECT mood_id FROM mood WHERE LOWER(mood_name) LIKE %s", (f"%{mood_name.lower()}%",))
    mood_ids = cursor.fetchall()

    if not mood_ids:
        cursor.close()
        conn.close()
        return []

    # Build query with all matching mood IDs
    mood_id_list = [f"'{row[0]}'" for row in mood_ids]
    mood_id_clause = f"c.mood_id IN ({', '.join(mood_id_list)})"

    # Get images for this mood
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE {mood_id_clause}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_by_location(location_name: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images for a specific location using SQL query

    Args:
        location_name (str): Location name to search for
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Try to find sublocation first
    cursor.execute("SELECT sublocation_id FROM sublocation WHERE LOWER(sublocation_name) LIKE %s", (f"%{location_name.lower()}%",))
    sublocation_ids = cursor.fetchall()

    if sublocation_ids:
        # Build query with all matching sublocation IDs
        sublocation_id_list = [f"'{row[0]}'" for row in sublocation_ids]
        location_clause = f"c.sublocation_id IN ({', '.join(sublocation_id_list)})"
    else:
        # Try to match against location field
        location_clause = f"LOWER(c.location) LIKE '%{location_name.lower()}%'"

    # Get images for this location
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE {location_clause}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_with_multiple_players(query: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images containing multiple specific players

    Args:
        query (str): Query text containing multiple player names
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all player names
    cursor.execute("SELECT player_id, player_name, team_code FROM players")
    player_data = cursor.fetchall()
    players = {row[1].lower(): (row[0], row[1]) for row in player_data}

    # Print the query for debugging
    print(f"Processing multiple players query: '{query}'")

    # Use get_player_names_in_query to identify player names in the query
    player_names_in_query = get_player_names_in_query(query)
    if player_names_in_query:
        print(f"Player names found in query: {player_names_in_query}")

    # Name variations mapping for all players
    name_variations = {
        "beuran hendricks": ["beuran", "hendricks", "b hendricks", "b. hendricks"],
        "david wiese": ["david", "wiese", "d wiese", "d. wiese"],
        "donovan ferreira": ["donovan", "ferreira", "d ferreira", "d. ferreira"],
        "devon conway": ["devon", "conway", "d conway", "d. conway"],
        "doug bracewell": ["doug", "bracewell", "d bracewell", "d. bracewell"],
        "eric simons": ["eric", "simons", "e simons", "e. simons"],
        "evan jones": ["evan", "jones", "e jones", "e. jones"],
        "faf du plessis": ["faf", "du plessis", "faf duplessis", "duplessis", "f du plessis", "f. du plessis"],
        "gerald coetzee": ["gerald", "coetzee", "g coetzee", "g. coetzee"],
        "hardus viljoen": ["hardus", "viljoen", "h viljoen", "h. viljoen"],
        "imran tahir": ["imran", "tahir", "i tahir", "i. tahir"],
        "jonny bairstow": ["jonny", "bairstow", "j bairstow", "j. bairstow"],
        "jp king": ["j.p. king", "j p king", "j.p king", "jp", "king", "j. king"],
        "kasi viswanathan": ["kasi", "viswanathan", "k viswanathan", "k. viswanathan"],
        "lakshmi narayanan": ["lakshmi", "narayanan", "l narayanan", "l. narayanan"],
        "leus du plooy": ["leus", "du plooy", "leus duplooy", "l du plooy", "l. du plooy"],
        "lutho sipamla": ["lutho", "sipamla", "l sipamla", "l. sipamla"],
        "maheesh theekshana": ["maheesh", "theekshana", "m theekshana", "m. theekshana"],
        "matheesha pathirana": ["matheesha", "pathirana", "m pathirana", "m. pathirana"],
        "moeen ali": ["moen ali", "mo ali", "moeen", "moen", "m ali", "m. ali"],
        "sanjay natarajan": ["sanjay", "natarajan", "s natarajan", "s. natarajan"],
        "sibonelo makhanya": ["sibonelo", "makhanya", "s makhanya", "s. makhanya"],
        "stephen fleming": ["fleming", "steve fleming", "stephen", "steve", "s fleming", "s. fleming"],
        "tabraiz shamsi": ["tabraiz", "shamsi", "t shamsi", "t. shamsi"],
        "tommy simsek": ["tommy", "simsek", "t simsek", "t. simsek"],
        "tshepo moreki": ["tshepo", "moreki", "t moreki", "t. moreki"],
        "wihan lubbe": ["wihan", "lubbe", "w lubbe", "w. lubbe"]
    }

    # Find all player IDs mentioned in the query
    query_lower = query.lower()
    mentioned_player_ids = []

    # Use the get_player_names_in_query function to get player names in the query
    player_names_in_query = get_player_names_in_query(query)

    # Get player IDs for the identified player names
    for player_name in player_names_in_query:
        for name, (pid, _) in players.items():
            if name.lower() == player_name:
                if pid not in mentioned_player_ids:
                    mentioned_player_ids.append(pid)
                    print(f"Found player ID {pid} for player name '{player_name}'")
                break

    # If no player IDs were found using get_player_names_in_query, fall back to the old method
    if not mentioned_player_ids:
        print("No player IDs found using get_player_names_in_query, falling back to direct matching")

        # Check for exact matches
        for name, (pid, _) in players.items():
            if name.lower() in query_lower:
                mentioned_player_ids.append(pid)
                print(f"Found player ID {pid} for player name '{name}' using exact match")

        # Check for name variations
        for standard_name, variations in name_variations.items():
            if standard_name in query_lower or any(var in query_lower for var in variations):
                # Find the player ID for this standard name
                for name, (pid, _) in players.items():
                    if name.lower() == standard_name or any(var == name.lower() for var in variations):
                        if pid not in mentioned_player_ids:
                            mentioned_player_ids.append(pid)
                            print(f"Found player ID {pid} for player name '{name}' using name variations")
                        break

        # Check for partial matches
        for name, (pid, _) in players.items():
            if pid not in mentioned_player_ids:  # Skip if already found
                name_parts = name.lower().split()
                if len(name_parts) > 1:  # Only for names with first and last name
                    first_name, last_name = name_parts[0], name_parts[-1]
                    # Check if first or last name appears as a whole word
                    if re.search(r'\b' + re.escape(first_name) + r'\b', query_lower) or \
                       re.search(r'\b' + re.escape(last_name) + r'\b', query_lower):
                        mentioned_player_ids.append(pid)
                        print(f"Found player ID {pid} for player name '{name}' using partial match")

    # Check if this is a general group photo query without specific player names
    group_photo_terms = ["group photo", "team photo", "players together", "group of players", "multiple players"]
    is_group_photo_query = any(term in query_lower for term in group_photo_terms)

    # If no player names are found but it's a group photo query, continue with a different approach
    if len(mentioned_player_ids) < 2 and not is_group_photo_query:
        cursor.close()
        conn.close()
        return []

    # Check for specific action or location in the query
    action_clause = ""
    location_clause = ""

    # Check for action terms
    action_terms = ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]
    for action in action_terms:
        if action in query_lower:
            cursor.execute("SELECT action_id FROM action WHERE LOWER(action_name) LIKE %s", (f"%{action.lower()}%",))
            action_ids = cursor.fetchall()
            if action_ids:
                action_id_list = [f"'{row[0]}'" for row in action_ids]
                action_clause = f" AND c.action_id IN ({', '.join(action_id_list)})"
                break

    # Check for location terms
    location_terms = ["stadium", "field", "nets", "dressing room", "press room"]
    for location in location_terms:
        if location in query_lower:
            cursor.execute("SELECT sublocation_id FROM sublocation WHERE LOWER(sublocation_name) LIKE %s", (f"%{location.lower()}%",))
            sublocation_ids = cursor.fetchall()
            if sublocation_ids:
                sublocation_id_list = [f"'{row[0]}'" for row in sublocation_ids]
                location_clause = f" AND c.sublocation_id IN ({', '.join(sublocation_id_list)})"
                break

    # Always enforce no_of_faces >= 2 for multiple player queries
    # This ensures we only get images with at least 2 people in them
    together_clause = " AND c.no_of_faces >= 2"

    # Build the player filter - look for images that have multiple players in the caption or metadata
    player_names = []
    for pid in mentioned_player_ids:
        for name, (id, _) in players.items():
            if id == pid:
                player_names.append(name)
                break

    # Get images that contain multiple players
    # We'll use a more flexible approach to find images with multiple players:
    # 1. If specific player names are mentioned, try to find images with those players
    # 2. If no specific players or only one player is mentioned, find images with multiple faces
    # 3. Always ensure the no_of_faces is at least 2 to confirm multiple people are in the image
    # 4. Add additional filtering for terms like "together", "same frame", etc.

    # Check for specific terms indicating players together
    together_terms = ["together", "same frame", "single frame", "with each other", "standing together", "group", "team"]
    together_term_present = any(term in query_lower for term in together_terms)

    # Build the WHERE clause based on the query type
    if is_group_photo_query:
        # For group photo queries, find images with multiple faces and terms like "players", "team", etc.
        general_terms = ["players", "team", "group", "together", "multiple"]
        term_conditions = []
        for term in general_terms:
            term_conditions.append(f"LOWER(c.caption) LIKE '%{term}%'")
            term_conditions.append(f"LOWER(c.description) LIKE '%{term}%'")

        # Add specific terms from the query
        for term in group_photo_terms:
            if term in query_lower:
                term_parts = term.split()
                for part in term_parts:
                    if len(part) > 3:  # Only use meaningful words
                        term_conditions.append(f"LOWER(c.caption) LIKE '%{part}%'")
                        term_conditions.append(f"LOWER(c.description) LIKE '%{part}%'")

        player_clause = " OR ".join(term_conditions)
        player_clause = f"({player_clause})"
    elif len(player_names) >= 2:
        # If we have multiple player names, try to find images with at least one of them
        # This is more flexible than requiring all names to be present
        player_conditions = []
        for name in player_names:
            player_conditions.append(f"(LOWER(c.caption) LIKE '%{name.lower()}%' OR LOWER(c.description) LIKE '%{name.lower()}%')")

        # Use OR between player conditions to find images with any of the players
        player_clause = " OR ".join(player_conditions)

        # Wrap in parentheses for proper SQL syntax
        player_clause = f"({player_clause})"
    elif len(player_names) == 1:
        # If we have only one player name, find images with that player and multiple faces
        name = player_names[0]
        player_clause = f"(LOWER(c.caption) LIKE '%{name.lower()}%' OR LOWER(c.description) LIKE '%{name.lower()}%')"
    else:
        # If no specific player names, find images with terms like "players", "team", etc.
        general_terms = ["players", "team", "group", "together", "multiple"]
        term_conditions = []
        for term in general_terms:
            term_conditions.append(f"LOWER(c.caption) LIKE '%{term}%'")
            term_conditions.append(f"LOWER(c.description) LIKE '%{term}%'")

        player_clause = " OR ".join(term_conditions)
        player_clause = f"({player_clause})"

    # If specific "together" terms are present, add them to the search criteria
    # This helps prioritize images that explicitly mention players together
    if together_term_present:
        together_term_conditions = []
        for term in together_terms:
            if term in query_lower:
                together_term_conditions.append(f"LOWER(c.caption) LIKE '%{term}%'")
                together_term_conditions.append(f"LOWER(c.description) LIKE '%{term}%'")

        if together_term_conditions:
            together_term_clause = " OR ".join(together_term_conditions)
            player_clause = f"({player_clause}) OR ({together_term_clause})"

    # Build the complete WHERE clause
    where_clause = f"WHERE ({player_clause}){action_clause}{location_clause}{together_clause}"

    # Get images matching the criteria
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    {where_clause}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    # Additional verification step: ensure the results are relevant
    # This helps filter out false positives
    verified_results = []
    for doc, similarity in results:
        caption = doc.metadata.get('caption', '').lower()
        description = doc.page_content.lower()
        no_of_faces = doc.metadata.get('no_of_faces', 0)

        # For queries specifically asking for players together, strictly enforce multiple faces
        if ("together" in query_lower or "and" in query_lower or "&" in query_lower or
            "same frame" in query_lower or "single frame" in query_lower):
            # Always ensure there are multiple faces in the image for "together" queries
            if no_of_faces is None or no_of_faces < 2:
                continue

        # If we have specific player names, check if at least one is present
        # This is more flexible than requiring all names to be present
        if player_names:
            any_player_present = False
            for name in player_names:
                name_lower = name.lower()
                # Check for full name or parts of the name
                name_parts = name_lower.split()

                # For names with multiple parts, check if the name is present
                if len(name_parts) > 1:
                    # Check if full name is present
                    if name_lower in caption or name_lower in description:
                        any_player_present = True
                        break
                    # If full name isn't present, check if first and last name are both present
                    first_name, last_name = name_parts[0], name_parts[-1]
                    if (first_name in caption and last_name in caption) or \
                       (first_name in description and last_name in description):
                        any_player_present = True
                        break
                else:
                    # For single-part names, just check if the name is present
                    if name_lower in caption or name_lower in description:
                        any_player_present = True
                        break

            # If no player names are present, skip this result
            if not any_player_present:
                continue

        # Check if the caption or description indicates multiple players
        multiple_player_terms = ["players", "team", "group", "together", "multiple"]
        has_multiple_player_terms = any(term in caption or term in description for term in multiple_player_terms)

        # If the query is about multiple players together, prioritize results with relevant terms
        if "together" in query_lower or "and" in query_lower or "&" in query_lower or "same frame" in query_lower:
            # If the caption or description mentions multiple players, include it
            if has_multiple_player_terms:
                # Double-check that images for "together" queries have multiple faces
                if no_of_faces is None or no_of_faces < 2:
                    continue
                verified_results.append((doc, similarity))
            # Otherwise, only include it if it has a high number of faces
            elif no_of_faces >= 2:  # Changed from 3 to 2 to be consistent with the together_clause
                verified_results.append((doc, similarity))
        else:
            # For other queries, include all results
            verified_results.append((doc, similarity))

    cursor.close()
    conn.close()

    return verified_results

def get_images_by_activity(query: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images matching specific activities like "traveling", "celebrating", etc.

    Args:
        query (str): Query text containing activity description
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Map common activities to keywords and phrases to search for
    activity_mapping = {
        "traveling": ["travel", "bus", "airport", "plane", "flight", "journey", "arrival", "departure"],
        "celebrating": ["celebrate", "celebration", "trophy", "win", "victory", "champagne", "party"],
        "training": ["training", "practice", "drill", "exercise", "warm-up", "warm up", "net session", "nets"],
        "meeting fans": ["fan", "fans", "autograph", "selfie", "crowd", "spectator", "supporter"],
        "press conference": ["press", "media", "interview", "microphone", "journalist", "reporter", "conference", "press meeting"],
        "team huddle": ["huddle", "team talk", "gathering", "group", "circle", "meeting", "discussion"],
        "eating": ["meal", "dinner", "lunch", "breakfast", "restaurant", "food", "eating", "dining"],
        "relaxing": ["relax", "leisure", "hotel", "pool", "beach", "rest", "break", "downtime"]
    }

    query_lower = query.lower()

    # Identify which activity is being queried
    target_activity = None
    for activity, keywords in activity_mapping.items():
        if activity in query_lower or any(keyword in query_lower for keyword in keywords):
            target_activity = activity
            break

    if not target_activity:
        cursor.close()
        conn.close()
        return []

    # Get keywords for the target activity
    activity_keywords = activity_mapping[target_activity]

    # Build query conditions for each keyword
    keyword_conditions = []
    for keyword in activity_keywords:
        keyword_lower = keyword.lower()
        keyword_conditions.append(f"LOWER(c.caption) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.description) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.focus) LIKE '%{keyword_lower}%'")

    # Combine conditions with OR
    combined_condition = " OR ".join(keyword_conditions)

    # Get images matching the activity
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE {combined_condition}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results

def get_images_by_keywords(keywords: List[str], k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images matching keywords in caption or description

    Args:
        keywords (List[str]): List of keywords to search for
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    if not keywords:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query conditions for each keyword
    keyword_conditions = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        keyword_conditions.append(f"LOWER(c.caption) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.description) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.focus) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.shot_type) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.apparel) LIKE '%{keyword_lower}%'")
        keyword_conditions.append(f"LOWER(c.brands_and_logos) LIKE '%{keyword_lower}%'")

    # Combine conditions with OR
    combined_condition = " OR ".join(keyword_conditions)

    # Get images matching keywords
    limit_clause = f"LIMIT {k}" if k > 0 else ""
    cursor.execute(f"""
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
    WHERE {combined_condition}
    {limit_clause}
    """)

    results = []
    for row in cursor.fetchall():
        # Create metadata for retrieval
        metadata = {
            "id": row[0],
            "file_name": row[1],
            "url": row[2],
            "image_url": row[2],  # Duplicate for compatibility
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

        # Use a fixed similarity score for SQL results
        similarity = 0.1  # Low distance = high similarity
        results.append((doc, similarity))

    cursor.close()
    conn.close()

    return results