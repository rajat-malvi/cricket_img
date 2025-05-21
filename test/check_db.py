"""
Script to check database tables and content
"""

import psycopg2
import json

def get_db_connection():
    """
    Get a connection to the PostgreSQL database
    """
    return psycopg2.connect(
        dbname="jsk_data",
        user="postgres",
        password="Skd6397@@",
        host="localhost",
        port="5432"
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

def check_embeddings():
    """
    Check the embeddings table
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if embeddings table exists
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'embeddings')")
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            # Get count of embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            print(f"Embeddings count: {count}")
            
            # Get a sample embedding
            cursor.execute("SELECT document_id, embedding FROM embeddings LIMIT 1")
            sample = cursor.fetchone()
            
            if sample:
                doc_id, embedding = sample
                print(f"Sample embedding for document_id {doc_id}:")
                
                # Check embedding type and format
                print(f"Embedding type: {type(embedding)}")
                
                # Print first few values of the embedding
                if isinstance(embedding, list):
                    print(f"First 5 values: {embedding[:5]}")
                elif hasattr(embedding, '__iter__'):
                    try:
                        print(f"First 5 values: {list(embedding)[:5]}")
                    except:
                        print("Could not convert embedding to list")
                else:
                    print("Embedding is not iterable")
        else:
            print("Embeddings table does not exist")
    except Exception as e:
        print(f"Error checking embeddings: {e}")
    
    cursor.close()
    conn.close()

def check_documents():
    """
    Check the documents table
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if documents table exists
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'documents')")
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            # Get count of documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            print(f"Documents count: {count}")
            
            # Get a sample document
            cursor.execute("SELECT id, content, metadata FROM documents LIMIT 1")
            sample = cursor.fetchone()
            
            if sample:
                doc_id, content, metadata = sample
                print(f"Sample document (id: {doc_id}):")
                print(f"Content: {content[:100]}...")
                
                # Check metadata type and format
                print(f"Metadata type: {type(metadata)}")
                
                # Try to print metadata
                if isinstance(metadata, dict):
                    print(f"Metadata keys: {list(metadata.keys())}")
                elif isinstance(metadata, str):
                    try:
                        metadata_dict = json.loads(metadata)
                        print(f"Metadata keys: {list(metadata_dict.keys())}")
                    except:
                        print(f"Metadata (first 100 chars): {metadata[:100]}...")
                else:
                    print(f"Metadata: {metadata}")
        else:
            print("Documents table does not exist")
    except Exception as e:
        print(f"Error checking documents: {e}")
    
    cursor.close()
    conn.close()

def check_similarity_search():
    """
    Test the similarity search functionality
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if we have the pgvector extension
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        pgvector_exists = cursor.fetchone() is not None
        print(f"pgvector extension exists: {pgvector_exists}")
        
        # Check if we have embeddings
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'embeddings')")
        embeddings_exist = cursor.fetchone()[0]
        
        if embeddings_exist:
            # Get count of embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"Found {count} embeddings")
                
                # Try a simple similarity search
                if pgvector_exists:
                    try:
                        # Get a sample embedding to use for search
                        cursor.execute("SELECT embedding FROM embeddings LIMIT 1")
                        sample_embedding = cursor.fetchone()[0]
                        
                        # Use the sample embedding to search
                        cursor.execute("""
                        SELECT d.id, d.content, 1 - (e.embedding <=> %s) as similarity
                        FROM embeddings e
                        JOIN documents d ON e.document_id = d.id
                        ORDER BY e.embedding <=> %s
                        LIMIT 3
                        """, (sample_embedding, sample_embedding))
                        
                        results = cursor.fetchall()
                        print(f"Similarity search results: {len(results)}")
                        
                        for doc_id, content, similarity in results:
                            print(f"Document {doc_id}: Similarity {similarity:.4f}")
                            print(f"Content: {content[:50]}...")
                    except Exception as e:
                        print(f"Error in similarity search: {e}")
                else:
                    print("pgvector extension not available, skipping similarity search test")
            else:
                print("No embeddings found")
        else:
            print("Embeddings table does not exist")
    except Exception as e:
        print(f"Error in similarity search test: {e}")
    
    cursor.close()
    conn.close()

def check_player_query():
    """
    Test a player name query
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if we have the players table
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'players')")
        players_exist = cursor.fetchone()[0]
        
        if players_exist:
            # Get count of players
            cursor.execute("SELECT COUNT(*) FROM players")
            count = cursor.fetchone()[0]
            print(f"Players count: {count}")
            
            if count > 0:
                # Get a sample player
                cursor.execute("SELECT player_id, player_name FROM players LIMIT 5")
                players = cursor.fetchall()
                
                print("Sample players:")
                for player_id, player_name in players:
                    print(f"- {player_name} (ID: {player_id})")
                
                # Try a query for a specific player
                test_player = players[0][1]  # Use the first player's name
                
                print(f"\nTesting query for player: {test_player}")
                
                # Query cricket_data for this player
                cursor.execute("""
                SELECT c.id, c.file_name, c.url, p.player_name, a.action_name, e.event_name
                FROM cricket_data c
                JOIN players p ON c.player_id = p.player_id
                LEFT JOIN action a ON c.action_id = a.action_id
                LEFT JOIN event e ON c.event_id = e.event_id
                WHERE p.player_name = %s
                LIMIT 3
                """, (test_player,))
                
                results = cursor.fetchall()
                print(f"Found {len(results)} images for {test_player}")
                
                for row in results:
                    print(f"Image ID: {row[0]}, File: {row[1]}")
                    print(f"URL: {row[2]}")
                    print(f"Player: {row[3]}, Action: {row[4]}, Event: {row[5]}")
                    print()
            else:
                print("No players found")
        else:
            print("Players table does not exist")
    except Exception as e:
        print(f"Error in player query test: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    print("Checking database...")
    check_tables()
    print("\nChecking documents...")
    check_documents()
    print("\nChecking embeddings...")
    check_embeddings()
    print("\nTesting similarity search...")
    check_similarity_search()
    print("\nTesting player query...")
    check_player_query()
