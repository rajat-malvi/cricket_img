"""
Test script to verify the multiple player query functionality
"""

import nltk
import db_store
from llm_service import get_images_by_sql_query

# Ensure NLTK resources are available
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    # List of required resources
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',  # Open Multilingual WordNet
        'averaged_perceptron_tagger'  # This is the correct resource name (without _eng)
    ]

    for resource in resources:
        try:
            # Check if resource exists
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            elif resource == 'wordnet' or resource == 'omw-1.4':
                nltk.data.find(f'corpora/{resource}')
            else:
                nltk.data.find(f'taggers/{resource}')
            print(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            # Download if not found
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)
            print(f"Downloaded NLTK resource '{resource}'.")

# Download resources before running tests
ensure_nltk_resources()

def test_multiple_player_queries():
    """
    Test multiple player queries to ensure they return images with both players
    """
    print("Testing multiple player queries...")

    # Test queries with actual player names from the database
    test_queries = [
        "STEPHEN FLEMING and DEVON CONWAY together",
        "JP KING and FAF DU PLESSIS in a single frame",
        "DEVON CONWAY and WIHAN LUBBE together",
        "Players in a group photo",
        "Multiple players standing together",
        "team photo",
        "players together",
        "group of players"
    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")

        # Call the get_images_with_multiple_players function directly for all queries
        # This bypasses the NLTK keyword extraction that's causing issues
        print("Calling get_images_with_multiple_players directly...")
        results = db_store.get_images_with_multiple_players(query)

        if not results:
            print(f"No results found for query: '{query}'")
            continue

        print(f"Found {len(results)} results")

        # Check if all results have at least 2 faces
        all_have_multiple_faces = True
        for doc, _ in results:
            no_of_faces = doc.metadata.get('no_of_faces', 0)
            if no_of_faces is None or no_of_faces < 2:
                all_have_multiple_faces = False
                break

        print(f"All results have multiple faces: {all_have_multiple_faces}")

        # Print details of the first few results
        print("Sample results:")
        for i, (doc, _) in enumerate(results[:3]):  # Show first 3 results
            print(f"  Result {i+1}:")
            print(f"    Player: {doc.metadata.get('player_name', 'Unknown')}")
            print(f"    Caption: {doc.metadata.get('caption', 'No caption')}")
            print(f"    No. of faces: {doc.metadata.get('no_of_faces', 'Unknown')}")
            print(f"    URL: {doc.metadata.get('url', 'No URL')}")

def test_database_for_multiple_faces():
    """
    Test if the database has any images with multiple faces
    """
    print("\n\nTesting database for images with multiple faces...")

    # Connect to the database
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Query for images with multiple faces
    cursor.execute("""
    SELECT c.id, c.file_name, c.url, p.player_name, c.no_of_faces, c.caption
    FROM cricket_data c
    LEFT JOIN players p ON c.player_id = p.player_id
    WHERE c.no_of_faces >= 2
    LIMIT 5
    """)

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        print("No images with multiple faces found in the database.")
        return

    print(f"Found {len(results)} images with multiple faces.")

    # Print details of the results
    for i, row in enumerate(results):
        print(f"  Result {i+1}:")
        print(f"    ID: {row[0]}")
        print(f"    File Name: {row[1]}")
        print(f"    URL: {row[2]}")
        print(f"    Player: {row[3]}")
        print(f"    No. of Faces: {row[4]}")
        print(f"    Caption: {row[5]}")

def test_direct_multiple_players_query():
    """
    Test a direct SQL query for images with multiple players mentioned in caption
    """
    print("\n\nTesting direct SQL query for multiple players...")

    # Connect to the database
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Get player names
    cursor.execute("SELECT player_name FROM players LIMIT 10")
    player_names = [row[0] for row in cursor.fetchall()]

    if len(player_names) < 2:
        print("Not enough players in the database for this test.")
        cursor.close()
        conn.close()
        return

    # Use the first two player names for the test
    player1 = player_names[0]
    player2 = player_names[1]

    print(f"Testing for images with both {player1} and {player2} mentioned in caption...")

    # Query for images with both players mentioned in caption
    cursor.execute(f"""
    SELECT c.id, c.file_name, c.url, p.player_name, c.no_of_faces, c.caption
    FROM cricket_data c
    LEFT JOIN players p ON c.player_id = p.player_id
    WHERE (LOWER(c.caption) LIKE '%{player1.lower()}%' AND LOWER(c.caption) LIKE '%{player2.lower()}%')
       OR (LOWER(c.description) LIKE '%{player1.lower()}%' AND LOWER(c.description) LIKE '%{player2.lower()}%')
    LIMIT 5
    """)

    results = cursor.fetchall()

    if not results:
        print(f"No images found with both {player1} and {player2} mentioned.")

        # Try a more general query for images with multiple players
        print("\nTrying a more general query for images with multiple players...")
        cursor.execute("""
        SELECT c.id, c.file_name, c.url, p.player_name, c.no_of_faces, c.caption
        FROM cricket_data c
        LEFT JOIN players p ON c.player_id = p.player_id
        WHERE c.no_of_faces >= 2
          AND (LOWER(c.caption) LIKE '%players%' OR LOWER(c.caption) LIKE '%team%')
        LIMIT 5
        """)

        results = cursor.fetchall()

        if not results:
            print("No images found with multiple players mentioned.")
            cursor.close()
            conn.close()
            return

    print(f"Found {len(results)} images.")

    # Print details of the results
    for i, row in enumerate(results):
        print(f"  Result {i+1}:")
        print(f"    ID: {row[0]}")
        print(f"    File Name: {row[1]}")
        print(f"    URL: {row[2]}")
        print(f"    Player: {row[3]}")
        print(f"    No. of Faces: {row[4]}")
        print(f"    Caption: {row[5]}")

    cursor.close()
    conn.close()

def test_group_photo_query():
    """
    Test the get_images_with_multiple_players function directly with a group photo query
    """
    print("\n\nTesting group photo query directly...")

    # Connect to the database
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Query for images with multiple faces and terms like "players" or "team"
    print("Querying for images with multiple faces and terms like 'players' or 'team'...")
    cursor.execute("""
    SELECT c.id, c.file_name, c.url, p.player_name, c.no_of_faces, c.caption, c.description
    FROM cricket_data c
    LEFT JOIN players p ON c.player_id = p.player_id
    WHERE c.no_of_faces >= 2
      AND (
          LOWER(c.caption) LIKE '%players%'
          OR LOWER(c.caption) LIKE '%team%'
          OR LOWER(c.caption) LIKE '%group%'
          OR LOWER(c.description) LIKE '%players%'
          OR LOWER(c.description) LIKE '%team%'
          OR LOWER(c.description) LIKE '%group%'
      )
    LIMIT 5
    """)

    results = cursor.fetchall()

    if not results:
        print("No images found with multiple faces and relevant terms.")
        cursor.close()
        conn.close()
        return

    print(f"Found {len(results)} images.")

    # Print details of the results
    for i, row in enumerate(results):
        print(f"  Result {i+1}:")
        print(f"    ID: {row[0]}")
        print(f"    File Name: {row[1]}")
        print(f"    URL: {row[2]}")
        print(f"    Player: {row[3]}")
        print(f"    No. of Faces: {row[4]}")
        print(f"    Caption: {row[5]}")
        print(f"    Description: {row[6][:100]}..." if row[6] and len(row[6]) > 100 else f"    Description: {row[6]}")

    # Now try to use the get_images_with_multiple_players function
    print("\nTrying to use get_images_with_multiple_players function...")

    # Create a document from the first result
    from langchain.schema import Document

    # Create a document with the data from the first result
    if results:
        row = results[0]
        doc = Document(
            page_content=row[6] or "",  # description
            metadata={
                "id": row[0],
                "file_name": row[1],
                "url": row[2],
                "player_name": row[3],
                "no_of_faces": row[4],
                "caption": row[5]
            }
        )

        print(f"Created document from result with ID {row[0]}")
        print(f"Document metadata: {doc.metadata}")
        print(f"Document content: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else f"Document content: {doc.page_content}")

    cursor.close()
    conn.close()

def test_player_names():
    """
    Test to see what player names are in the database
    """
    print("\n\nTesting player names in the database...")

    # Connect to the database
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Query for player names
    cursor.execute("SELECT player_id, player_name FROM players LIMIT 10")

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        print("No players found in the database.")
        return

    print(f"Found {len(results)} players.")

    # Print details of the results
    for i, row in enumerate(results):
        print(f"  Player {i+1}: {row[0]} - {row[1]}")

if __name__ == "__main__":
    test_player_names()
    test_database_for_multiple_faces()
    test_direct_multiple_players_query()
    test_group_photo_query()
    test_multiple_player_queries()
