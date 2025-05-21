"""
Test script to verify the query flow in the Cricket Image Chatbot
"""

import sys
import os
from typing import List, Tuple

from langchain.docstore.document import Document

from llm_service import query_images, get_images_by_sql_query
from vector_store import get_similar_images
from db_store import similarity_search, get_db_connection

def test_sql_query_first():
    """
    Test that SQL queries are tried first before falling back to similarity search
    """
    print("\n=== Testing SQL Query First Flow ===")
    
    # Test a query that should match using SQL
    query = "Show me images of Faf du Plessis batting"
    print(f"Testing query: '{query}'")
    
    # Get the results
    response_text, similar_images = query_images(query)
    
    # Print the results
    print(f"Response: {response_text}")
    print(f"Number of images found: {len(similar_images)}")
    
    # Print the first few images
    for i, (doc, score) in enumerate(similar_images[:3]):
        print(f"\nImage {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Score: {score:.4f}")
        print(f"URL: {doc.metadata.get('image_url', 'No URL')}")
        
        # Print filters used
        filters = []
        if "player_filter" in doc.metadata:
            filters.append(f"Player: {doc.metadata['player_filter']}")
        if "action_filter" in doc.metadata:
            filters.append(f"Action: {doc.metadata['action_filter']}")
        if "event_filter" in doc.metadata:
            filters.append(f"Event: {doc.metadata['event_filter']}")
        
        if filters:
            print(f"Filters: {', '.join(filters)}")

def test_general_sql_query():
    """
    Test the general SQL query function with different query types
    """
    print("\n=== Testing General SQL Query Function ===")
    
    # Test different query types
    queries = [
        "Show me images of players batting",
        "Find photos of players celebrating wickets",
        "Show me team photos",
        "Find images from evening matches",
        "Show me pictures of players with focused expressions"
    ]
    
    for query in queries:
        print(f"\nTesting query: '{query}'")
        
        # Get results directly from SQL query function
        results = get_images_by_sql_query(query)
        
        # Print results
        print(f"Number of images found: {len(results)}")
        
        if results:
            # Print the first result
            doc, score = results[0]
            print(f"First result content: {doc.page_content[:100]}...")
            print(f"Score: {score:.4f}")
            print(f"URL: {doc.metadata.get('image_url', 'No URL')}")
            
            # Print filters used
            filters = []
            if "player_filter" in doc.metadata:
                filters.append(f"Player: {doc.metadata['player_filter']}")
            if "action_filter" in doc.metadata:
                filters.append(f"Action: {doc.metadata['action_filter']}")
            if "event_filter" in doc.metadata:
                filters.append(f"Event: {doc.metadata['event_filter']}")
            if "mood_filter" in doc.metadata:
                filters.append(f"Mood: {doc.metadata['mood_filter']}")
            if "time_filter" in doc.metadata:
                filters.append(f"Time: {doc.metadata['time_filter']}")
            if doc.metadata.get("team_query", False):
                filters.append("Team query")
            
            if filters:
                print(f"Filters: {', '.join(filters)}")

def test_similarity_search_fallback():
    """
    Test the similarity search fallback when SQL queries return no results
    """
    print("\n=== Testing Similarity Search Fallback ===")
    
    # Test a query that likely won't match using SQL but might with similarity search
    query = "Show me images of cricket equipment"
    print(f"Testing query: '{query}'")
    
    # Get the results
    response_text, similar_images = query_images(query)
    
    # Print the results
    print(f"Response: {response_text}")
    print(f"Number of images found: {len(similar_images)}")
    
    # Print the first few images
    for i, (doc, score) in enumerate(similar_images[:3]):
        print(f"\nImage {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Score: {score:.4f}")
        print(f"URL: {doc.metadata.get('image_url', 'No URL')}")

def test_result_counts():
    """
    Test that the correct number of results are returned
    """
    print("\n=== Testing Result Counts ===")
    
    # Test SQL query result count
    query = "Show me images of players"
    print(f"Testing SQL query: '{query}'")
    sql_results = get_images_by_sql_query(query)
    print(f"SQL query returned {len(sql_results)} results")
    
    # Test similarity search result count
    query = "cricket"
    print(f"\nTesting similarity search: '{query}'")
    from vector_store import get_embeddings_model
    embeddings_model = get_embeddings_model()
    query_embedding = embeddings_model.embed_query(query)
    similarity_results = similarity_search(query_embedding, k=5, query_text=query)
    print(f"Similarity search returned {len(similarity_results)} results")

def main():
    """
    Run all tests
    """
    # Check if database is initialized
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        if count == 0:
            print("Error: Database is empty. Please run 'python init_db.py' first.")
            sys.exit(1)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please make sure the database is properly set up.")
        sys.exit(1)
    
    # Run the tests
    test_sql_query_first()
    test_general_sql_query()
    test_similarity_search_fallback()
    test_result_counts()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
