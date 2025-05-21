"""
Test script for query_images function to verify SQL vs embedding search
"""

import llm_service
import time

def test_query(query, query_type):
    """
    Test a query and print the results

    Args:
        query (str): The query to test
        query_type (str): Type of query (player or regular)
    """
    print(f"Testing {query_type} query: '{query}'")

    # Record start time
    start_time = time.time()

    # Call the query_images function
    response_text, similar_images = llm_service.query_images(query)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print the response text
    print("\nResponse text:")
    print(response_text)

    # Print the similar images
    print(f"\nSimilar images (found {len(similar_images)} in {elapsed_time:.2f} seconds):")
    if similar_images:
        for i, (doc, score) in enumerate(similar_images, 1):
            if i > 3:  # Only show first 3 images for brevity
                print(f"\n... and {len(similar_images) - 3} more images")
                break

            print(f"\nImage {i}:")
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:100]}...")

            # Print metadata
            print("Metadata:")
            for key, value in doc.metadata.items():
                if key in ['image_url', 'url', 'player_name', 'action_name', 'event_name']:
                    print(f"  {key}: {value}")
    else:
        print("No similar images found")

    print(f"\n{query_type.capitalize()} query test completed in {elapsed_time:.2f} seconds")
    print("-" * 50)

def test_player_query():
    """Test a player name query (should use SQL)"""
    query = "Show me images of Faf du Plessis"
    test_query(query, "player")

def test_regular_query():
    """Test a regular query (should use embedding similarity)"""
    query = "Show me cricket images with players celebrating"
    test_query(query, "regular")

if __name__ == "__main__":
    print("Starting tests to verify SQL vs embedding search...")

    # Test a player query (should use SQL)
    test_player_query()

    print("\n")

    # Test a regular query (should use embedding similarity)
    test_regular_query()

    print("\nAll tests completed")
