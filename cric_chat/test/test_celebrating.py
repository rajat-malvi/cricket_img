"""
Test script for the "celebrating images" query
"""

import llm_service
import time

def test_celebrating_query():
    """
    Test the "celebrating images" query
    """
    query = "celebrating images"
    print(f"Testing query: '{query}'")
    
    # Record start time
    start_time = time.time()
    
    # Call the query_images function
    response_text, similar_images = llm_service.query_images(query)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print the response text
    print("\nResponse text:")
    print(response_text)
    
    # Print the number of similar images
    print(f"\nNumber of similar images: {len(similar_images)}")
    
    # Print the similar images
    print("\nSimilar images:")
    if similar_images:
        for i, (doc, score) in enumerate(similar_images, 1):
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
    
    print(f"\nQuery completed in {elapsed_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    print("Testing 'celebrating images' query...")
    test_celebrating_query()
    print("\nTest completed")
