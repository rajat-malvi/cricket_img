"""
Test script for query refinement when no results are found
"""

import llm_service
import query_refinement
import time

def test_no_results_query():
    """
    Test the query refinement functionality when no results are found
    """
    # Test with a query that should not return any results
    query = "show me images of cricket players on the moon"
    print(f"Testing query that should not return results: '{query}'")

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

    # Check if the response contains query suggestions
    if "You might want to try these alternative queries:" in response_text:
        print("\nQuery refinement is working correctly!")

        # Extract and print the suggested queries
        suggestions_start = response_text.find("You might want to try these alternative queries:")
        suggestions = response_text[suggestions_start:].split("\n")[1:]
        print("\nSuggested queries:")
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("\nQuery refinement is NOT working correctly!")

    print(f"\nQuery completed in {elapsed_time:.2f} seconds")
    print("-" * 50)

def test_direct_refine_query():
    """
    Test the refine_query function directly
    """
    query = "show me images of cricket players on the moon"
    print(f"Testing refine_query function directly with: '{query}'")

    # Record start time
    start_time = time.time()

    # Call the refine_query function
    refined_queries = query_refinement.refine_query(query)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print the refined queries
    print("\nRefined queries:")
    for i, refined_query in enumerate(refined_queries, 1):
        print(f"{i}. {refined_query}")

    print(f"\nRefine_query function completed in {elapsed_time:.2f} seconds")
    print("-" * 50)

def test_try_refined_queries():
    """
    Test the try_refined_queries function directly
    """
    query = "show me images of cricket players on the moon"
    print(f"Testing try_refined_queries function with: '{query}'")

    # Record start time
    start_time = time.time()

    # Call the try_refined_queries function
    result = llm_service.try_refined_queries(query)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    if result:
        successful_query, similar_images = result
        print(f"\nFound images using refined query: '{successful_query}'")
        print(f"Found {len(similar_images)} images")
    else:
        print("\nNo images found with any refined queries")

    print(f"\ntry_refined_queries function completed in {elapsed_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    print("Testing query refinement when no results are found...")

    # Test the query_images function with a query that should not return results
    test_no_results_query()

    # Test the refine_query function directly
    test_direct_refine_query()

    # Test the try_refined_queries function directly
    test_try_refined_queries()

    print("\nAll tests completed")
