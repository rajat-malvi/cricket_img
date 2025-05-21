"""
Test script to verify handling of different question types in the Cricket Image Chatbot
"""

import sys
from llm_service import query_images, identify_question_type, get_count_from_db, get_info_from_db

def test_question_type_identification():
    """
    Test the question type identification function
    """
    print("\n=== Testing Question Type Identification ===")
    
    # Test different question types
    questions = {
        "Show me images of Faf du Plessis": "image_request",
        "How many images of players batting are there?": "count_request",
        "Who is Faf du Plessis?": "info_request",
        "Tell me about cricket rules": "info_request",
        "What is the weather like today?": "info_request",
        "Hello, how are you?": "general"
    }
    
    for question, expected_type in questions.items():
        identified_type = identify_question_type(question)
        result = "✓" if identified_type == expected_type else "✗"
        print(f"{result} '{question}' -> {identified_type} (expected: {expected_type})")

def test_count_requests():
    """
    Test handling of count requests
    """
    print("\n=== Testing Count Requests ===")
    
    # Test different count requests
    count_questions = [
        "How many images of Faf du Plessis are there?",
        "Count the number of batting images",
        "What is the total number of images in the database?",
        "How many pictures show players celebrating?"
    ]
    
    for question in count_questions:
        print(f"\nTesting: '{question}'")
        
        # Get count directly
        count = get_count_from_db(question)
        print(f"Count from database: {count}")
        
        # Get response from query_images
        response, images = query_images(question)
        print(f"Response: {response}")
        print(f"Images returned: {len(images)}")

def test_info_requests():
    """
    Test handling of information requests
    """
    print("\n=== Testing Information Requests ===")
    
    # Test different information requests
    info_questions = [
        "Who is Faf du Plessis?",
        "Tell me about Jonny Bairstow",
        "What team does Maheesh Theekshana play for?",
        "Tell me about cricket rules"
    ]
    
    for question in info_questions:
        print(f"\nTesting: '{question}'")
        
        # Get info directly
        info = get_info_from_db(question)
        print(f"Info from database: {info}")
        
        # Get response from query_images
        response, images = query_images(question)
        print(f"Response: {response}")
        print(f"Images returned: {len(images)}")

def test_image_requests():
    """
    Test handling of image requests
    """
    print("\n=== Testing Image Requests ===")
    
    # Test different image requests
    image_questions = [
        "Show me images of Faf du Plessis",
        "Find photos of players celebrating",
        "Display pictures of cricket matches",
        "Show me team photos"
    ]
    
    for question in image_questions:
        print(f"\nTesting: '{question}'")
        
        # Get response from query_images
        response, images = query_images(question)
        print(f"Response: {response}")
        print(f"Images returned: {len(images)}")
        
        # Print the first image if available
        if images:
            doc, score = images[0]
            print(f"First image content: {doc.page_content[:100]}...")
            print(f"URL: {doc.metadata.get('image_url', 'No URL')}")

def test_general_questions():
    """
    Test handling of general questions
    """
    print("\n=== Testing General Questions ===")
    
    # Test different general questions
    general_questions = [
        "Hello, how are you?",
        "What is cricket?",
        "Tell me about Joburg Super Kings",
        "What is the IPL?"
    ]
    
    for question in general_questions:
        print(f"\nTesting: '{question}'")
        
        # Get response from query_images
        response, images = query_images(question)
        print(f"Response: {response}")
        print(f"Images returned: {len(images)}")

def main():
    """
    Run all tests
    """
    # Run the tests
    test_question_type_identification()
    test_count_requests()
    test_info_requests()
    test_image_requests()
    test_general_questions()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
