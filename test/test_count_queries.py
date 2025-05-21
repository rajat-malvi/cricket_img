import llm_service

def test_count_queries():
    """Test count queries for press meets and promotional events"""
    
    # Test press meet query
    press_query = "how many players attend the press meet"
    response_text, images = llm_service.query_images(press_query)
    print(f"Query: {press_query}")
    print(f"Response: {response_text}")
    print()
    
    # Test promotional event query
    promo_query = "how many players attend the promotional event"
    response_text, images = llm_service.query_images(promo_query)
    print(f"Query: {promo_query}")
    print(f"Response: {response_text}")
    print()
    
    # Test general count query
    general_query = "how many cricket images are there"
    response_text, images = llm_service.query_images(general_query)
    print(f"Query: {general_query}")
    print(f"Response: {response_text}")

if __name__ == "__main__":
    test_count_queries()
