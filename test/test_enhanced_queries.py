import llm_service

def test_query_refinement():
    """Test query refinement for different types of queries"""
    
    # Test press meet query with different wording
    press_query = "how many players attend the press meet"
    response_text, images = llm_service.query_images(press_query)
    print(f"Query: {press_query}")
    print(f"Response: {response_text}")
    print()
    
    # Test press meet query with synonyms
    press_query_alt = "how many players were at the media briefing"
    response_text, images = llm_service.query_images(press_query_alt)
    print(f"Query: {press_query_alt}")
    print(f"Response: {response_text}")
    print()
    
    # Test promotional event query with different wording
    promo_query = "how many players attend the promotional event"
    response_text, images = llm_service.query_images(promo_query)
    print(f"Query: {promo_query}")
    print(f"Response: {response_text}")
    print()
    
    # Test promotional event query with synonyms
    promo_query_alt = "how many players were at the marketing campaign"
    response_text, images = llm_service.query_images(promo_query_alt)
    print(f"Query: {promo_query_alt}")
    print(f"Response: {response_text}")
    print()
    
    # Test action query with stemming
    action_query = "show me players batting"
    response_text, images = llm_service.query_images(action_query)
    print(f"Query: {action_query}")
    print(f"Response: {response_text}")
    print(f"Number of images: {len(images)}")
    print()
    
    # Test action query with synonyms
    action_query_alt = "show me players hitting the ball"
    response_text, images = llm_service.query_images(action_query_alt)
    print(f"Query: {action_query_alt}")
    print(f"Response: {response_text}")
    print(f"Number of images: {len(images)}")
    print()

if __name__ == "__main__":
    test_query_refinement()
