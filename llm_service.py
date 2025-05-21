"""
LLM service module for the Cricket Image Chatbot
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional
from langchain.docstore.document import Document

import config
import db_store
import query_refinement
from vector_store import get_similar_images
from groq_service import GroqAPI

# Initialize the LLM service
groq_api = GroqAPI()

def query_images(query: str, force_similarity: bool = False) -> Tuple[str, List[Tuple[Document, float]], bool]:
    """
    Process a natural language query and return appropriate response

    Args:
        query (str): Natural language query
        force_similarity (bool): Force using similarity search even if SQL would be used normally

    Returns:
        Tuple[str, List[Tuple[Document, float]], bool]: Tuple of (response_text, similar_images, used_similarity)
    """
    print(f"Processing query: '{query}'")

    # Step 1: Classify the query type
    query_type = classify_query_type(query)
    print(f"Query classified as: {query_type}")

    # Flag to track if similarity search was used
    used_similarity = False

    # If force_similarity is True, skip SQL queries and go straight to vector search
    if force_similarity:
        print("Forcing similarity search as requested...")
        similar_images = get_similar_images(query, k=0, similarity_threshold=0.4)
        used_similarity = True  # Mark that similarity search was used

        # Step 5: Generate appropriate response based on query type
        if query_type == "descriptive":
            response_text = generate_descriptive_response(query, similar_images)
        else:  # image query
            response_text = generate_response_text(query, similar_images)

        return response_text, similar_images, used_similarity

    # Step 2: Handle special query types
    # Check if this is a team photos query
    # if is_team_photos_query(query):
    #     return "No cricket images matching team photos from practice sessions are available.", [], False

    # Check if this is a practice images query
    if is_practice_images_query(query):
        response_text, images, used_sim = handle_practice_images_query(query)
        return response_text, images, used_sim  # Practice images use direct SQL queries

    # Step 3: Handle query based on its type
    if query_type == "counting":
        response_text, images, used_sim = handle_counting_query(query)
        return response_text, images, used_sim  # Counting queries use direct SQL
    elif query_type == "tabular":
        response_text, images, used_sim = handle_tabular_query(query)
        return response_text, images, used_sim  # Tabular queries use direct SQL

    # Step 4: For image and descriptive queries, retrieve relevant data
    # Try to get results using direct SQL queries first
    print("Attempting to retrieve results using SQL queries...")
    # Remove the limit (k) to get all matching images
    similar_images = get_images_by_sql_query(query, k=0)

    # If SQL queries didn't yield results, try vector similarity search
    if not similar_images:
        print("No results from SQL queries, trying vector similarity search...")
        # Remove the limit (k) to get all matching images with similarity above threshold
        similar_images = get_similar_images(query, k=0, similarity_threshold=0.4)
        used_similarity = True  # Mark that similarity search was used

    # If still no results, try query refinement
    if not similar_images:
        print(f"No results found for query: '{query}'. Trying query refinement...")
        refined_result = try_refined_queries(query)

        if refined_result:
            successful_query, similar_images, is_similarity = refined_result
            print(f"Found results using refined query: '{successful_query}'")
            used_similarity = is_similarity  # Update based on refinement method
        else:
            print("No results found even after query refinement")

    # Step 5: Generate appropriate response based on query type
    if query_type == "descriptive":
        response_text = generate_descriptive_response(query, similar_images)
    else:  # image query
        response_text = generate_response_text(query, similar_images)

    return response_text, similar_images, used_similarity

def classify_query_type(query: str) -> str:
    """
    Classify the query type

    Args:
        query (str): Query text

    Returns:
        str: Query type ("counting", "tabular", "descriptive", or "image")
    """
    if is_counting_query(query):
        return "counting"
    elif is_tabular_query(query):
        return "tabular"
    elif is_descriptive_query(query):
        return "descriptive"
    else:
        return "image"  # Default to image query

def is_counting_query(query: str) -> bool:
    """
    Check if the query is asking for a count

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for a count, False otherwise
    """
    query_lower = query.lower()
    count_patterns = [
        r"how many",
        r"count of",
        r"number of",
        r"total number",
        r"total count",
        r"count",
        r"tally",
        r"quantity",
        r"sum"
    ]

    return any(re.search(pattern, query_lower) for pattern in count_patterns)

def is_tabular_query(query: str) -> bool:
    """
    Check if the query is asking for tabular data

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for tabular data, False otherwise
    """
    query_lower = query.lower()
    tabular_patterns = [
        r"table",
        r"list all",
        r"show all",
        r"display all",
        r"summarize",
        r"summary of",
        r"statistics",
        r"stats",
        r"breakdown",
        r"compare",
        r"comparison"
    ]

    return any(re.search(pattern, query_lower) for pattern in tabular_patterns)

def is_descriptive_query(query: str) -> bool:
    """
    Check if the query is asking for descriptive information

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for descriptive information, False otherwise
    """
    query_lower = query.lower()
    descriptive_patterns = [
        r"describe",
        r"explain",
        r"tell me about",
        r"what is",
        r"who is",
        r"when",
        r"where",
        r"why",
        r"how does",
        r"details about",
        r"information on"
    ]

    # If it's not a counting query or image query, it's likely a descriptive query
    return (any(re.search(pattern, query_lower) for pattern in descriptive_patterns) or
            (not is_counting_query(query_lower) and not is_image_query(query_lower)))

def is_image_query(query: str) -> bool:
    """
    Check if the query is asking for images

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for images, False otherwise
    """
    query_lower = query.lower()
    image_patterns = [
        r"show",
        r"display",
        r"image",
        r"picture",
        r"photo",
        r"see",
        r"view",
        r"look at"
    ]

    # If it contains image-related terms and doesn't ask for counts
    return any(re.search(pattern, query_lower) for pattern in image_patterns) and not is_counting_query(query_lower)

def is_team_photos_query(query: str) -> bool:
    """
    Check if the query is asking for team photos from practice sessions

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for team photos from practice sessions, False otherwise
    """
    query_lower = query.lower()
    team_patterns = [
        r"team photo",
        r"team picture",
        r"team image",
        r"group photo",
        r"group picture",
        r"group image"
    ]

    practice_patterns = [
        r"practice",
        r"training",
        r"net session",
        r"nets"
    ]

    has_team = any(re.search(pattern, query_lower) for pattern in team_patterns)
    has_practice = any(re.search(pattern, query_lower) for pattern in practice_patterns)

    return has_team and has_practice

def is_practice_images_query(query: str) -> bool:
    """
    Check if the query is asking for practice images of players

    Args:
        query (str): Query text

    Returns:
        bool: True if the query is asking for practice images of players, False otherwise
    """
    query_lower = query.lower()
    practice_patterns = [
        r"practice image",
        r"practice picture",
        r"practice photo",
        r"training image",
        r"training picture",
        r"training photo"
    ]

    player_patterns = [
        r"player",
        r"cricketer",
        r"batsman",
        r"bowler",
        r"all-rounder",
        r"all rounder"
    ]

    has_practice = any(re.search(pattern, query_lower) for pattern in practice_patterns)
    has_player = any(re.search(pattern, query_lower) for pattern in player_patterns)

    return has_practice and has_player

def handle_counting_query(query: str) -> Tuple[str, List[Tuple[Document, float]], bool]:
    """
    Handle a counting query

    Args:
        query (str): Query text

    Returns:
        Tuple[str, List[Tuple[Document, float]], bool]: Tuple of (response_text, similar_images, used_similarity)
    """
    query_lower = query.lower()

    # Check for player-specific count
    if db_store.is_player_query(query_lower):
        # Extract player name
        conn = db_store.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, player_name FROM players")
        players = {row[1].lower(): row[0] for row in cursor.fetchall()}

        player_id = None
        player_name = None
        for name in players.keys():
            if name in query_lower:
                player_id = players[name]
                player_name = name
                break

        if player_id:
            # Count images for this player
            cursor.execute("SELECT COUNT(*) FROM cricket_data WHERE player_id = %s", (player_id,))
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return f"There are {count} images of {player_name.title()} in the database.", [], False

    # Check for action-specific count
    action_terms = ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]
    for action in action_terms:
        if action in query_lower:
            # Get action ID
            conn = db_store.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT action_id FROM action WHERE LOWER(action_name) LIKE %s", (f"%{action.lower()}%",))
            action_ids = cursor.fetchall()

            if action_ids:
                # Build query with all matching action IDs
                action_id_list = [f"'{row[0]}'" for row in action_ids]
                action_id_clause = f"action_id IN ({', '.join(action_id_list)})"

                # Count images for this action
                cursor.execute(f"SELECT COUNT(*) FROM cricket_data WHERE {action_id_clause}")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                return f"There are {count} images of players {action} in the database.", [], False

    # Check for event-specific count
    if "press meet" in query_lower or "press conference" in query_lower or "media" in query_lower:
        count = db_store.get_count_from_db("press_meet")
        response = f"There are {count} images from press meets or media interactions in the database."
    elif "practice" in query_lower or "training" in query_lower or "net session" in query_lower:
        count = db_store.get_count_from_db("practice")
        response = f"There are {count} images from practice or training sessions in the database."
    elif "match" in query_lower or "game" in query_lower or "fixture" in query_lower:
        count = db_store.get_count_from_db("match")
        response = f"There are {count} images from matches in the database."
    elif "promotional" in query_lower or "promotion" in query_lower or "marketing" in query_lower:
        count = db_store.get_count_from_db("promotional")
        response = f"There are {count} images from promotional events in the database."

    # Check for mood-specific count
    mood_terms = ["happy", "serious", "celebrating", "smiling", "intense"]
    for mood in mood_terms:
        if mood in query_lower:
            # Get mood ID
            conn = db_store.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT mood_id FROM mood WHERE LOWER(mood_name) LIKE %s", (f"%{mood.lower()}%",))
            mood_ids = cursor.fetchall()

            if mood_ids:
                # Build query with all matching mood IDs
                mood_id_list = [f"'{row[0]}'" for row in mood_ids]
                mood_id_clause = f"mood_id IN ({', '.join(mood_id_list)})"

                # Count images for this mood
                cursor.execute(f"SELECT COUNT(*) FROM cricket_data WHERE {mood_id_clause}")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                return f"There are {count} images of players with {mood} mood in the database.", [], False

    # Check for location-specific count
    location_terms = ["stadium", "field", "nets", "dressing room", "press room"]
    for location in location_terms:
        if location in query_lower:
            # Try to find sublocation first
            conn = db_store.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT sublocation_id FROM sublocation WHERE LOWER(sublocation_name) LIKE %s", (f"%{location.lower()}%",))
            sublocation_ids = cursor.fetchall()

            if sublocation_ids:
                # Build query with all matching sublocation IDs
                sublocation_id_list = [f"'{row[0]}'" for row in sublocation_ids]
                location_clause = f"sublocation_id IN ({', '.join(sublocation_id_list)})"

                # Count images for this location
                cursor.execute(f"SELECT COUNT(*) FROM cricket_data WHERE {location_clause}")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                return f"There are {count} images from {location} in the database.", [], False
            else:
                # Try to match against location field
                cursor.execute(f"SELECT COUNT(*) FROM cricket_data WHERE LOWER(location) LIKE '%{location.lower()}%'")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                return f"There are {count} images from {location} in the database.", [], False

    # Default: total count
    count = db_store.get_count_from_db("total")
    response = f"There are a total of {count} cricket images in the database."

    return response, [], False

def handle_tabular_query(query: str) -> Tuple[str, List[Tuple[Document, float]], bool]:
    """
    Handle a tabular query

    Args:
        query (str): Query text

    Returns:
        Tuple[str, List[Tuple[Document, float]], bool]: Tuple of (response_text, similar_images, used_similarity)
    """
    query_lower = query.lower()
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Check for player statistics
    if "player" in query_lower and ("stats" in query_lower or "statistics" in query_lower or "breakdown" in query_lower):
        # Get player image counts
        cursor.execute("""
        SELECT p.player_name, COUNT(c.id) as image_count
        FROM players p
        LEFT JOIN cricket_data c ON p.player_id = c.player_id
        GROUP BY p.player_name
        ORDER BY image_count DESC
        """)

        results = cursor.fetchall()

        # Format as a table
        table = "| Player Name | Image Count |\n|------------|------------|\n"
        for player_name, count in results:
            table += f"| {player_name} | {count} |\n"

        response = f"Here's a breakdown of images by player:\n\n{table}"
        return response, [], False

    # Check for event statistics
    elif "event" in query_lower and ("stats" in query_lower or "statistics" in query_lower or "breakdown" in query_lower):
        # Get event image counts
        cursor.execute("""
        SELECT e.event_name, COUNT(c.id) as image_count
        FROM event e
        LEFT JOIN cricket_data c ON e.event_id = c.event_id
        GROUP BY e.event_name
        ORDER BY image_count DESC
        """)

        results = cursor.fetchall()

        # Format as a table
        table = "| Event Type | Image Count |\n|------------|------------|\n"
        for event_name, count in results:
            table += f"| {event_name} | {count} |\n"

        response = f"Here's a breakdown of images by event type:\n\n{table}"
        return response, [], False

    # Check for action statistics
    elif "action" in query_lower and ("stats" in query_lower or "statistics" in query_lower or "breakdown" in query_lower):
        # Get action image counts
        cursor.execute("""
        SELECT a.action_name, COUNT(c.id) as image_count
        FROM action a
        LEFT JOIN cricket_data c ON a.action_id = c.action_id
        GROUP BY a.action_name
        ORDER BY image_count DESC
        """)

        results = cursor.fetchall()

        # Format as a table
        table = "| Action Type | Image Count |\n|------------|------------|\n"
        for action_name, count in results:
            table += f"| {action_name} | {count} |\n"

        response = f"Here's a breakdown of images by action type:\n\n{table}"
        return response, [], False

    # Check for mood statistics
    elif "mood" in query_lower and ("stats" in query_lower or "statistics" in query_lower or "breakdown" in query_lower):
        # Get mood image counts
        cursor.execute("""
        SELECT m.mood_name, COUNT(c.id) as image_count
        FROM mood m
        LEFT JOIN cricket_data c ON m.mood_id = c.mood_id
        GROUP BY m.mood_name
        ORDER BY image_count DESC
        """)

        results = cursor.fetchall()

        # Format as a table
        table = "| Mood Type | Image Count |\n|------------|------------|\n"
        for mood_name, count in results:
            table += f"| {mood_name} | {count} |\n"

        response = f"Here's a breakdown of images by mood type:\n\n{table}"
        return response, [], False

    # Check for location statistics
    elif "location" in query_lower and ("stats" in query_lower or "statistics" in query_lower or "breakdown" in query_lower):
        # Get location image counts
        cursor.execute("""
        SELECT s.sublocation_name, COUNT(c.id) as image_count
        FROM sublocation s
        LEFT JOIN cricket_data c ON s.sublocation_id = c.sublocation_id
        GROUP BY s.sublocation_name
        ORDER BY image_count DESC
        """)

        results = cursor.fetchall()

        # Format as a table
        table = "| Location | Image Count |\n|------------|------------|\n"
        for location_name, count in results:
            table += f"| {location_name} | {count} |\n"

        response = f"Here's a breakdown of images by location:\n\n{table}"
        return response, [], False

    # Default: general statistics
    else:
        # Get general statistics
        cursor.execute("SELECT COUNT(*) FROM cricket_data")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT player_id) FROM cricket_data")
        player_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT event_id) FROM cricket_data")
        event_count = cursor.fetchone()[0]

        # Format as a table
        table = "| Statistic | Count |\n|------------|------------|\n"
        table += f"| Total Images | {total_count} |\n"
        table += f"| Players Featured | {player_count} |\n"
        table += f"| Event Types | {event_count} |\n"

        response = f"Here are the general statistics for the cricket image database:\n\n{table}"
        return response, [], False

def handle_practice_images_query(query: str) -> Tuple[str, List[Tuple[Document, float]], bool]:
    """
    Handle a query for practice images of players

    Args:
        query (str): Query text

    Returns:
        Tuple[str, List[Tuple[Document, float]], bool]: Tuple of (response_text, similar_images, used_similarity)
    """
    # Get practice images (limit to 5)
    similar_images = db_store.get_images_by_practice(5)

    if similar_images:
        response = f"Here are some practice images of players. I found {len(similar_images)} images that match your query."
    else:
        response = "I couldn't find any practice images of players in the database."

    # Practice images use direct SQL queries, not similarity search
    return response, similar_images, False

def get_images_by_sql_query(query: str, k: int = 0) -> List[Tuple[Document, float]]:
    """
    Get images by SQL query based on the query text.
    This function tries various SQL query strategies before falling back to vector search.

    Args:
        query (str): Query text
        k (int): Number of results to return (default: 0, which means return all results)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    query_lower = query.lower()
    results = []

    # Check for group photo queries first
    group_photo_terms = ["group photo", "team photo", "players together", "group of players", "multiple players"]
    if any(term in query_lower for term in group_photo_terms):
        print(f"Detected group photo query: '{query}'")
        results = db_store.get_images_with_multiple_players(query, k)
        if results:
            print(f"Found {len(results)} results using group photo SQL query")
            return results

    # 1. Check for player name queries (including multiple players and player+action combinations)
    if db_store.is_player_query(query):
        print(f"Detected player query: '{query}'")

        # Check for multiple players
        if " and " in query_lower or "&" in query_lower or "together" in query_lower or "same frame" in query_lower or "single frame" in query_lower:
            print("Detected multiple players in query")
            # Try to get images with multiple players
            results = db_store.get_images_with_multiple_players(query, k)
            if results:
                print(f"Found {len(results)} results using multiple players SQL query")
                return results
            else:
                print("No results found with direct multiple players query, trying with similarity search")
                # If no results, try similarity search immediately for multiple players queries
                similar_images = get_similar_images(query, k=0, similarity_threshold=0.3)  # Lower threshold for better recall
                if similar_images:
                    # Filter for images with at least 2 faces
                    filtered_images = [(doc, score) for doc, score in similar_images
                                      if doc.metadata.get('no_of_faces', 0) is not None
                                      and int(doc.metadata.get('no_of_faces', 0)) >= 2]
                    if filtered_images:
                        print(f"Found {len(filtered_images)} results using similarity search with face filtering")
                        return filtered_images

        # Single player query
        results = db_store.get_images_by_player_name(query, k)
        if results:
            print(f"Found {len(results)} results using player name SQL query")
            return results

    # 2. Check for event type queries
    # Press meet queries
    if db_store.is_press_meet_query(query):
        print(f"Detected press meet query: '{query}'")
        results = db_store.get_images_by_press_meet(k)
        if results:
            print(f"Found {len(results)} results using press meet SQL query")
            return results

    # Practice queries
    if db_store.is_practice_query(query):
        print(f"Detected practice query: '{query}'")
        results = db_store.get_images_by_practice(k)
        if results:
            print(f"Found {len(results)} results using practice SQL query")
            return results

    # 3. Check for action type queries (batting, bowling, etc.)
    action_terms = ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]
    for action in action_terms:
        if action in query_lower:
            print(f"Detected action query for '{action}'")
            results = db_store.get_images_by_action(action, k)
            if results:
                print(f"Found {len(results)} results using action SQL query")
                return results

    # 4. Check for mood type queries (happy, serious, etc.)
    mood_terms = ["happy", "serious", "celebrating", "smiling", "intense"]
    for mood in mood_terms:
        if mood in query_lower:
            print(f"Detected mood query for '{mood}'")
            results = db_store.get_images_by_mood(mood, k)
            if results:
                print(f"Found {len(results)} results using mood SQL query")
                return results

    # 5. Check for location type queries
    location_terms = ["stadium", "field", "nets", "dressing room", "press room"]
    for location in location_terms:
        if location in query_lower:
            print(f"Detected location query for '{location}'")
            results = db_store.get_images_by_location(location, k)
            if results:
                print(f"Found {len(results)} results using location SQL query")
                return results

    # 6. Check for activity queries (traveling, celebrating, etc.)
    activity_terms = ["traveling", "travel", "journey", "celebrating", "training",
                     "meeting fans", "press conference", "team huddle", "eating", "relaxing"]
    if any(term in query_lower for term in activity_terms):
        print(f"Detected activity query: '{query}'")
        results = db_store.get_images_by_activity(query, k)
        if results:
            print(f"Found {len(results)} results using activity SQL query")
            return results

    # 6. Try a general keyword search in captions and descriptions
    # Extract key nouns and adjectives from the query
    import nltk
    try:
        # Make sure we have the required NLTK resources
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading required NLTK resource 'averaged_perceptron_tagger'...")
            nltk.download('averaged_perceptron_tagger')

        tokens = nltk.word_tokenize(query_lower)
        tagged = nltk.pos_tag(tokens)

        # Extract nouns and adjectives
        keywords = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('JJ')]

        if keywords:
            print(f"Extracted keywords for SQL search: {keywords}")
            results = db_store.get_images_by_keywords(keywords, k)
            if results:
                print(f"Found {len(results)} results using keyword SQL query")
                return results
    except Exception as e:
        print(f"Error in keyword extraction: {e}")

    # No SQL results found
    print(f"No results found using SQL queries for: '{query}'")
    return []

def try_refined_queries(query: str) -> Optional[Tuple[str, List[Tuple[Document, float]], bool]]:
    """
    Try refined queries when no results are found

    Args:
        query (str): Original query

    Returns:
        Optional[Tuple[str, List[Tuple[Document, float]], bool]]: Tuple of (successful_query, similar_images, used_similarity) or None if no results found
    """
    print(f"Refining query: '{query}'")

    # Generate refined queries using our comprehensive refine_query function
    refined_queries = query_refinement.refine_query(query)

    print(f"Generated {len(refined_queries)} refined queries")

    # Try each refined query
    for refined_query in refined_queries:
        if refined_query == query:
            continue  # Skip the original query

        print(f"Trying refined query: '{refined_query}'")

        # First try SQL queries - no limit on results
        similar_images = get_images_by_sql_query(refined_query, k=0)

        if similar_images:
            print(f"Found {len(similar_images)} results using refined SQL query: '{refined_query}'")
            return refined_query, similar_images, False  # SQL query was used, not similarity

        # If SQL queries didn't yield results, try vector similarity search - no limit on results
        similar_images = get_similar_images(refined_query, k=0, similarity_threshold=0.4)

        if similar_images:
            print(f"Found {len(similar_images)} results using refined vector search query: '{refined_query}'")
            return refined_query, similar_images, True  # Similarity search was used

    # If no results found with any refined query
    print("No results found with any refined queries")
    return None

def generate_response_text(query: str, similar_images: List[Tuple[Document, float]]) -> str:
    """
    Generate response text for image queries based on the query and similar images

    Args:
        query (str): Query text
        similar_images (List[Tuple[Document, float]]): List of (document, similarity_score) tuples

    Returns:
        str: Response text with all image URLs
    """
    if not similar_images:
        return f"Please try a different search term for cricket images related to '{query}'."

    # Check if this is a query for multiple players together
    query_lower = query.lower()
    is_multiple_players_query = False
    is_fans_interaction_query = False

    # Check for multiple player indicators
    multiple_player_terms = ["and", "&", "with", "together", "same frame", "single frame", "standing together"]
    if any(term in query_lower for term in multiple_player_terms):
        # Check if we have player names in the query
        from db_store import is_player_query
        if is_player_query(query_lower):
            is_multiple_players_query = True

    # Check if this is a query about players interacting with fans
    fan_terms = ["fan", "fans", "supporter", "supporters", "crowd", "audience", "spectator", "spectators", "interacting", "interaction"]
    if any(term in query_lower for term in fan_terms):
        is_fans_interaction_query = True

    # Prepare prompt for the LLM
    prompt = f"""
    User Question: {query}

    I found {len(similar_images)} cricket images that match the query. Here are some details about the images:

    """

    # Add details about a sample of images for context (for the LLM's understanding)
    # We'll include all URLs in the final response
    sample_size = min(5, len(similar_images))
    for i, (doc, _) in enumerate(similar_images[:sample_size]):
        prompt += f"Image {i+1}:\n"
        prompt += f"- Player: {doc.metadata.get('player_name', 'Unknown')}\n"
        prompt += f"- Event: {doc.metadata.get('event_name', 'Unknown')}\n"
        prompt += f"- Action: {doc.metadata.get('action_name', 'Unknown')}\n"
        prompt += f"- Mood: {doc.metadata.get('mood_name', 'Unknown')}\n"
        prompt += f"- Location: {doc.metadata.get('sublocation_name', 'Unknown')}\n"
        prompt += f"- Caption: {doc.metadata.get('caption', 'No caption')}\n"
        prompt += f"- Number of faces: {doc.metadata.get('no_of_faces', 'Unknown')}\n\n"

    # Add specific instructions for multiple player queries
    if is_multiple_players_query:
        prompt += f"""
        This is a query for multiple players together in the same image.
        Please emphasize in your response that these images show the players together.
        Mention the number of faces detected in the images if available.
        """

    # Add specific instructions for fan interaction queries
    if is_fans_interaction_query:
        prompt += f"""
        This is a query about players interacting with fans.
        Please provide a very concise response confirming that you have such images.
        DO NOT list the URLs in your response - the images will be displayed separately.
        """

    prompt += f"""
    Please provide a VERY concise response to the user's query based on these images.
    Focus on answering the query directly without mentioning similarity scores or technical details.
    Keep the response extremely brief (1-2 sentences) and conversational, especially for image queries.
    For image queries, prioritize showing the images over providing lengthy text explanations.
    DO NOT include any URLs in your response - the images will be displayed separately.
    """

    # Generate response using the LLM
    llm_response = groq_api.generate(prompt)

    # Filter out images with only 1 face for "together" queries
    filtered_images = similar_images
    if is_multiple_players_query:
        filtered_images = [(doc, score) for doc, score in similar_images
                          if doc.metadata.get('no_of_faces', 0) is not None
                          and int(doc.metadata.get('no_of_faces', 0)) >= 2]

        # If filtering removed all images, provide a specific message
        if not filtered_images and similar_images:
            return f"Here are images related to {query}. For images with multiple players in the same frame, please try a more specific query."

    # For fan interaction queries, we don't need to add URLs to the response
    # as they will be displayed by the display_similar_images function
    if is_fans_interaction_query:
        return llm_response

    # For other queries, add a note that images will be displayed below
    return llm_response

def generate_descriptive_response(query: str, similar_images: List[Tuple[Document, float]]) -> str:
    """
    Generate comprehensive descriptive response text based on the query and similar images

    Args:
        query (str): Query text
        similar_images (List[Tuple[Document, float]]): List of (document, similarity_score) tuples

    Returns:
        str: Comprehensive response text with relevant information and optional image URLs
    """
    if not similar_images:
        return f"Please try a different search term for information related to '{query}'."

    # Extract entity type from query
    entity_type = "general"
    query_lower = query.lower()

    if db_store.is_player_query(query_lower):
        entity_type = "player"
    elif any(term in query_lower for term in ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]):
        entity_type = "action"
    elif any(term in query_lower for term in ["press meet", "practice", "match", "promotional"]):
        entity_type = "event"
    elif any(term in query_lower for term in ["happy", "serious", "celebrating", "smiling", "intense"]):
        entity_type = "mood"
    elif any(term in query_lower for term in ["stadium", "field", "nets", "dressing room", "press room"]):
        entity_type = "location"

    # Get additional database statistics for context
    conn = db_store.get_db_connection()
    cursor = conn.cursor()

    # Get entity-specific statistics
    stats_info = ""
    if entity_type == "player" and db_store.is_player_query(query_lower):
        # Find which player is being queried
        cursor.execute("SELECT player_id, player_name FROM players")
        players = {row[1].lower(): (row[0], row[1]) for row in cursor.fetchall()}

        player_id = None
        player_name = None
        for name, (pid, original_name) in players.items():
            if name in query_lower:
                player_id = pid
                player_name = original_name
                break

        if player_id:
            # Get player statistics
            cursor.execute("""
            SELECT
                COUNT(*) as total_images,
                COUNT(DISTINCT e.event_id) as event_count,
                COUNT(DISTINCT a.action_id) as action_count,
                STRING_AGG(DISTINCT e.event_name, ', ') as events,
                STRING_AGG(DISTINCT a.action_name, ', ') as actions
            FROM cricket_data c
            LEFT JOIN event e ON c.event_id = e.event_id
            LEFT JOIN action a ON c.action_id = a.action_id
            WHERE c.player_id = %s
            """, (player_id,))

            stats = cursor.fetchone()
            if stats:
                stats_info = f"""
                Player Statistics for {player_name}:
                - Total Images: {stats[0]}
                - Events Participated: {stats[1]} ({stats[3]})
                - Actions Performed: {stats[2]} ({stats[4]})
                """

    elif entity_type == "action":
        # Find which action is being queried
        action_terms = ["batting", "bowling", "fielding", "celebrating", "wicket keeping"]
        target_action = next((term for term in action_terms if term in query_lower), None)

        if target_action:
            cursor.execute("""
            SELECT a.action_name, COUNT(*) as image_count
            FROM cricket_data c
            JOIN action a ON c.action_id = a.action_id
            WHERE LOWER(a.action_name) LIKE %s
            GROUP BY a.action_name
            """, (f"%{target_action}%",))

            stats = cursor.fetchall()
            if stats:
                stats_info = "Action Statistics:\n"
                for action_name, count in stats:
                    stats_info += f"- {action_name}: {count} images\n"

    elif entity_type == "event":
        # Find which event is being queried
        event_terms = {
            "press meet": ["press meet", "press conference", "media", "interview"],
            "practice": ["practice", "training", "net session"],
            "match": ["match", "game", "fixture"],
            "promotional": ["promotional", "promotion", "marketing"]
        }

        target_event = None
        for event, terms in event_terms.items():
            if any(term in query_lower for term in terms):
                target_event = event
                break

        if target_event:
            cursor.execute("""
            SELECT e.event_name, COUNT(*) as image_count
            FROM cricket_data c
            JOIN event e ON c.event_id = e.event_id
            WHERE LOWER(e.event_name) LIKE %s
            GROUP BY e.event_name
            """, (f"%{target_event}%",))

            stats = cursor.fetchall()
            if stats:
                stats_info = "Event Statistics:\n"
                for event_name, count in stats:
                    stats_info += f"- {event_name}: {count} images\n"

    cursor.close()
    conn.close()

    # Prepare prompt for the LLM
    prompt = f"""
    User Question: {query}

    I found information related to this {entity_type} query. Here are some details:

    """

    # Add statistics if available
    if stats_info:
        prompt += f"\n{stats_info}\n"

    # Add details about a sample of images
    sample_size = min(5, len(similar_images))
    for i, (doc, _) in enumerate(similar_images[:sample_size]):
        prompt += f"Example {i+1}:\n"
        prompt += f"- Player: {doc.metadata.get('player_name', 'Unknown')}\n"
        prompt += f"- Event: {doc.metadata.get('event_name', 'Unknown')}\n"
        prompt += f"- Action: {doc.metadata.get('action_name', 'Unknown')}\n"
        prompt += f"- Mood: {doc.metadata.get('mood_name', 'Unknown')}\n"
        prompt += f"- Location: {doc.metadata.get('sublocation_name', 'Unknown')}\n"
        prompt += f"- Caption: {doc.metadata.get('caption', 'No caption')}\n"
        prompt += f"- Description: {doc.page_content}\n\n"

    # Add entity-specific instructions
    if entity_type == "player":
        prompt += """
        Please provide a comprehensive response about this player based on the information.
        Include:
        1. Their role in the team
        2. Events they participated in
        3. Actions they're known for
        4. Any notable characteristics or achievements
        5. A summary of the available images
        """
    elif entity_type == "action":
        prompt += """
        Please provide a comprehensive response about this cricket action based on the information.
        Include:
        1. What this action involves in cricket
        2. Players who perform this action in the available data
        3. Notable examples from the data
        4. Technical aspects of this action if available
        5. A summary of the available images
        """
    elif entity_type == "event":
        prompt += """
        Please provide a comprehensive response about this type of cricket event based on the information.
        Include:
        1. What happens at this event
        2. Players who participated in this event
        3. The significance of this event
        4. Notable examples from the data
        5. A summary of the available images
        """
    elif entity_type == "mood":
        prompt += """
        Please provide a comprehensive response about this mood in cricket context based on the information.
        Include:
        1. When this mood typically occurs in cricket
        2. Players who exhibit this mood in the available data
        3. Events associated with this mood
        4. Notable examples from the data
        5. A summary of the available images
        """
    elif entity_type == "location":
        prompt += """
        Please provide a comprehensive response about this cricket location based on the information.
        Include:
        1. What this location is used for in cricket
        2. Events that take place at this location
        3. Players associated with this location in the data
        4. Notable examples from the data
        5. A summary of the available images
        """
    else:
        prompt += """
        Please provide a comprehensive response to the user's query based on this information.
        Include:
        1. Direct answers to the query
        2. Related information that might be helpful
        3. Examples from the data
        4. A summary of the available images
        """

    prompt += f"""
    Make your response informative, educational, and conversational.
    Don't mention similarity scores or technical details.
    Keep your response concise and to the point, especially when images will be displayed.
    For image-related queries, keep the text extremely brief (1-2 sentences).

    After your response, I will add relevant image URLs if appropriate for this query.
    """

    # Generate response using the LLM
    llm_response = groq_api.generate(prompt)

    # Determine if we should include image URLs based on the query type
    # For descriptive queries, we'll include a few relevant images if they help illustrate the answer
    should_include_images = "image" in query_lower or "picture" in query_lower or "photo" in query_lower

    if should_include_images:
        # Add a selection of relevant image URLs to the response
        full_response = llm_response + "\n\n### Relevant Images:\n\n"
        image_limit = min(10, len(similar_images))  # Show up to 10 images for descriptive queries
        for i, (doc, _) in enumerate(similar_images[:image_limit]):
            image_url = doc.metadata.get('url', 'No URL available')
            player_name = doc.metadata.get('player_name', 'Unknown player')
            event_name = doc.metadata.get('event_name', 'Unknown event')
            action_name = doc.metadata.get('action_name', 'Unknown action')
            full_response += f"{i+1}. {player_name} - {action_name} at {event_name}: {image_url}\n"

        return full_response
    else:
        # For purely descriptive queries, just return the LLM response
        return llm_response
