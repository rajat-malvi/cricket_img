"""
Streamlit app for the Cricket Image Chatbot
"""

import re
import requests
import streamlit as st
from typing import List, Tuple
from urllib.parse import urlparse
import datetime

import config
from llm_service import query_images
from vector_store import get_or_create_vector_store
from langchain.docstore.document import Document
import db_store
from login import show_login_page
from auth import initialize_auth_session_state, save_user_query, get_user_queries

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and accessible

    Args:
        url (str): URL to check

    Returns:
        bool: True if the URL is valid and accessible, False otherwise
    """
    try:
        # Check if the URL has a valid format
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        # Try to make a HEAD request to check if the URL is accessible
        # Use a timeout to avoid hanging
        response = requests.head(url, timeout=2)
        return response.status_code < 400
    except Exception:
        return False

def convert_google_drive_url(url: str) -> str:
    """
    Convert a Google Drive sharing URL to a direct image URL

    Args:
        url (str): Google Drive sharing URL

    Returns:
        str: Direct image URL or original URL if not a Google Drive URL
    """
    # Check if it's a Google Drive URL
    if "drive.google.com/file/d/" in url:
        try:
            # Extract the file ID
            file_id = url.split("/d/")[1].split("/")[0]

            # Try multiple formats that might work for Google Drive
            # Format 1: Using uc?export=view (most common)
            direct_url = f"https://drive.google.com/uc?export=view&id={file_id}"

            # Format 2: Using thumbnail format (works for images even with restricted access)
            thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"

            # Format 3: Using a larger thumbnail
            large_thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w2000"

            # Return the large thumbnail URL as it's more likely to work with restricted files
            return large_thumbnail_url
        except Exception:
            # If there's any error in extraction, return the original URL
            return url
    return url

def get_google_drive_embed_html(file_id: str, width: int = 800, height: int = 600) -> str:
    """
    Generate HTML for embedding a Google Drive file in an iframe

    Args:
        file_id (str): Google Drive file ID
        width (int): Width of the iframe
        height (int): Height of the iframe

    Returns:
        str: HTML for embedding the file
    """
    return f"""
    <iframe src="https://drive.google.com/file/d/{file_id}/preview"
            width="{width}" height="{height}" frameborder="0"
            allow="autoplay; encrypted-media" allowfullscreen>
    </iframe>
    """

def extract_urls_from_response(response: str) -> List[Tuple[str, str]]:
    """
    Extract image URLs and descriptions from the LLM response

    Args:
        response (str): The LLM response

    Returns:
        List[Tuple[str, str]]: List of (url, description) tuples or empty list if no URLs found
    """
    # Check if response contains any URLs
    if not response or "http" not in response:
        return []

    # Simple regex to extract URLs
    url_pattern = r'https?://[^\s)"]+'
    urls = re.findall(url_pattern, response)

    # If no URLs found, return empty list
    if not urls:
        return []

    # Extract descriptions (this is a simple approach, might need refinement)
    descriptions = []
    for url in urls:
        # Find the text around the URL
        url_index = response.find(url)

        # Look for the end of the sentence or paragraph
        end_index = response.find('\n', url_index)
        if end_index == -1:
            end_index = len(response)

        # Look for the beginning of the sentence or paragraph
        start_index = response.rfind('\n', 0, url_index)
        if start_index == -1:
            start_index = 0
        else:
            start_index += 1  # Skip the newline

        # Extract the description
        description = response[start_index:end_index].strip()
        description = description.replace(url, '').strip()
        descriptions.append(description)

    # Pair URLs with descriptions
    return list(zip(urls, descriptions))

def initialize_session_state():
    """Initialize session state variables"""
    # Initialize authentication session state
    initialize_auth_session_state()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize current query for feedback
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Initialize feedback tracking
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()  # Set of (query, image_url) pairs that have received feedback

    # Initialize similarity threshold (default 40%)
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.4

def handle_feedback(doc_id: int, image_url: str, rating: int):
    """
    Handle user feedback on image relevance

    Args:
        doc_id (int): Document ID
        image_url (str): Image URL
        rating (int): User rating (1 for positive, -1 for negative)
    """
    # Get the current query
    query = st.session_state.current_query

    # Create a unique key for this feedback
    feedback_key = (query, image_url)

    # Check if feedback has already been given for this query-image pair
    if feedback_key in st.session_state.feedback_given:
        st.warning("You've already provided feedback for this image.")
        return

    # Store the feedback in the database
    success = db_store.store_feedback(doc_id, query, image_url, rating)

    if success:
        # Add to the set of feedback given
        st.session_state.feedback_given.add(feedback_key)

        if rating == 1:
            st.success("Thank you for your positive feedback! This will help improve future search results.")
        else:
            st.success("Thank you for your feedback! We'll try to show more relevant images next time.")
    else:
        st.error("Failed to store feedback. Please try again.")

def display_similar_images(similar_images: List[Tuple[Document, float]], similarity_threshold: float = 0.0, key_suffix: str = "", show_slider: bool = True):
    """
    Display similar images with a similarity threshold slider - show ALL matching images

    Args:
        similar_images (List[Tuple[Document, float]]): List of (document, similarity_score) tuples
        similarity_threshold (float): Minimum similarity score (0.0-1.0) to include results (default: 0.0)
        key_suffix (str): Optional suffix to make the slider key unique
        show_slider (bool): Whether to show the similarity threshold slider (default: True)
    """
    if not similar_images:
        # Don't display anything if no images are found
        return

    # Initialize the similarity threshold in session state if not already present
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = similarity_threshold

    # Only show the slider if requested (for similarity search results)
    if show_slider:
        # Create a unique key for the slider
        slider_key = f"similarity_slider_{key_suffix}" if key_suffix else "similarity_slider_main"

        # Add a slider to control the similarity threshold
        new_threshold = st.slider(
            "Adjust similarity threshold (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.similarity_threshold * 100),
            step=5,
            key=slider_key
        )

        # Update the session state with the new threshold
        st.session_state.similarity_threshold = new_threshold / 100.0
    else:
        # If not showing slider, just use the current threshold
        new_threshold = int(st.session_state.similarity_threshold * 100)

    # Filter images based on the threshold
    filtered_images = []
    for doc, score in similar_images:
        # Convert distance to similarity (0-1 scale)
        similarity = 1.0 - score
        # Only include results that meet the threshold
        if similarity >= st.session_state.similarity_threshold:
            filtered_images.append((doc, score))

    # If no images meet the threshold, show a message
    if not filtered_images:
        st.info(f"Please adjust the similarity threshold below {new_threshold}% to see more images.")
        return

    # Display total count of matching images
    total_images = len(filtered_images)

    # Always show all images without restriction
    display_message = f"Showing All {total_images} Matching Images"

    # Add additional info for face detection if applicable
    if any(doc.metadata.get('no_of_faces', 0) is not None and int(doc.metadata.get('no_of_faces', 0)) >= 2 for doc, _ in filtered_images):
        display_message += f" (With Multiple Faces)"

    # Add similarity threshold info if applicable
    if show_slider:
        display_message += f" (Similarity ≥ {new_threshold}%)"

    st.subheader(display_message)

    # # Add a note about the number of images
    # if total_images > 20:
    #     st.info(f"Displaying all {total_images} matching images. Scroll down to see all results.")

    # Add a download button for image URLs if there are many results
    if total_images > 5:
        # Create a text file with all image URLs
        url_text = "Image URLs:\n\n"
        for i, (doc, _) in enumerate(filtered_images):
            image_url = doc.metadata.get('url', 'No URL available')
            player_name = doc.metadata.get('player_name', 'Unknown player')
            event_name = doc.metadata.get('event_name', 'Unknown event')
            action_name = doc.metadata.get('action_name', 'Unknown action')
            url_text += f"{i+1}. {player_name} - {action_name} at {event_name}: {image_url}\n"

        # Add download button with a unique key
        st.download_button(
            label="Download All Image URLs",
            data=url_text,
            file_name="cricket_image_urls.txt",
            mime="text/plain",
            key=f"download_button_{key_suffix}"
        )

    # Create columns for the images - use 3 columns for layout
    cols = st.columns(3)

    # Display all images (no limit)
    for i, (doc, _) in enumerate(filtered_images):
        with cols[i % 3]:
            # Create a card-like container for each image result
            with st.container():
                # Get the image URL from metadata - try different possible keys
                url = None
                for url_key in ['image_url', 'url', 'URL']:
                    if url_key in doc.metadata and doc.metadata[url_key]:
                        url = doc.metadata[url_key]
                        break

                # If no URL found, create a message
                if not url:
                    st.info("Image URL information not available")
                    # Display available metadata
                    display_image_metadata(doc)
                    continue

                # Create a caption with metadata
                caption = ""
                if 'caption' in doc.metadata:
                    caption = f"{doc.metadata['caption']}"

                # Display caption first for better context
                if caption:
                    st.markdown(f"**{caption}**")

                # Extract file ID if it's a Google Drive URL
                file_id = None
                if "drive.google.com" in url and "/d/" in url:
                    try:
                        file_id = url.split("/d/")[1].split("/")[0]
                    except Exception:
                        file_id = None

                # Try different image sources in order of preference
                image_displayed = False

                # For Google Drive files, try multiple approaches
                if file_id:
                    # 1. Try iframe embedding first (most reliable for Google Drive)
                    try:
                        iframe_html = get_google_drive_embed_html(file_id, width=300, height=300)
                        st.components.v1.html(iframe_html, height=320)
                        image_displayed = True
                    except Exception:
                        # Silently fail and try next method
                        pass

                    # 2. If iframe failed, try large thumbnail
                    if not image_displayed:
                        large_thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w2000"
                        try:
                            st.image(large_thumbnail_url, use_container_width=True)
                            image_displayed = True
                        except Exception:
                            # Silently fail and try next method
                            pass

                    # 3. Try regular thumbnail if large thumbnail failed
                    if not image_displayed:
                        thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
                        try:
                            st.image(thumbnail_url, use_container_width=True)
                            image_displayed = True
                        except Exception:
                            # Silently fail and try next method
                            pass

                    # 4. Try direct URL if thumbnails failed
                    if not image_displayed:
                        direct_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                        try:
                            st.image(direct_url, use_container_width=True)
                            image_displayed = True
                        except Exception:
                            # Silently fail
                            pass

                # For non-Google Drive URLs, try the original URL
                elif not "drive.google.com" in url:
                    try:
                        st.image(url, use_container_width=True)
                        image_displayed = True
                    except Exception:
                        # Silently fail
                        pass

                # If all image loading attempts failed
                if not image_displayed:
                    st.error("Could not load image from any source")

                    # Provide alternative links for Google Drive
                    if file_id:
                        st.markdown("**Try these alternative links:**")
                        st.markdown(f"- [Large Thumbnail](https://drive.google.com/thumbnail?id={file_id}&sz=w2000)")
                        st.markdown(f"- [Small Thumbnail](https://drive.google.com/thumbnail?id={file_id}&sz=w1000)")
                        st.markdown(f"- [Direct Link](https://drive.google.com/uc?export=view&id={file_id})")
                        st.markdown(f"- [Open in Google Drive]({url})")
                        st.markdown(f"- [Preview in Google Drive](https://drive.google.com/file/d/{file_id}/preview)")

                    # Display a placeholder image with caption
                    st.markdown("### Image Preview Not Available")

                # Display image information after the image
                st.markdown(f"**Original URL:** [{url}]({url})")

                # Display player and event information
                player_name = doc.metadata.get('player_name', 'Unknown player')
                event_name = doc.metadata.get('event_name', 'Unknown event')
                action_name = doc.metadata.get('action_name', 'Unknown action')
                st.markdown(f"**Player:** {player_name}")
                st.markdown(f"**Event:** {event_name}")
                st.markdown(f"**Action:** {action_name}")

                # Add feedback buttons
                if image_displayed and 'document_id' in doc.metadata:
                    # Create a unique key for this feedback
                    feedback_key = (st.session_state.current_query, url)

                    # Check if feedback has already been given
                    if feedback_key in st.session_state.feedback_given:
                        st.info("Thank you for your feedback on this image!")
                    else:
                        # Create columns for the feedback buttons
                        fb_cols = st.columns(2)

                        # Add thumbs up button
                        with fb_cols[0]:
                            if st.button("👍 Relevant", key=f"thumbs_up_{i}_{doc.metadata['document_id']}"):
                                handle_feedback(doc.metadata['document_id'], url, 1)

                        # Add thumbs down button
                        with fb_cols[1]:
                            if st.button("👎 Not Relevant", key=f"thumbs_down_{i}_{doc.metadata['document_id']}"):
                                handle_feedback(doc.metadata['document_id'], url, -1)

                # Always display metadata
                display_image_metadata(doc)

def display_image_metadata(doc: Document):
    """
    Display important metadata from a document

    Args:
        doc (Document): Document containing metadata
    """
    # Display additional metadata
    with st.expander("Image Details"):
        # Show the most important metadata
        important_fields = ['player_name', 'action_name', 'event_name', 'mood_name',
                           'sublocation_name', 'timeofday', 'shot_type', 'focus']

        # First display a summary of the most important fields
        for field in important_fields:
            if field in doc.metadata and doc.metadata[field]:
                st.write(f"**{field.replace('_', ' ').title()}:** {doc.metadata[field]}")

        # Then show all other metadata
        st.markdown("#### All Metadata")
        for key, value in doc.metadata.items():
            # Skip embedding field and already displayed important fields
            if not value or key in ['embedding'] or key in important_fields:
                continue

            # Skip ID fields that have corresponding name fields already displayed
            if key == 'player_id' and 'player_name' in doc.metadata:
                continue
            if key == 'action_id' and 'action_name' in doc.metadata:
                continue
            if key == 'event_id' and 'event_name' in doc.metadata:
                continue
            if key == 'mood_id' and 'mood_name' in doc.metadata:
                continue
            if key == 'sublocation_id' and 'sublocation_name' in doc.metadata:
                continue

            # Display the value
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def display_chat_history():
    """Display the chat history"""
    for chat_idx, (role, content) in enumerate(st.session_state.chat_history):
        if role == "user":
            # Store the query for feedback on images in the response
            query = content
            st.session_state.current_query = query

            # Display the user message
            st.chat_message("user").write(content)
        elif isinstance(content, str):
            # Handle old format where content is just a string
            with st.chat_message("assistant"):
                st.write(content)

                # Extract and display images
                url_desc_pairs = extract_urls_from_response(content)
                # Only attempt to display images if URLs were found
                if url_desc_pairs:
                    for url_idx, (url, desc) in enumerate(url_desc_pairs):
                        with st.container():
                            if desc:
                                st.markdown(f"**Description:** {desc}")

                            # Display image information
                            st.markdown(f"**Original URL:** [{url}]({url})")

                            # Extract file ID if it's a Google Drive URL
                            file_id = None
                            if "drive.google.com" in url and "/d/" in url:
                                try:
                                    file_id = url.split("/d/")[1].split("/")[0]
                                except Exception:
                                    file_id = None

                            # Try different image sources in order of preference
                            image_displayed = False

                            # Display description if available
                            if desc:
                                st.markdown(f"**{desc}**")

                            # For Google Drive files, try multiple approaches
                            if file_id:
                                # 1. Try iframe embedding first (most reliable for Google Drive)
                                try:
                                    iframe_html = get_google_drive_embed_html(file_id, width=300, height=300)
                                    st.components.v1.html(iframe_html, height=320)
                                    image_displayed = True
                                    st.success("Image embedded successfully (iframe)")
                                except Exception:
                                    # Silently fail and try next method
                                    pass

                                # 2. If iframe failed, try large thumbnail
                                if not image_displayed:
                                    large_thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w2000"
                                    try:
                                        st.image(large_thumbnail_url, use_container_width=True)
                                        image_displayed = True
                                        st.success("Image loaded successfully (large thumbnail)")
                                    except Exception:
                                        # Silently fail and try next method
                                        pass

                                # 3. Try regular thumbnail if large thumbnail failed
                                if not image_displayed:
                                    thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
                                    try:
                                        st.image(thumbnail_url, use_container_width=True)
                                        image_displayed = True
                                        st.success("Image loaded successfully (thumbnail)")
                                    except Exception:
                                        # Silently fail and try next method
                                        pass

                                # 4. Try direct URL if thumbnails failed
                                if not image_displayed:
                                    direct_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                                    try:
                                        st.image(direct_url, use_container_width=True)
                                        image_displayed = True
                                        st.success("Image loaded successfully (direct link)")
                                    except Exception:
                                        # Silently fail
                                        pass

                            # For non-Google Drive URLs, try the original URL
                            elif not "drive.google.com" in url:
                                try:
                                    st.image(url, caption=desc, use_container_width=True)
                                    image_displayed = True
                                    st.success("Image loaded successfully (original URL)")
                                except Exception:
                                    # Silently fail
                                    pass

                            # If all image loading attempts failed
                            if not image_displayed:
                                st.error("Could not load image from any source")

                                # Provide alternative links for Google Drive
                                if file_id:
                                    st.markdown("**Try these alternative links:**")
                                    st.markdown(f"- [Large Thumbnail](https://drive.google.com/thumbnail?id={file_id}&sz=w2000)")
                                    st.markdown(f"- [Small Thumbnail](https://drive.google.com/thumbnail?id={file_id}&sz=w1000)")
                                    st.markdown(f"- [Direct Link](https://drive.google.com/uc?export=view&id={file_id})")
                                    st.markdown(f"- [Open in Google Drive]({url})")
                                    st.markdown(f"- [Preview in Google Drive](https://drive.google.com/file/d/{file_id}/preview)")

                                # Display a placeholder image with caption
                                st.markdown("### Image Preview Not Available")

                            # Add feedback buttons for history images if displayed
                            if image_displayed:
                                # Get document ID for this URL
                                doc_id = db_store.get_document_id_from_url(url)

                                if doc_id:
                                    # Create a unique key for this feedback
                                    feedback_key = (query, url)

                                    # Check if feedback has already been given
                                    if feedback_key in st.session_state.feedback_given:
                                        st.info("Thank you for your feedback on this image!")
                                    else:
                                        # Create columns for the feedback buttons
                                        fb_cols = st.columns(2)

                                        # Add thumbs up button
                                        with fb_cols[0]:
                                            if st.button("👍 Relevant", key=f"hist_up_{chat_idx}_{url_idx}_{doc_id}"):
                                                handle_feedback(doc_id, url, 1)

                                        # Add thumbs down button
                                        with fb_cols[1]:
                                            if st.button("👎 Not Relevant", key=f"hist_down_{chat_idx}_{url_idx}_{doc_id}"):
                                                handle_feedback(doc_id, url, -1)
        else:
            # Handle new format where content is a tuple of (response_text, similar_images, used_similarity)
            # Check if the content has the used_similarity flag (newer format)
            if len(content) == 3:
                response_text, similar_images, used_similarity = content
            else:
                # Backward compatibility with older format
                response_text, similar_images = content
                used_similarity = False  # Default to not showing slider for older entries

            with st.chat_message("assistant"):
                st.write(response_text)

                # Determine if this was an image request
                query = st.session_state.chat_history[chat_idx-1][1] if chat_idx > 0 else ""
                is_image_request = any(term in query.lower() for term in [
                    "show", "display", "find", "get", "image", "images", "photo",
                    "photos", "picture", "pictures", "see", "view", "look"
                ])

                # Handle image display based on results
                if similar_images:
                    # Check if this is a query for multiple players together
                    query_lower = query.lower()
                    is_multiple_players_query = False
                    is_fans_interaction_query = False

                    # Check for multiple player indicators
                    multiple_player_terms = ["and", "&", "with", "together", "same frame", "single frame", "standing together", "one frame"]
                    if any(term in query_lower for term in multiple_player_terms):
                        # Check if we have player names in the query
                        from db_store import is_player_query, get_player_names_in_query
                        if is_player_query(query_lower):
                            is_multiple_players_query = True
                            print(f"Detected multiple players query in history: '{query_lower}'")

                            # For debugging, print the player names found in the query
                            player_names = get_player_names_in_query(query_lower)
                            if player_names:
                                print(f"Player names found in history query: {player_names}")

                    # Check if this is a query about players interacting with fans
                    fan_terms = ["fan", "fans", "supporter", "supporters", "crowd", "audience", "spectator", "spectators", "interacting", "interaction"]
                    if any(term in query_lower for term in fan_terms):
                        is_fans_interaction_query = True
                        print(f"Detected fan interaction query in history: '{query_lower}'")

                    # For "together" queries, filter out images with only 1 face
                    if is_multiple_players_query:
                        filtered_images = [(doc, score) for doc, score in similar_images
                                          if doc.metadata.get('no_of_faces', 0) is not None
                                          and int(doc.metadata.get('no_of_faces', 0)) >= 2]

                        # If filtering removed all images, show a message
                        if not filtered_images and similar_images:
                            st.info("Here are images related to your query. For images with multiple players in the same frame, please try a more specific query.")
                        else:
                            # Display ALL similar images with the current similarity threshold
                            # No limit on number of images displayed
                            display_similar_images(filtered_images, st.session_state.similarity_threshold,
                                                 key_suffix=f"history_{chat_idx}", show_slider=used_similarity)
                    else:
                        # Display ALL similar images with the current similarity threshold
                        display_similar_images(similar_images, st.session_state.similarity_threshold,
                                              key_suffix=f"history_{chat_idx}", show_slider=used_similarity)
                elif is_image_request and "No cricket images matching" not in response_text:
                    # Only show the "no images" message if it was an image request
                    # and the response doesn't already indicate no images were found
                    st.info("Please try a different search term for cricket images.")

def display_user_sidebar():
    """Display the user sidebar with query history"""
    with st.sidebar:
        st.title("User Dashboard")

        # Display user info
        if st.session_state.user:
            st.write(f"Welcome, **{st.session_state.user['name']}**!")
            st.write(f"Email: {st.session_state.user['email']}")

            # Add logout button
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.is_authenticated = False
                st.session_state.chat_history = []
                st.rerun()

            # Display query history
            st.subheader("Your Recent Queries")

            # Get user's query history
            user_queries = get_user_queries(st.session_state.user['id'])

            if user_queries:
                for query, timestamp in user_queries:
                    # Format the timestamp
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")

                    # Create a clickable query history item
                    if st.button(f"{formatted_time}: {query}", key=f"history_{query}_{formatted_time}"):
                        # Set as current query
                        st.session_state.current_query = query

                        # Add to chat history
                        st.session_state.chat_history.append(("user", query))

                        # Execute the query immediately
                        with st.session_state.get('_lock', st.empty()):
                            # Generate response
                            response_text, similar_images, used_similarity = query_images(query)

                            # Add assistant response to chat history
                            st.session_state.chat_history.append(("assistant", (response_text, similar_images, used_similarity)))

                        # Rerun to display the results
                        st.rerun()
            else:
                st.info("No query history yet. Start asking questions!")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title=config.STREAMLIT_TITLE,
        page_icon="🏏",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Check if user is authenticated
    if not st.session_state.is_authenticated:
        # Show login page
        show_login_page()
        return

    # Display the sidebar with user info and query history
    display_user_sidebar()

    # Main content area
    st.title(f"🏏 {config.STREAMLIT_TITLE}")
    st.markdown(config.STREAMLIT_DESCRIPTION)

    # Initialize vector store (this will create it if it doesn't exist)
    with st.spinner("Loading image database..."):
        # We need to call this to ensure the vector store is initialized,
        # even though we don't use the return value directly
        get_or_create_vector_store()

    # Display chat history
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask about cricket images, player stats, or other cricket information..."):
        # Store the current query for feedback
        st.session_state.current_query = prompt

        # Save the query to the user's history
        if st.session_state.user:
            save_user_query(st.session_state.user['id'], prompt)

        # Add user message to chat history
        st.session_state.chat_history.append(("user", prompt))

        # Display user message
        st.chat_message("user").write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response text, similar images, and whether similarity search was used
                response_text, similar_images, used_similarity = query_images(prompt)

                # Display the response text
                st.write(response_text)

                # Determine query type based on response content
                is_tabular_response = "| " in response_text and " |" in response_text  # Check for markdown table
                is_counting_response = any(term in response_text.lower() for term in ["there are", "found", "count", "total of"])
                is_descriptive_response = len(response_text.split()) > 50 and not is_tabular_response  # Longer text response

                # Handle image display based on results and response type
                if similar_images:
                    # Check if this is a query for multiple players together
                    query_lower = prompt.lower()
                    is_multiple_players_query = False
                    is_fans_interaction_query = False

                    # Check for multiple player indicators
                    multiple_player_terms = ["and", "&", "with", "together", "same frame", "single frame", "standing together", "one frame"]
                    if any(term in query_lower for term in multiple_player_terms):
                        # Check if we have player names in the query
                        from db_store import is_player_query, get_player_names_in_query
                        if is_player_query(query_lower):
                            is_multiple_players_query = True
                            print(f"Detected multiple players query: '{query_lower}'")

                            # For debugging, print the player names found in the query
                            player_names = get_player_names_in_query(query_lower)
                            if player_names:
                                print(f"Player names found in query: {player_names}")

                    # Check if this is a query about players interacting with fans
                    fan_terms = ["fan", "fans", "supporter", "supporters", "crowd", "audience", "spectator", "spectators", "interacting", "interaction"]
                    if any(term in query_lower for term in fan_terms):
                        is_fans_interaction_query = True
                        print(f"Detected fan interaction query: '{query_lower}'")

                    # For "together" queries, filter out images with only 1 face
                    if is_multiple_players_query:
                        filtered_images = [(doc, score) for doc, score in similar_images
                                          if doc.metadata.get('no_of_faces', 0) is not None
                                          and int(doc.metadata.get('no_of_faces', 0)) >= 2]

                        # If filtering removed all images, show a message
                        if not filtered_images and similar_images:
                            st.info("Here are images related to your query. For images with multiple players in the same frame, please try a more specific query.")
                        else:
                            display_similar_images(filtered_images, st.session_state.similarity_threshold,
                                                 key_suffix="current_query", show_slider=used_similarity)
                    else:
                        display_similar_images(similar_images, st.session_state.similarity_threshold,
                                             key_suffix="current_query", show_slider=used_similarity)
                elif "No cricket images matching" not in response_text and not is_tabular_response and not is_counting_response:
                    # Only show the "no images" message if it's not already indicated in the response
                    # and it's not a tabular or counting response
                    st.info("Looking for similar images that might be relevant to your query...")

                    # Fall back to similarity search if no direct results were found
                    with st.spinner("Searching for similar images..."):
                        _, fallback_images, used_similarity = query_images(prompt, force_similarity=True)
                        if fallback_images:
                            st.success("Found some similar images that might be relevant:")

                            # Check if this is a query for multiple players together
                            query_lower = prompt.lower()
                            is_multiple_players_query = False
                            is_fans_interaction_query = False

                            # Check for multiple player indicators
                            multiple_player_terms = ["and", "&", "with", "together", "same frame", "single frame", "standing together", "one frame"]
                            if any(term in query_lower for term in multiple_player_terms):
                                # Check if we have player names in the query
                                from db_store import is_player_query, get_player_names_in_query
                                if is_player_query(query_lower):
                                    is_multiple_players_query = True
                                    print(f"Detected multiple players query in fallback: '{query_lower}'")

                                    # For debugging, print the player names found in the query
                                    player_names = get_player_names_in_query(query_lower)
                                    if player_names:
                                        print(f"Player names found in fallback query: {player_names}")

                            # Check if this is a query about players interacting with fans
                            fan_terms = ["fan", "fans", "supporter", "supporters", "crowd", "audience", "spectator", "spectators", "interacting", "interaction"]
                            if any(term in query_lower for term in fan_terms):
                                is_fans_interaction_query = True
                                print(f"Detected fan interaction query in fallback: '{query_lower}'")

                            # For "together" queries, filter out images with only 1 face
                            if is_multiple_players_query:
                                filtered_images = [(doc, score) for doc, score in fallback_images
                                                  if doc.metadata.get('no_of_faces', 0) is not None
                                                  and int(doc.metadata.get('no_of_faces', 0)) >= 2]

                                # If filtering removed all images, show a message
                                if not filtered_images and fallback_images:
                                    st.info("Here are images related to your query. For images with multiple players in the same frame, please try a more specific query.")
                                else:
                                    # Display ALL similar images with the current similarity threshold
                                    # No limit on number of images displayed
                                    display_similar_images(filtered_images, st.session_state.similarity_threshold,
                                                         key_suffix="fallback_query", show_slider=True)
                            else:
                                display_similar_images(fallback_images, st.session_state.similarity_threshold,
                                                     key_suffix="fallback_query", show_slider=True)
                        else:
                            st.info("Please try a different search term for cricket images.")

                # Add additional context for different response types
                if is_tabular_response:
                    st.info("The table above shows the requested statistics from the cricket database.")
                elif is_counting_response and not similar_images:
                    st.info("This is a count based on the database records. You can also ask to see images related to this query.")
                elif is_descriptive_response and not similar_images:
                    st.info("This is a descriptive answer based on the database. You can also ask to see related images.")

        # Add assistant response to chat history (including similar images and whether similarity was used)
        st.session_state.chat_history.append(("assistant", (response_text, similar_images, used_similarity)))

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    import nltk

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

if __name__ == "__main__":
    # Ensure NLTK resources are available before starting the app
    ensure_nltk_resources()
    main()



# "Show me images of Faf du Plessis batting"
# "Find photos of players celebrating wickets"
# "Do you have any close-up shots of bowlers in action?"
# "Show me team photos from practice sessions"
# "Find images of players wearing the Joburg Super Kings jersey"
# "Do you have photos taken during evening matches?"
# "Show me pictures of players with focused expressions"
# "Find images that show cricket equipment clearly"
# "Do you have photos of players interacting with fans?"
# "Show me images with multiple players in the frame"


# 1. jp king images
# 2. photos of jpking
# 3. jp king solo pictures
# 4. jp king batting in the nets
# 5. moen Ali and fleming
# 6. Moen Ali and Fleming in a frame
# 7. Moen Ali and Fleming in a single frame
# 8. photos of fleming and moen ali together
# 9. photos of fleming and moen ali standing together
# 10. players traveling