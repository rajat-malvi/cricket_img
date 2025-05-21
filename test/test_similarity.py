"""
Test script to verify the similarity search functionality
"""

import sys
import traceback
from vector_store import get_embeddings_model
from db_store import similarity_search, get_db_connection

def test_similarity_search():
    """
    Test the similarity search functionality
    """
    print("Testing similarity search...")

    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        print("Database connection successful")

        # Check if pgvector extension is installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cursor.fetchone():
            print("pgvector extension is installed")
        else:
            print("pgvector extension is NOT installed")

        # Check embeddings table
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        print(f"Found {count} embeddings in the database")

        cursor.close()
        conn.close()

        # Get embeddings model
        print("Getting embeddings model...")
        embeddings_model = get_embeddings_model()
        print("Embeddings model loaded successfully")

        # Generate a test query embedding
        test_query = "cricket player batting"
        print(f"Test query: '{test_query}'")

        print("Generating query embedding...")
        query_embedding = embeddings_model.embed_query(test_query)
        print(f"Generated query embedding with dimension: {len(query_embedding)}")

        # Perform similarity search
        print("Performing similarity search...")
        results = similarity_search(query_embedding, k=5, query_text=test_query)

        if results:
            print(f"Found {len(results)} results:")
            for i, (doc, score) in enumerate(results):
                # Convert score to similarity percentage (0-100%)
                similarity_pct = (1.0 - score) * 100
                print(f"  Result {i+1}: Score = {score:.4f}, Similarity = {similarity_pct:.2f}%")

                # Print a snippet of the document content
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  Content: {content_preview}")

                # Print image URL if available
                if 'image_url' in doc.metadata:
                    print(f"  Image URL: {doc.metadata['image_url']}")
                print()
        else:
            print("No results found.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_similarity_search()
