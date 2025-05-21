"""
Vector store module for the Cricket Image Chatbot using PostgreSQL
"""

import json
from typing import List, Tuple, Any
try:
    # Try the new import path first
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fall back to the deprecated import path
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

import config
import db_store

def get_embeddings_model():
    """
    Get the embeddings model with fallback options

    Returns:
        An embeddings model instance
    """
    try:
        # Try HuggingFace embeddings first
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading HuggingFace embeddings: {e}")

        # Try OpenAI embeddings as fallback
        try:
            from langchain_openai import OpenAIEmbeddings
            print("Falling back to OpenAI embeddings")
            return OpenAIEmbeddings()
        except Exception as e2:
            print(f"Error loading OpenAI embeddings: {e2}")

            # Simple fallback embeddings for testing
            from langchain_core.embeddings import Embeddings
            import numpy as np

            class DummyEmbeddings(Embeddings):
                """Dummy embeddings for testing"""

                def embed_documents(self, texts):
                    """Generate fixed-size random embeddings for documents"""
                    return [np.random.rand(384).tolist() for _ in texts]

                def embed_query(self, text):
                    """Generate fixed-size random embedding for query"""
                    return np.random.rand(384).tolist()

            print("Using dummy embeddings for testing")
            return DummyEmbeddings()

def get_or_create_vector_store() -> Any:
    """
    Get the existing vector store or create a new one

    This function maintains the same interface as the FAISS version
    but uses PostgreSQL for storage instead.

    Returns:
        Any: A dummy object that maintains compatibility with the rest of the code
    """
    # Check if the database has been initialized with documents
    if not db_store.database_exists():
        # Check if reference data exists
        if not db_store.reference_data_exists():
            # Initialize the database with reference data
            print("Reference data not found in database. Please run 'python init_db.py' to initialize the database.")
            raise RuntimeError("Database not initialized. Run 'python init_db.py' first.")

        # Generate documents from database
        documents = db_store.generate_documents_from_db()

        # Generate embeddings and store in database
        embeddings_model = get_embeddings_model()
        texts = [doc.page_content for doc in documents]
        embeddings = embeddings_model.embed_documents(texts)
        db_store.insert_documents(documents, embeddings)

    # Return a dummy object to maintain compatibility
    # The actual database operations are handled by db_store functions
    return DummyVectorStore()

def get_similar_images(query: str, k: int = 0, similarity_threshold: float = 0.0) -> List[Tuple[Document, float]]:
    """
    Get similar images for a query with similarity scores

    Args:
        query (str): The query text
        k (int): Number of results to return (default: 0, which means return all results)
        similarity_threshold (float): Minimum similarity score (0.0-1.0) to include results (default: 0.0)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    # Ensure the database is initialized
    get_or_create_vector_store()

    try:
        # Get embeddings for the query
        embeddings_model = get_embeddings_model()
        query_embedding = embeddings_model.embed_query(query)

        # Debug: Print embedding dimensions
        print(f"Generated query embedding with dimension: {len(query_embedding)}")

        # Perform similarity search with scores - check against ALL documents
        # Pass the original query text to enable feedback-based adjustments
        # Pass the similarity threshold to filter results
        results = db_store.similarity_search(
            query_embedding,
            k=k,
            query_text=query,
            similarity_threshold=similarity_threshold
        )

        # Debug: Print similarity scores
        if results:
            print(f"Found {len(results)} results with similarity scores (threshold: {similarity_threshold}):")
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

            # Return all results that meet the threshold
            return results

        # If no results, return an empty list instead of random documents
        print("No results from similarity search, returning empty list")
        return []
    except Exception as e:
        print(f"Error in similarity search: {e}")
        # Return an empty list instead of random documents
        print("Error occurred, returning empty list")
        return []

def get_random_documents(k: int = 5) -> List[Tuple[Document, float]]:
    """
    Get random documents from the database as a fallback

    Note: This function is kept for compatibility but should not be used
    in the main similarity search flow.

    Args:
        k (int): Number of documents to return (default: 5)

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity_score) tuples
    """
    try:
        # Connect to the database
        conn = db_store.get_db_connection()
        cursor = conn.cursor()

        # Get random documents
        cursor.execute("""
        SELECT content, metadata
        FROM documents
        ORDER BY RANDOM()
        LIMIT %s
        """, (k,))

        results = cursor.fetchall()
        print(f"Retrieved {len(results)} random documents")

        # Convert to Document objects with dummy scores
        documents_with_scores = []
        for content, metadata_json in results:
            try:
                # Debug: Print metadata type
                print(f"Metadata type: {type(metadata_json)}")

                # Handle different metadata formats using Python dictionaries instead of JSON
                if isinstance(metadata_json, dict):
                    # If it's already a dict, use it directly
                    metadata = metadata_json
                    print("Using metadata dict directly")
                elif isinstance(metadata_json, str):
                    # If it's a string, parse it as a dictionary
                    try:
                        metadata = json.loads(metadata_json)
                        print("Parsed metadata from string")
                    except:
                        # If parsing fails, use a simple dictionary
                        metadata = {"content": metadata_json}
                        print("Created simple metadata dictionary from string")
                elif hasattr(metadata_json, 'decode'):
                    # If it's bytes-like, decode and parse
                    try:
                        metadata = json.loads(metadata_json.decode('utf-8'))
                        print("Parsed metadata from bytes")
                    except:
                        # If parsing fails, use a simple dictionary
                        metadata = {"content": str(metadata_json)}
                        print("Created simple metadata dictionary from bytes")
                else:
                    # If it's something else, convert to string and use as is
                    metadata = {"content": str(metadata_json)}
                    print(f"Created simple metadata dictionary from {type(metadata_json)}")

                # Create document with metadata
                doc = Document(page_content=content, metadata=metadata)

                # Use a consistent score of 0.5 for all random documents
                score = 0.5
                documents_with_scores.append((doc, score))

                print(f"Added random document with score {score:.4f}")

            except Exception as e:
                print(f"Error processing document metadata: {e}")
                # Create a document with empty metadata as fallback
                try:
                    doc = Document(page_content=content, metadata={})
                    score = 0.5
                    documents_with_scores.append((doc, score))
                    print(f"Added document with empty metadata and score {score:.4f}")
                except Exception as e2:
                    print(f"Failed to create document with empty metadata: {e2}")
                    continue

        cursor.close()
        conn.close()

        return documents_with_scores
    except Exception as e:
        print(f"Error getting random documents: {e}")
        # Return an empty list instead of creating dummy documents
        print("Returning empty list instead of dummy documents")
        return []

class DummyVectorStore:
    """
    A dummy class that provides the same interface as the original vector store for compatibility
    """

    def as_retriever(self, search_type=None, search_kwargs=None):
        """
        Return a retriever that maintains compatibility with the existing code

        Args:
            search_type (str): The type of search to perform (not used)
            search_kwargs (dict): Search parameters

        Returns:
            DummyRetriever: A retriever that uses PostgreSQL for storage
        """
        return DummyRetriever(search_kwargs.get("k", 5) if search_kwargs else 5)

    def similarity_search_with_score(self, query, k=5):
        """
        Perform a similarity search with scores

        Args:
            query (str): The query text
            k (int): Number of results to return

        Returns:
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples
        """
        return get_similar_images(query, k=k)

class DummyRetriever:
    """
    A dummy retriever that uses PostgreSQL for storage
    """

    def __init__(self, k=5):
        """
        Initialize the retriever

        Args:
            k (int): Number of results to return
        """
        self.k = k

    def invoke(self, query):
        """
        Retrieve documents for a query

        Args:
            query (str): The query text

        Returns:
            List[Document]: List of retrieved documents
        """
        results = get_similar_images(query, k=self.k)
        return [doc for doc, _ in results]
