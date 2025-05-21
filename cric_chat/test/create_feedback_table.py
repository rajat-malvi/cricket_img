"""
Script to create the feedback table in the PostgreSQL database
"""

import psycopg2
import config

def create_feedback_table():
    """
    Create the feedback table in the PostgreSQL database
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        
        cursor = conn.cursor()
        
        # Create the feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            image_url TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating IN (1, -1)),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS feedback_query_idx ON feedback (query)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS feedback_document_id_idx ON feedback (document_id)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS feedback_image_url_idx ON feedback (image_url)
        """)
        
        # Commit the changes
        conn.commit()
        print("Feedback table created successfully")
        
        # Close the connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating feedback table: {e}")

if __name__ == "__main__":
    create_feedback_table()
