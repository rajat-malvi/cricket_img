"""
Script to create the users table in the PostgreSQL database
"""

import psycopg2
import config

def create_users_table():
    """
    Create the users table in the PostgreSQL database
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
        
        # Create the users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create the user_queries table to store user query history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_queries (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("Users and user_queries tables created successfully.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating users table: {e}")

if __name__ == "__main__":
    create_users_table()
