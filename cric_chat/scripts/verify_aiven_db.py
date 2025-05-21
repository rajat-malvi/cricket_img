"""
Script to verify the Aiven PostgreSQL database connection and setup
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Load environment variables
load_dotenv()

def get_aiven_db_connection():
    """
    Get a connection to the Aiven PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
    """
    try:
        return psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            sslmode='require'
        )
    except Exception as e:
        print(f"Error connecting to Aiven database: {e}")
        raise

def check_connection():
    """
    Check the connection to the Aiven PostgreSQL database
    """
    try:
        conn = get_aiven_db_connection()
        cursor = conn.cursor()
        
        # Get PostgreSQL version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        
        print(f"Successfully connected to Aiven PostgreSQL database")
        print(f"PostgreSQL version: {version}")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error connecting to Aiven PostgreSQL database: {e}")
        return False

def check_pgvector_extension():
    """
    Check if the pgvector extension is installed
    """
    try:
        conn = get_aiven_db_connection()
        cursor = conn.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        extension = cursor.fetchone()
        
        if extension:
            print("pgvector extension is installed")
        else:
            print("pgvector extension is NOT installed")
            print("Attempting to install pgvector extension...")
            
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                print("pgvector extension installed successfully")
            except Exception as e:
                print(f"Error installing pgvector extension: {e}")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error checking pgvector extension: {e}")
        return False

def check_tables():
    """
    Check the tables in the Aiven PostgreSQL database
    """
    try:
        conn = get_aiven_db_connection()
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = cursor.fetchall()
        
        print("Database tables:")
        for table in tables:
            print(f"- {table[0]}")
            
            # Get row count for each table
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  Rows: {count}")
        
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error checking tables: {e}")
        return False

def main():
    """
    Main function to verify the Aiven PostgreSQL database
    """
    print("Verifying Aiven PostgreSQL database...")
    
    # Check connection
    print("\nChecking connection...")
    if not check_connection():
        print("Failed to connect to Aiven PostgreSQL database")
        sys.exit(1)
    
    # Check pgvector extension
    print("\nChecking pgvector extension...")
    check_pgvector_extension()
    
    # Check tables
    print("\nChecking tables...")
    check_tables()
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()
