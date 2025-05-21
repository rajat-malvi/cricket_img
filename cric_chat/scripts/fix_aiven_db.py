"""
Script to fix the discrepancy between local and Aiven PostgreSQL databases
This script will:
1. Check row counts in both databases
2. Delete all data from Aiven database
3. Re-migrate data from local database to Aiven
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from migrate_to_aiven import (
    get_local_db_connection,
    get_aiven_db_connection,
    migrate_reference_data,
    migrate_cricket_data,
    migrate_users_data,
    generate_and_store_embeddings
)

# Load environment variables
load_dotenv()

def check_row_counts():
    """
    Check row counts in both local and Aiven databases

    Returns:
        tuple: (local_count, aiven_count) - row counts for cricket_data table
    """
    try:
        # Connect to local database
        local_conn = get_local_db_connection()
        local_cursor = local_conn.cursor()

        # Connect to Aiven database
        aiven_conn = get_aiven_db_connection()
        aiven_cursor = aiven_conn.cursor()

        # Get row count from local database
        local_cursor.execute("SELECT COUNT(*) FROM cricket_data")
        local_count = local_cursor.fetchone()[0]
        print(f"Local database cricket_data row count: {local_count}")

        # Get row count from Aiven database
        aiven_cursor.execute("SELECT COUNT(*) FROM cricket_data")
        aiven_count = aiven_cursor.fetchone()[0]
        print(f"Aiven database cricket_data row count: {aiven_count}")

        # Close connections
        local_cursor.close()
        local_conn.close()
        aiven_cursor.close()
        aiven_conn.close()

        return local_count, aiven_count

    except Exception as e:
        print(f"Error checking row counts: {e}")
        sys.exit(1)

def delete_aiven_data():
    """
    Delete all data from Aiven database tables
    """
    try:
        # Connect to Aiven database
        aiven_conn = get_aiven_db_connection()
        aiven_cursor = aiven_conn.cursor()

        print("Deleting data from Aiven database...")

        # Delete data from tables in reverse order of dependencies
        # We need to handle foreign key constraints manually

        # First, delete from tables with foreign keys
        print("Deleting data from embeddings...")
        aiven_cursor.execute("DELETE FROM embeddings")
        aiven_conn.commit()

        print("Deleting data from documents...")
        aiven_cursor.execute("DELETE FROM documents")
        aiven_conn.commit()

        print("Deleting data from feedback...")
        try:
            aiven_cursor.execute("DELETE FROM feedback")
            aiven_conn.commit()
        except Exception as e:
            print(f"Warning: Could not delete data from feedback: {e}")

        print("Deleting data from user_queries...")
        try:
            aiven_cursor.execute("DELETE FROM user_queries")
            aiven_conn.commit()
        except Exception as e:
            print(f"Warning: Could not delete data from user_queries: {e}")

        print("Deleting data from users...")
        try:
            aiven_cursor.execute("DELETE FROM users")
            aiven_conn.commit()
        except Exception as e:
            print(f"Warning: Could not delete data from users: {e}")

        print("Deleting data from cricket_data...")
        aiven_cursor.execute("DELETE FROM cricket_data")
        aiven_conn.commit()

        # Then delete from reference tables
        print("Deleting data from players...")
        aiven_cursor.execute("DELETE FROM players")
        aiven_conn.commit()

        print("Deleting data from action...")
        aiven_cursor.execute("DELETE FROM action")
        aiven_conn.commit()

        print("Deleting data from event...")
        aiven_cursor.execute("DELETE FROM event")
        aiven_conn.commit()

        print("Deleting data from mood...")
        aiven_cursor.execute("DELETE FROM mood")
        aiven_conn.commit()

        print("Deleting data from sublocation...")
        aiven_cursor.execute("DELETE FROM sublocation")
        aiven_conn.commit()

        # Reset sequences
        tables = [
            "embeddings", "documents", "feedback", "user_queries",
            "users", "cricket_data"
        ]

        for table in tables:
            try:
                print(f"Resetting sequence for {table}...")
                aiven_cursor.execute(f"ALTER SEQUENCE IF EXISTS {table}_id_seq RESTART WITH 1")
                aiven_conn.commit()
            except Exception as e:
                print(f"Warning: Could not reset sequence for {table}: {e}")

        # Close connection
        aiven_cursor.close()
        aiven_conn.close()

        print("Data deleted from Aiven database")

    except Exception as e:
        print(f"Error deleting data from Aiven database: {e}")
        sys.exit(1)

def main():
    """
    Main function to fix the discrepancy between local and Aiven databases
    """
    print("Starting database verification and fix process...")

    # Check row counts
    local_count, aiven_count = check_row_counts()

    print(f"Local has {local_count} rows, Aiven has {aiven_count} rows")
    print("Forcing deletion of Aiven database data and re-migration...")

    # Delete data from Aiven database
    delete_aiven_data()

    # Re-migrate data from local to Aiven
    print("\nRe-migrating data from local to Aiven...")

    # Step 1: Migrate reference data
    print("\nStep 1: Migrating reference data...")
    migrate_reference_data()

    # Step 2: Migrate cricket data
    print("\nStep 2: Migrating cricket data...")
    migrate_cricket_data()

    # Step 3: Migrate users data
    print("\nStep 3: Migrating users data...")
    migrate_users_data()

    # Step 4: Generate and store embeddings
    print("\nStep 4: Generating and storing embeddings...")
    generate_and_store_embeddings()

    # Verify row counts again
    print("\nVerifying row counts after migration...")
    new_local_count, new_aiven_count = check_row_counts()

    if new_local_count == new_aiven_count:
        print(f"Migration successful! Both databases now have {new_local_count} rows")
    else:
        print(f"Warning: Discrepancy still exists after migration: Local has {new_local_count} rows, Aiven has {new_aiven_count} rows")

    print("\nDatabase verification and fix process complete!")

if __name__ == "__main__":
    main()
