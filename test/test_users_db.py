"""
Script to test the users database setup
"""

import psycopg2
import config
from auth import register_user, login_user

def test_users_db():
    """
    Test the users database setup
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
        
        # Check if users table exists
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'users'
        )
        """)
        
        users_table_exists = cursor.fetchone()[0]
        
        if users_table_exists:
            print("Users table exists.")
            
            # Check if user_queries table exists
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'user_queries'
            )
            """)
            
            user_queries_table_exists = cursor.fetchone()[0]
            
            if user_queries_table_exists:
                print("User queries table exists.")
                
                # Test user registration
                print("\nTesting user registration...")
                success, message = register_user("Test User", "test@example.com", "password123")
                print(f"Registration result: {success}, Message: {message}")
                
                # Test user login
                print("\nTesting user login...")
                success, message, user_data = login_user("test@example.com", "password123")
                print(f"Login result: {success}, Message: {message}")
                if user_data:
                    print(f"User data: {user_data}")
                
                # Clean up test user
                print("\nCleaning up test user...")
                cursor.execute("DELETE FROM users WHERE email = 'test@example.com'")
                conn.commit()
                print("Test user deleted.")
                
            else:
                print("User queries table does not exist.")
        else:
            print("Users table does not exist.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error testing users database: {e}")

if __name__ == "__main__":
    test_users_db()
