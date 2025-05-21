import psycopg2

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="Skd6397@@",
        host="localhost",
        port="5432"
    )
    
    cursor = conn.cursor()
    
    # Try to create the pgvector extension
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("pgvector extension created successfully")
    except Exception as e:
        print(f"Error creating pgvector extension: {e}")
    
    # Check if the extension exists
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    result = cursor.fetchone()
    if result:
        print("pgvector extension is installed")
    else:
        print("pgvector extension is not installed")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
