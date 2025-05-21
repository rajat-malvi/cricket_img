import psycopg2

# Connect to the database
conn = psycopg2.connect(
    dbname="jsk_data",
    user="postgres",
    password="Skd6397@@",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

# Get the schema of the players table
cursor.execute("""
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'players'
ORDER BY ordinal_position
""")

columns = cursor.fetchall()
print("Players table schema:")
for column in columns:
    print(f"Column: {column[0]}, Type: {column[1]}")

# Close the connection
cursor.close()
conn.close()
