import os
import pandas as pd
import psycopg2

# Database connection parameters
DB_NAME = "jsk_data"
DB_USER = "postgres"
DB_PASSWORD = "Skd6397@@"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """
    Get a connection to the PostgreSQL database

    Returns:
        connection: PostgreSQL database connection
    """
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def fix_players_table():
    """
    Fix the players table by loading the data correctly
    """
    # Read the CSV file
    players_csv = "data/Players.csv"
    df = pd.read_csv(players_csv)

    # Clean column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Add the team_code column if it doesn't exist
    if 'team_code' not in df.columns:
        # Add default team code 'J' for JSK
        df['team_code'] = 'J'

    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM players")

    # Insert data
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO players (player_id, player_name, team_code) VALUES (%s, %s, %s)",
            (row['player_id'], row['player_name'], row['team_code'])
        )

    # Commit changes
    conn.commit()
    print(f"Successfully inserted {len(df)} players into the database.")

    # Verify the data
    cursor.execute("SELECT COUNT(*) FROM players")
    count = cursor.fetchone()[0]
    print(f"Total players in database: {count}")

    # Close connection
    cursor.close()
    conn.close()

if __name__ == "__main__":
    fix_players_table()
