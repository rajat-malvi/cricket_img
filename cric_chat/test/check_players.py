import psycopg2
import config

def check_players():
    """
    Check the players table in the database
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

        # Get all player names from the database
        cursor.execute("SELECT player_id, player_name, team_code FROM players")
        players = cursor.fetchall()

        print(f"Found {len(players)} players in the database:")
        for player_id, player_name, team_code in players:
            print(f"ID: {player_id}, Name: {player_name}, Team: {team_code}")

        # Close the connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error checking players: {e}")

if __name__ == "__main__":
    check_players()
