import psycopg2
import config

def update_team_code():
    """
    Update the team code for all players from 'C' (CSK) to 'J' (JSK)
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

        # Update all players to have team code 'J' for JSK
        cursor.execute("UPDATE players SET team_code = 'J' WHERE team_code = 'C'")
        
        # Commit the changes
        conn.commit()
        
        # Check how many rows were updated
        print(f"Updated {cursor.rowcount} players to team code 'J' for JSK")
        
        # Verify the update
        cursor.execute("SELECT player_id, player_name, team_code FROM players")
        players = cursor.fetchall()
        
        print(f"\nVerifying {len(players)} players in the database:")
        for player_id, player_name, team_code in players:
            print(f"ID: {player_id}, Name: {player_name}, Team: {team_code}")

        # Close the connection
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error updating team code: {e}")

if __name__ == "__main__":
    update_team_code()
