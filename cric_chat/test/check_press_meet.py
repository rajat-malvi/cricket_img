import db_store

def check_press_meet_count():
    conn = db_store.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Query for press meet related events
        cursor.execute("""
        SELECT COUNT(*)
        FROM cricket_data c
        LEFT JOIN event e ON c.event_id = e.event_id
        WHERE e.event_name ILIKE '%press%' OR e.event_name ILIKE '%media%' OR e.event_name ILIKE '%conference%'
        """)
        count = cursor.fetchone()[0]
        print(f"Number of press meet related images: {count}")
        
        # Show the actual events
        cursor.execute("""
        SELECT DISTINCT e.event_name, COUNT(*)
        FROM cricket_data c
        LEFT JOIN event e ON c.event_id = e.event_id
        WHERE e.event_name ILIKE '%press%' OR e.event_name ILIKE '%media%' OR e.event_name ILIKE '%conference%'
        GROUP BY e.event_name
        """)
        events = cursor.fetchall()
        print("Press meet related events:")
        for event, count in events:
            print(f"  {event}: {count} images")
    except Exception as e:
        print(f"Error querying press meet count: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    check_press_meet_count()
