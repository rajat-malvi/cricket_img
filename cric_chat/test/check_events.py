import db_store

def check_event_table():
    conn = db_store.get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM event")
        rows = cursor.fetchall()
        print("Event table contents:")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error querying event table: {e}")
    finally:
        cursor.close()
        conn.close()

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

def check_promotional_event_count():
    conn = db_store.get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Query for promotional event related images
        cursor.execute("""
        SELECT COUNT(*)
        FROM cricket_data c
        LEFT JOIN event e ON c.event_id = e.event_id
        WHERE e.event_name ILIKE '%promot%' OR e.event_name ILIKE '%sponsor%' OR e.event_name ILIKE '%event%'
        """)
        count = cursor.fetchone()[0]
        print(f"Number of promotional event related images: {count}")
        
        # Show the actual events
        cursor.execute("""
        SELECT DISTINCT e.event_name, COUNT(*)
        FROM cricket_data c
        LEFT JOIN event e ON c.event_id = e.event_id
        WHERE e.event_name ILIKE '%promot%' OR e.event_name ILIKE '%sponsor%' OR e.event_name ILIKE '%event%'
        GROUP BY e.event_name
        """)
        events = cursor.fetchall()
        print("Promotional event related events:")
        for event, count in events:
            print(f"  {event}: {count} images")
    except Exception as e:
        print(f"Error querying promotional event count: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Checking database tables...")
    check_event_table()
    print("\nChecking press meet counts...")
    check_press_meet_count()
    print("\nChecking promotional event counts...")
    check_promotional_event_count()
