import anki_vector
import pyodbc
import datetime
from anki_vector.events import Events

# Configurazione database
SERVER = '(localdb)\MSSQLlocalDB'
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

def connect_db():
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    return pyodbc.connect(conn_str)

def save_event(event_type, description):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO NavMapUpdates (Timestamp, EventType, Description) VALUES (?, ?, ?)",
        (timestamp, event_type, description)
    )
    conn.commit()
    cursor.close()
    conn.close()

def on_nav_map_update(robot, event_type, event):
    print(f"🧭 Mappa aggiornata.")
    save_event(event_type, str(event))

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False, enable_nav_map_feed=True) as robot:
        print("✅ Connesso a Vector, sto ascoltando gli aggiornamenti della mappa...")
        robot.events.subscribe(on_nav_map_update, Events.nav_map_update)

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("❌ Interrotto manualmente.")

if __name__ == "__main__":
    main()
