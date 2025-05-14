import anki_vector
import pyodbc
import datetime
from anki_vector.events import Events

# Parametri database
SERVER = '(localdb)\\MSSQLlocalDB'
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

# Connessione DB
def connect_db():
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    return pyodbc.connect(conn_str)

# Salvataggio evento
def save_event(event_type, description):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO VisionModeEvents (Timestamp, EventType, Description) VALUES (?, ?, ?)",
        (timestamp, event_type, description)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Callback evento
def on_event(robot, event_type, event):
    print(f"Evento: {event_type} -> {event}")
    save_event(event_type, str(event))

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False) as robot:
        print("✅ Connesso a Vector, in ascolto eventi visivi...")

        robot.events.subscribe(on_event, Events.mirror_mode_disabled)
        robot.events.subscribe(on_event, Events.vision_modes_auto_disabled)

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("❌ Interrotto manualmente.")

if __name__ == "__main__":
    main()
