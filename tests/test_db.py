import anki_vector
import pyodbc
import datetime
from anki_vector.events import Events

# Parametri per il database
SERVER = '(localdb)\MSSQLlocalDB'   # (o il nome del tuo server)
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

# Funzione di connessione al database
def connect_db():
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    return pyodbc.connect(conn_str)

# Funzione per salvare un evento
def save_event(event_type, description):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO Events (Timestamp, EventType, Description) VALUES (?, ?, ?)",
        (timestamp, event_type, description)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Funzione callback eventi
def on_robot_event(robot, event_type, event):
    print(f"Evento: {event_type} -> {event}")
    save_event(event_type, str(event))

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False, enable_face_detection=True, enable_custom_object_detection=True) as robot:
        print("✅ Connesso a Vector, sto ascoltando eventi...")
        
        # ATTIVA i vision modes manualmente
        robot.vision.enable_face_detection(True)
        robot.vision.enable_custom_object_detection(True)

        # Iscrivi agli eventi che vogliamo tracciare
        robot.events.subscribe(on_robot_event, Events.robot_observed_face)
        robot.events.subscribe(on_robot_event, Events.wake_word)
        robot.events.subscribe(on_robot_event, Events.robot_state)

        # Aspetta indefinitamente (CTRL+C per terminare)
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("❌ Interrotto manualmente.")

if __name__ == "__main__":
    main()
