import anki_vector
import pyodbc
import datetime
import anki_vector.connection
from anki_vector.events import Events
from anki_vector.connection import ControlPriorityLevel

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
        "INSERT INTO [WakeWordEvents] (Timestamp, EventType, Description) VALUES (?, ?, ?)",
        (timestamp, event_type, description)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Funzione callback eventi
def on_robot_event(robot, event_type, event):
    print(f"Evento: {event_type} -> {event}")
    save_event(event_type, str(event))

# Script principale
def main():
    with anki_vector.Robot(
        serial="00701d95",
        cache_animation_lists=False
    ) as robot:
        print("✅ Connesso a Vector, sto ascoltando eventi vocali e stimolazione...")

        # Iscrizione a più eventi
        robot.events.subscribe(on_robot_event, Events.wake_word)
        robot.events.subscribe(on_robot_event, Events.user_intent)

        # Loop infinito
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("❌ Interrotto manualmente.")

if __name__ == "__main__":
    main()

# from anki_vector import Robot
# from anki_vector.events import Events

# def on_wake_word(robot, event_type, event):
#     print("✨ Wake word detected!")

# with Robot(serial="00701d95", cache_animation_lists=False) as robot:
#     print("🎤 Sto ascoltando il wake word...")
#     robot.events.subscribe(on_wake_word, Events.wake_word)

#     try:
#         while True:
#             pass
#     except KeyboardInterrupt:
#         print("👋 Uscita.")