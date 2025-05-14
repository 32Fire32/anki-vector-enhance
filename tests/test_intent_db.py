import anki_vector
import pyodbc
import datetime
from anki_vector.events import Events

# Parametri per il database
SERVER = '(localdb)\\MSSQLlocalDB'   # Sostituisci con il tuo server
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

# Funzione di connessione al database
def connect_db():
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    return pyodbc.connect(conn_str)

# Funzione per salvare un intento
def save_intent(intent_id, intent_type, json_data):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO RobotVoiceIntents (Timestamp, IntentId, IntentType, JsonData) VALUES (?, ?, ?, ?)",
        (timestamp, intent_id, intent_type, json_data)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Funzione callback per gli eventi di intento
def on_user_intent(robot, event_type, event):
    print(f"🎯 Intent riconosciuto: {event}")
    intent_id = getattr(event, 'intent_id', None)
    intent_type = getattr(event, 'intent_type', None)
    json_data = getattr(event, 'json_data', None)
    save_intent(intent_id, intent_type, json_data)

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False) as robot:
        print("🎤 Sto ascoltando gli intenti vocali...")
        robot.events.subscribe(on_user_intent, Events.user_intent)

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("👋 Uscita.")

if __name__ == "__main__":
    main()
