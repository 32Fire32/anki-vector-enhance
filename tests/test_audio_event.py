import anki_vector
import pyodbc
import datetime
from anki_vector.events import Events
from anki_vector.audio import AudioSendMode
import time

# DB config
SERVER = '(localdb)\\MSSQLLocalDB'
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

def connect_db():
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
    return pyodbc.connect(conn_str)

def save_audio_event(event):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    event_type = "audio_send_mode_changed"
    mode = event.mode
    description = str(event)
    cursor.execute(
        "INSERT INTO AudioEvents (Timestamp, EventType, Mode, Description) VALUES (?, ?, ?, ?)",
        (timestamp, event_type, mode, description)
    )
    conn.commit()
    cursor.close()
    conn.close()

def on_audio_event(robot, event_type, event):
    print(f"[EVENTO] Cambio modalità audio: mode={event.mode}")
    save_audio_event(event)

def main():
    with anki_vector.Robot(serial="00701d95", enable_audio_feed=True, cache_animation_lists=False) as robot:
        print("🎧 Connesso. Iscrizione a evento audio_send_mode_changed")
        robot.events.subscribe(on_audio_event, Events.audio_send_mode_changed)

        print("🔄 Modifico modalità audio (Streaming)...")
        robot.audio.set_audio_send_mode(AudioSendMode.Streaming)
        time.sleep(2)

        print("🔄 Modifico modalità audio (Off)...")
        robot.audio.set_audio_send_mode(AudioSendMode.Off)
        time.sleep(2)

        print("🛑 Fine test. Premi CTRL+C per uscire o termina il programma.")
        while True:
            pass

if __name__ == "__main__":
    main()
