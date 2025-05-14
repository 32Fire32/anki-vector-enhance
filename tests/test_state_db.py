import anki_vector
import pyodbc
import json
from datetime import datetime

# Configura il collegamento al tuo database
SERVER = '(localdb)\MSSQLlocalDB'   # (o il nome del tuo server)
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

connection_string = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

def save_robot_state(robot, event_type, event):
    """Salva solo l'evento robot_state nei campi spezzati"""
    try:
        now = datetime.now()

        # Parsing dei dati principali
        pose_x = event.pose.x if event.pose else None
        pose_y = event.pose.y if event.pose else None
        head_angle = event.head_angle_rad if hasattr(event, 'head_angle_rad') else None
        accel_x = event.accel.x if event.accel else None
        accel_y = event.accel.y if event.accel else None
        accel_z = event.accel.z if event.accel else None
        prox_distance = event.prox_data.distance_mm if event.prox_data else None
        touch_raw = event.touch_data.raw_touch_value if event.touch_data else None

        # Inserimento nel database
        cursor.execute("""
            INSERT INTO RobotStates (Timestamp, EventType, Pose_X, Pose_Y, Head_Angle_Rad, Accel_X, Accel_Y, Accel_Z, Prox_Distance_MM, Touch_Raw_Value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, now, "robot_state", pose_x, pose_y, head_angle, accel_x, accel_y, accel_z, prox_distance, touch_raw)
        conn.commit()

        print(f"✅ Evento robot_state salvato alle {now}!")
    except Exception as e:
        print(f"❌ Errore salvando evento: {e}")

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False) as robot:
        print("✅ Connesso a Vector, sto ascoltando eventi robot_state...")

        robot.events.subscribe(save_robot_state, anki_vector.events.Events.robot_state)

        # Resta in ascolto infinito
        while True:
            pass

if __name__ == "__main__":
    main()
