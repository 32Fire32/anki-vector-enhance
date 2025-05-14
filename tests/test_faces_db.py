import anki_vector
import pyodbc
from datetime import datetime

# Dati di connessione al database
SERVER = '(localdb)\MSSQLlocalDB'   # (o il nome del tuo server)
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

def save_robot_face(robot, event_type, event):
    try:
        now = datetime.now()

        face_id = event.face_id if hasattr(event, 'face_id') else None
        pose_x = event.pose.x if event.pose else None
        pose_y = event.pose.y if event.pose else None
        pose_z = event.pose.z if event.pose else None
        img_x = event.img_rect.x_top_left if event.img_rect else None
        img_y = event.img_rect.y_top_left if event.img_rect else None
        img_w = event.img_rect.width if event.img_rect else None
        img_h = event.img_rect.height if event.img_rect else None

        cursor.execute("""
            INSERT INTO RobotFaces (Timestamp, Face_Id, Pose_X, Pose_Y, Pose_Z, Img_TopLeft_X, Img_TopLeft_Y, Img_Width, Img_Height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, now, face_id, pose_x, pose_y, pose_z, img_x, img_y, img_w, img_h)
        conn.commit()

        print(f"✅ Volto salvato alle {now}!")
    except Exception as e:
        print(f"❌ Errore salvando volto: {e}")

def main():
    with anki_vector.Robot(serial="00701d95", cache_animation_lists=False, enable_face_detection=True) as robot:
        print("✅ Connesso a Vector, sto ascoltando volti osservati...")
        robot.vision.enable_face_detection(True)

        robot.events.subscribe(save_robot_face, anki_vector.events.Events.robot_observed_face)

        while True:
            pass

if __name__ == "__main__":
    main()
