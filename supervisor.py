import anki_vector
from anki_vector.events import Events
import pyodbc
from datetime import datetime
import time
import logging
import os
import sys

# Imposta la working directory corretta
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Configurazione logging
logging.basicConfig(
    filename="supervisor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configurazione database
SERVER = '(localdb)\MSSQLLocalDB'
DATABASE = 'Vector'
DRIVER = 'ODBC Driver 17 for SQL Server'

connection_string = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

def save_user_intent(robot, event_type, event):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    now = datetime.now()
    try:      
        intent_id = getattr(event, 'intent_id', None)
        json_data = getattr(event, 'json_data', None)

        cursor.execute("""
            INSERT INTO UserIntents (Timestamp, Intent_Id, Intent_Type)
            VALUES (?, ?, ?)
        """, (
            datetime.now(),
            intent_id,
            json_data
        ))
        conn.commit()
        print(f"✅ Evento save_user_intent salvato alle {datetime.now()}!")
    except Exception as e:
        logging.error(f"Errore nel salvataggio UserIntent: {e}")
        logging.exception(e)
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass


def save_observed_object(robot, event_type, event):
    print(f"evento:save_observed_object.")
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    now = datetime.now()
    try:
        object_id = getattr(event, 'object_id', None)
        object_type = getattr(event, 'object_type', None)
        pos_x = getattr(event.pose, 'x', None) if event.pose else None
        pos_y = getattr(event.pose, 'y', None) if event.pose else None
        pos_z = getattr(event.pose, 'z', None) if event.pose else None
        img_x = event.img_rect.x_top_left if event.img_rect else None
        img_y = event.img_rect.y_top_left if event.img_rect else None
        img_w = event.img_rect.width if event.img_rect else None
        img_h = event.img_rect.height if event.img_rect else None
            
        cursor.execute("""
            INSERT INTO ObservedObjects (Timestamp, Object_Id, Object_Type, Pos_X, Pos_Y, Pos_Z, Img_X, Img_Y, Img_W, Img_H)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            object_id,
            object_type,
            pos_x,
            pos_y,
            pos_z,
            img_x,
            img_y,
            img_w,
            img_h
        ))
        conn.commit()
        print(f"✅ evento salvato :save_observed_object.")
    except Exception as e:
        logging.error(f"Errore nel salvataggio ObservedObject: {e}")
        logging.exception(e)
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass   

def save_nav_map_update(robot, event_type, event):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    now = datetime.now()
    print(f"evento:save_nav_map_update.")
    try:
        map_info = getattr(event, 'map_info', None)
        root_depth = getattr(map_info, 'root_depth', None)
        root_size_mm = getattr(map_info, 'root_size_mm', None)
        root_center_x = getattr(map_info, 'root_center_x', None)
        root_center_y = getattr(map_info, 'root_center_y', None)
        cursor.execute("""
            INSERT INTO NavMapUpdates (Timestamp, Root_Depth, Root_Size_MM, Root_Center_X, Root_Center_Y)
            VALUES (?, ?, ?, ?, ?)
        """, (
            now,
            root_depth,
            root_size_mm,
            root_center_x,
            root_center_y
        ))
        conn.commit()
        print(f"evento salvato :save_nav_map_update.")
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Errore nel salvataggio NavMapUpdate: {e}")
        logging.exception(e)

def save_robot_face(robot, event_type, event):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    now = datetime.now()
    print(f"evento:save_robot_face.")
    try:
        face_id = event.face_id
        pose_x = event.pose.x if event.pose else None
        pose_y = event.pose.y if event.pose else None
        pose_z = event.pose.z if event.pose else None
        img_topLeft_x = event.img_rect.x_top_left if event.img_rect else None
        img_topLeft_y = event.img_rect.y_top_left if event.img_rect else None
        img_width = event.img_rect.width if event.img_rect else None
        img_height = event.img_rect.height if event.img_rect else None
            
        cursor.execute("""
            INSERT INTO RobotFaces (Timestamp, Face_Id, Pose_X, Pose_Y, Pose_Z, Img_TopLeft_X, ImgTopLeft_Y, Img_Width, Img_Height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            face_id,
            pose_x,
            pose_y,
            pose_z,
            img_topLeft_x,
            img_topLeft_y,
            img_width,
            img_height
        ))
        conn.commit()
        print(f"evento salvato:save_robot_face.")
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Errore nel salvataggio RobotFace: {e}")
        logging.exception(e)

def save_robot_state(robot, event_type, event):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    now = datetime.now()
    print(f"evento:save_robot_state.")
    try:
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
        logging.error(f"Errore nel salvataggio RobotState: {e}")
        logging.exception(e)

def main():
    try:
        with anki_vector.Robot(serial="00701d95", cache_animation_lists=False, enable_face_detection=True,behavior_control_level=None, enable_nav_map_feed=True) as robot:
            logging.info("\u2705 Supervisor attivo. In ascolto...")

            # robot.events.subscribe(save_robot_state, Events.robot_state)
            robot.events.subscribe(save_robot_face, Events.robot_observed_face)

            # robot.events.subscribe(save_user_intent, Events.user_intent)
            # robot.events.subscribe(save_observed_object, Events.robot_observed_object)
            # robot.events.subscribe(save_nav_map_update, Events.nav_map_update)

            while True:
                time.sleep(0.1)
    except Exception as e:
        logging.error(f"Errore principale: {e}")
        logging.exception(e)

if __name__ == "__main__":
    main()