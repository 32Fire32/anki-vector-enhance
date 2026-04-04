import anki_vector
from anki_vector.events import Events
import chromadb
from chromadb.config import Settings
from datetime import datetime
import time
import logging
import os
import sys
import uuid
import json

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

# Configurazione database (ChromaDB)
CHROMADB_DIR = os.environ.get('CHROMADB_DIR', './vector_memory_chroma')

# Initialize ChromaDB client and collections
chroma_client = chromadb.PersistentClient(
    path=CHROMADB_DIR,
    settings=Settings(anonymized_telemetry=False, allow_reset=True),
)

_dummy_embedding = [0.0, 0.0, 0.0]

supervisor_events = chroma_client.get_or_create_collection(
    name="supervisor_events",
    metadata={"hnsw:space": "cosine"},
)

def _store_event(event_type: str, data: dict):
    """Store an event in ChromaDB supervisor_events collection."""
    try:
        eid = str(uuid.uuid4())
        meta = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
        }
        # ChromaDB metadata values must be str, int, float, or bool
        for k, v in data.items():
            if v is None:
                meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        supervisor_events.add(
            ids=[eid],
            embeddings=[_dummy_embedding],
            metadatas=[meta],
        )
    except Exception as e:
        logging.error(f"Errore nel salvataggio evento {event_type}: {e}")
        logging.exception(e)


def save_user_intent(robot, event_type, event):
    now = datetime.now()
    try:
        intent_id = getattr(event, 'intent_id', None)
        json_data = getattr(event, 'json_data', None)
        _store_event("user_intent", {
            "intent_id": intent_id or "",
            "intent_type": json_data or "",
        })
        print(f"✅ Evento save_user_intent salvato alle {now}!")
    except Exception as e:
        logging.error(f"Errore nel salvataggio UserIntent: {e}")
        logging.exception(e)


def save_observed_object(robot, event_type, event):
    print(f"evento:save_observed_object.")
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
        _store_event("observed_object", {
            "object_id": object_id or "",
            "object_type": object_type or "",
            "pos_x": pos_x or 0.0,
            "pos_y": pos_y or 0.0,
            "pos_z": pos_z or 0.0,
            "img_x": img_x or 0.0,
            "img_y": img_y or 0.0,
            "img_w": img_w or 0.0,
            "img_h": img_h or 0.0,
        })
        print(f"✅ evento salvato :save_observed_object.")
    except Exception as e:
        logging.error(f"Errore nel salvataggio ObservedObject: {e}")
        logging.exception(e)


def save_nav_map_update(robot, event_type, event):
    print(f"evento:save_nav_map_update.")
    try:
        map_info = getattr(event, 'map_info', None)
        root_depth = getattr(map_info, 'root_depth', None)
        root_size_mm = getattr(map_info, 'root_size_mm', None)
        root_center_x = getattr(map_info, 'root_center_x', None)
        root_center_y = getattr(map_info, 'root_center_y', None)
        _store_event("nav_map_update", {
            "root_depth": root_depth or 0,
            "root_size_mm": root_size_mm or 0.0,
            "root_center_x": root_center_x or 0.0,
            "root_center_y": root_center_y or 0.0,
        })
        print(f"evento salvato :save_nav_map_update.")
    except Exception as e:
        logging.error(f"Errore nel salvataggio NavMapUpdate: {e}")
        logging.exception(e)


def save_robot_face(robot, event_type, event):
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
        _store_event("robot_face", {
            "face_id": face_id or 0,
            "pose_x": pose_x or 0.0,
            "pose_y": pose_y or 0.0,
            "pose_z": pose_z or 0.0,
            "img_topleft_x": img_topLeft_x or 0.0,
            "img_topleft_y": img_topLeft_y or 0.0,
            "img_width": img_width or 0.0,
            "img_height": img_height or 0.0,
        })
        print(f"evento salvato:save_robot_face.")
    except Exception as e:
        logging.error(f"Errore nel salvataggio RobotFace: {e}")
        logging.exception(e)


def save_robot_state(robot, event_type, event):
    print(f"evento:save_robot_state.")
    try:
        pose_x = event.pose.x if event.pose else None
        pose_y = event.pose.y if event.pose else None
        head_angle = event.head_angle_rad if hasattr(event, 'head_angle_rad') else None
        accel_x = event.accel.x if event.accel else None
        accel_y = event.accel.y if event.accel else None
        accel_z = event.accel.z if event.accel else None
        prox_distance = event.prox_data.distance_mm if event.prox_data else None
        touch_raw = event.touch_data.raw_touch_value if event.touch_data else None
        _store_event("robot_state", {
            "pose_x": pose_x or 0.0,
            "pose_y": pose_y or 0.0,
            "head_angle_rad": head_angle or 0.0,
            "accel_x": accel_x or 0.0,
            "accel_y": accel_y or 0.0,
            "accel_z": accel_z or 0.0,
            "prox_distance_mm": prox_distance or 0.0,
            "touch_raw_value": touch_raw or 0,
        })
        now = datetime.now()
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