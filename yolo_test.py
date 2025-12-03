import anki_vector
from anki_vector.util import degrees
from yolov5 import YOLOv5
import torch
import cv2
import numpy as np
import os
from PIL import Image

# === CONFIG ===
YOLO_MODEL_PATH = "./yolov5n.pt"  # Assicurati di avere yolov5n.pt nella cartella!
YOLO_REPO_PATH = "./yolov5"       # Dove hai clonato YOLOv5

# Aggiungi yolov5 al path se serve
import sys
sys.path.append(YOLO_REPO_PATH)

# === FUNZIONE: Acquisisce immagine da Vector ===
def get_vector_image():
    with anki_vector.Robot() as robot:
        robot.camera.init_camera_feed()
        frame = robot.camera.latest_image
        if frame is not None:
            return frame
        else:
            print("❌ Immagine non disponibile")
            return None

# === FUNZIONE: Carica modello YOLO ===
def load_yolo_model():
    model = torch.hub.load(YOLO_REPO_PATH, 'custom', path=YOLO_MODEL_PATH, source='local')
    return model

# === FUNZIONE: Esegui inferenza e stampa risultati ===
def detect_objects(model, image):
    image_np = np.array(image)  # PIL -> numpy
    results = model(image_np)

    # Stampa a console
    print(results.pandas().xyxy[0])  # bounding box + classi

    # Mostra l'immagine con riquadri (facoltativo)
    results.render()
    img = results.ims[0]  # PIL Image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Risultati YOLO", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Carico YOLOv5...")
    model = load_yolo_model()

    print("📷 Acquisisco immagine da Vector...")
    image = get_vector_image()

    if image:
        print("🧠 Eseguo riconoscimento oggetti...")
        detect_objects(model, image)
    else:
        print("❌ Nessuna immagine da analizzare.")
