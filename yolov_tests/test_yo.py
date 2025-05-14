import torch
from pathlib import Path
from PIL import Image

# 1. Carica il modello YOLOv5 nano
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', trust_repo=True)

# 2. Carica un'immagine dal disco
img_path = 'test.jpg'  # Assicurati che ci sia un file immagine in C:\vector-sdk chiamato test.jpg
img = Image.open(img_path)

# 3. Esegui il riconoscimento
results = model(img)

# 4. Stampa e salva i risultati
results.print()      # Mostra risultati in console
results.save()       # Salva immagine con i box in runs/detect/exp
