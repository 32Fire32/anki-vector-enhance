import torch
from PIL import Image

# 1. Carica il modello YOLOv5 nano
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# 2. Carica l'immagine scattata da Vector
img_path = 'vector_image.jpg'  # Path dell'immagine scattata da Vector
img = Image.open(img_path)

# 3. Rilevamento oggetti
results = model(img)

# 4. Accedi al primo DataFrame contenente i risultati
df = results.pandas().xywh[0]

# 5. Applica una soglia di confidenza (per esempio 0.5) e filtra i risultati
filtered_results = df[df['confidence'] > 0.5]

# 6. Mostra in console gli oggetti rilevati con confidenza superiore alla soglia
print(filtered_results)

# 7. Salva l'immagine con i bounding box sovrapposti
results.save()  # Salva l'immagine con i bounding box sovrapposti
