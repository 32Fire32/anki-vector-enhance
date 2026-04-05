import base64, io, requests
from PIL import Image, ImageEnhance

img = Image.open("debug_frame.jpg")
img = ImageEnhance.Contrast(img).enhance(1.4)
img = ImageEnhance.Sharpness(img).enhance(2.5)
img = ImageEnhance.Brightness(img).enhance(1.1)
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

# Test 1: brand/character recognition
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{"role": "user", "content":
        "Do you recognize any specific characters, brands, or named vehicles in this image? "
        "For example characters from movies or TV shows (like Pixar Cars, Hot Wheels, etc.). "
        "Be specific if you can identify any.",
        "images": [b64]}],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 300},
}, timeout=120)
d = r.json()
print(f"=== Brand/Character recognition ===")
print(f"duration={d['total_duration']/1e9:.2f}s")
print(d["message"]["content"])

# Test 2: direct question about Lightning McQueen
r2 = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{"role": "user", "content":
        "Is there a Lightning McQueen or any Pixar Cars character toy in this image?",
        "images": [b64]}],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 200},
}, timeout=120)
d2 = r2.json()
print(f"\n=== Lightning McQueen specific ===")
print(f"duration={d2['total_duration']/1e9:.2f}s")
print(d2["message"]["content"])
