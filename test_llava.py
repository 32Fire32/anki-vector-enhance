"""Quick test: does llava:7b actually see objects in an image?"""
import base64, io, requests
from PIL import Image

img = Image.open("debug_frame.jpg")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=80)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"b64_len={len(b64)}")

# Test 1: Simple English prompt
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{"role": "user", "content": "What objects do you see in this image?", "images": [b64]}],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 200},
}, timeout=120)
d = r.json()
dur = d.get("total_duration", 0) / 1e9
print(f"duration={dur:.2f}s, prompt_eval={d.get('prompt_eval_count',0)}, eval={d.get('eval_count',0)}")
print(f"response: {d['message']['content']}")

# Test 2: Simplified structured prompt
prompt = (
    "List the objects you see in this image. "
    "For each object, use this format: name | color | detail\n"
    "Example: pen | red | ballpoint\n"
    "List up to 8 objects, most prominent first."
)
r2 = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 4096},
}, timeout=120)
d2 = r2.json()
dur2 = d2.get("total_duration", 0) / 1e9
print(f"\nduration={dur2:.2f}s, prompt_eval={d2.get('prompt_eval_count',0)}, eval={d2.get('eval_count',0)}")
print(f"response: {d2['message']['content']}")
