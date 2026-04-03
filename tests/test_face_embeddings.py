import numpy as np
import asyncio

from vector_personality.memory.sql_server_connector import SQLServerConnector


def test_find_similar_faces(monkeypatch):
    conn = SQLServerConnector()

    # Create two fake embeddings (A and B)
    a = np.ones(128, dtype=np.float32)
    a = a / np.linalg.norm(a)
    b = np.zeros(128, dtype=np.float32)
    b[0] = 1.0
    b = b / np.linalg.norm(b)

    # Query vector similar to 'a'
    q = a * 0.9 + b * 0.1
    q = q / np.linalg.norm(q)

    # Patch get_latest_embeddings to return these as bytes
    def fake_get_latest_embeddings():
        return [
            ('face-A', a.tobytes(), a.size),
            ('face-B', b.tobytes(), b.size)
        ]

    monkeypatch.setattr(conn, 'get_latest_embeddings', lambda: asyncio.get_event_loop().run_in_executor(None, lambda: fake_get_latest_embeddings()))

    # Run find_similar_faces
    loop = asyncio.get_event_loop()
    matches = loop.run_until_complete(conn.find_similar_faces(q, top_k=2))

    assert matches, "No matches found"
    assert matches[0]['face_id'] == 'face-A', f"Expected face-A as top match, got {matches[0]['face_id']}"
    assert matches[0]['score'] > matches[1]['score']
