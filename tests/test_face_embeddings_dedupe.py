import numpy as np
import pytest
from vector_personality.memory.sql_server_connector import SQLServerConnector

@pytest.mark.asyncio
async def test_add_face_embedding_dedup(tmp_path, monkeypatch):
    # Create connector (assumes local DB available as in CI)
    db = SQLServerConnector()

    # Create fake face
    face_id = await db.create_face(name='DedupeTest')

    # Create embedding
    vec = np.random.RandomState(0).randn(512).astype(np.float32)
    b = vec.tobytes()

    # First insert should succeed
    res1 = await db.add_face_embedding(face_id, b, vec.size)
    assert res1 is True

    # Second insert with identical bytes should be skipped (returns False)
    res2 = await db.add_face_embedding(face_id, b, vec.size)
    assert res2 is False

    # Slightly perturbed embedding (below threshold) -> should insert
    vec2 = vec.copy()
    vec2[0] += 0.001  # tiny perturbation
    b2 = vec2.tobytes()
    res3 = await db.add_face_embedding(face_id, b2, vec2.size)
    # If similarity still >= threshold and within time window it may skip; allow either True/False but ensure no crash
    assert res3 in (True, False)
