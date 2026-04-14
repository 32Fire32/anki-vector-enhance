#!/usr/bin/env python3
"""Migrate all existing conversations to use canonical user IDs.

Since only Nicola has used Vector so far, ALL conversations are mapped to user_1 (Nicola).

Run once:
    python tools/migrate_speaker_ids.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from vector_personality.memory.user_registry import UserRegistry

CHROMA_PATH = "vector_memory_chroma"
NICOLA_FACE_UUID = "9deb0e64-8c2e-4e07-b8fc-24786dffa716"
NICOLA_TELEGRAM_ID = 6067320095

client = chromadb.PersistentClient(path=CHROMA_PATH)
registry = UserRegistry()

# ── Step 1: Create Nicola as user_1 (if not already exists) ──────────────
existing = registry.get_all_users()
if not existing:
    nicola = registry.create_user(
        name="Nicola",
        face_uuids=[NICOLA_FACE_UUID],
        telegram_ids=[NICOLA_TELEGRAM_ID],
        notes="Primary owner",
    )
    print(f"✅ Created user: {nicola}")
else:
    nicola = existing[0]
    # Make sure Nicola has the face UUID and Telegram ID
    registry.update_user(
        nicola["user_id"],
        face_uuids=list(set(nicola.get("face_uuids", []) + [NICOLA_FACE_UUID])),
        telegram_ids=list(set(nicola.get("telegram_ids", []) + [NICOLA_TELEGRAM_ID])),
    )
    print(f"✅ User 1 already exists: {nicola['name']}")

canonical_id = registry.get_canonical_id(1)
print(f"   Canonical ID: {canonical_id}")

# ── Step 2: Migrate all conversations ────────────────────────────────────
print("\nMigrating conversations...")
for collection_name in ["conversations", "semantic_conversations"]:
    try:
        col = client.get_collection(collection_name)
    except Exception:
        print(f"  Collection {collection_name} not found - skipping")
        continue

    result = col.get(include=["metadatas"])
    ids = result["ids"]
    metas = result["metadatas"]

    to_update_ids = []
    to_update_metas = []
    skipped = 0

    for id_, meta in zip(ids, metas):
        old_speaker = meta.get("speaker_id", "")
        new_speaker = canonical_id  # everyone → user_1 (only Nicola used Vector)
        if old_speaker != new_speaker:
            updated_meta = dict(meta)
            updated_meta["speaker_id"] = new_speaker
            to_update_ids.append(id_)
            to_update_metas.append(updated_meta)
        else:
            skipped += 1

    if to_update_ids:
        # ChromaDB update requires re-upsert with existing embeddings
        # Use update() which only updates metadata
        col.update(ids=to_update_ids, metadatas=to_update_metas)
        print(f"  ✅ {collection_name}: updated {len(to_update_ids)} records (skipped {skipped} already correct)")
    else:
        print(f"  ✅ {collection_name}: all {skipped} records already correct")

# ── Step 3: Migrate entity_profiles ──────────────────────────────────────
print("\nMigrating entity_profiles...")
try:
    ep_col = client.get_collection("entity_profiles")
    result = ep_col.get(include=["metadatas"])
    ids = result["ids"]
    metas = result["metadatas"]

    to_update_ids = []
    to_update_metas = []
    for id_, meta in zip(ids, metas):
        if meta.get("speaker_id") != canonical_id:
            updated = dict(meta)
            updated["speaker_id"] = canonical_id
            to_update_ids.append(id_)
            to_update_metas.append(updated)

    if to_update_ids:
        ep_col.update(ids=to_update_ids, metadatas=to_update_metas)
        print(f"  ✅ entity_profiles: updated {len(to_update_ids)} records")
    else:
        print(f"  ✅ entity_profiles: already correct")
except Exception as e:
    print(f"  ⚠️  entity_profiles: {e}")

print(f"\n✅ Migration complete. All data now uses canonical_id='{canonical_id}' (Nicola)")
