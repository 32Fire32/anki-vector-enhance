#!/usr/bin/env python3
"""Inspect current speaker_ids and face UUIDs in the database."""
import chromadb

client = chromadb.PersistentClient(path='vector_memory_chroma')

# Show face UUIDs
faces = client.get_collection('faces')
r = faces.get(include=['metadatas'])
print('=== FACES ===')
for id_, m in zip(r['ids'], r['metadatas']):
    print(f'  {id_}: {m}')

# Show unique speaker_ids in conversations
convs = client.get_collection('conversations')
r2 = convs.get(include=['metadatas'])
total = len(r2['ids'])
print(f'\n=== CONVERSATIONS: {total} total ===')
speaker_ids = set(m.get('speaker_id', '?') for m in r2['metadatas'])
print('  Unique speaker_ids:')
for sid in sorted(speaker_ids):
    count = sum(1 for m in r2['metadatas'] if m.get('speaker_id') == sid)
    print(f'    {sid!r}: {count} conversations')

# Show entity profiles
ep = client.get_collection('entity_profiles')
r3 = ep.get(include=['metadatas'])
print(f'\n=== ENTITY PROFILES: {len(r3["ids"])} ===')
for id_, m in zip(r3['ids'], r3['metadatas']):
    print(f'  {id_}: speaker_id={m.get("speaker_id")} name={m.get("entity_name")} facts={m.get("facts_json")}')
