#!/usr/bin/env python3
"""Clean up hallucinated conversations from the database."""
import chromadb

client = chromadb.PersistentClient(path='vector_memory_chroma')
conv_col = client.get_collection('conversations')

# Find the bad 'Hai 39 anni' response
results = conv_col.get(include=['metadatas'])
deleted = 0
for id_, meta in zip(results['ids'], results['metadatas']):
    response = meta.get('response_text', '').lower()
    if 'hai 39 anni' in response:
        print(f'Found hallucination: {id_}')
        print(f'  Response: {meta.get("response_text", "")}')
        # Delete it
        conv_col.delete(ids=[id_])
        deleted += 1

if deleted:
    print(f'✅ Deleted {deleted} hallucinated conversation(s)')
else:
    print('No hallucinations found')
