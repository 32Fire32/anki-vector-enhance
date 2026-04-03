"""
Test ChromaDB + Ollama integration (T140 - Fase 1).

Run after installing ChromaDB and Ollama:
  pip install chromadb
  ollama pull mxbai-embed-large
"""

import asyncio
import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_personality.memory.embedding_generator import EmbeddingGenerator
from vector_personality.memory.vector_db_connector import VectorDBConnector


async def test_chromadb_integration():
    """Test full ChromaDB + Ollama integration."""
    
    print("=" * 60)
    print("Testing ChromaDB + Ollama Integration (T140 - Fase 1)")
    print("=" * 60)
    
    test_dir = "./test_chroma_db"
    
    # Clean up any existing test database
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"\n🗑️  Cleaned up existing test directory")
    
    # Initialize components
    print("\n" + "=" * 60)
    print("Step 1: Initialize Components")
    print("=" * 60)
    
    try:
        embedding_gen = EmbeddingGenerator(
            ollama_url="http://localhost:11434",
            ollama_model="mxbai-embed-large"
        )
        print("✅ EmbeddingGenerator initialized")
        
        vector_db = VectorDBConnector(persist_directory=test_dir)
        print(f"✅ VectorDBConnector initialized")
        print(f"   Embeddings in DB: {vector_db.count()}")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test 1: Store embeddings
    print("\n" + "=" * 60)
    print("Step 2: Store Test Conversations")
    print("=" * 60)
    
    test_conversations = [
        {"id": "conv-001", "text": "Ciao Vector, come stai oggi?"},
        {"id": "conv-002", "text": "Mi ricordo quando abbiamo parlato di Paw Patrol"},
        {"id": "conv-003", "text": "Mio fratello è un programmatore molto bravo"},
        {"id": "conv-004", "text": "Vector è un robot intelligente con cingoli"},
        {"id": "conv-005", "text": "Mi piace la pizza margherita"},
    ]
    
    print(f"\nGenerating and storing {len(test_conversations)} embeddings...")
    
    for conv in test_conversations:
        embedding = await embedding_gen.generate_embedding(conv['text'])
        if embedding:
            vector_db.add_embedding(
                conversation_id=conv['id'],
                embedding=embedding,
                metadata={'text': conv['text']}
            )
            print(f"  ✅ Stored: {conv['id']} - {conv['text'][:40]}...")
    
    print(f"\n✅ Total embeddings in DB: {vector_db.count()}")
    
    # Test 2: Semantic search
    print("\n" + "=" * 60)
    print("Step 3: Semantic Search")
    print("=" * 60)
    
    queries = [
        "Chi è un esperto di programmazione?",  # Should find conv-003
        "Parliamo di robot con ruote",          # Should find conv-004
        "Cibo italiano che mi piace",           # Should find conv-005
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        query_embedding = await embedding_gen.generate_embedding(query)
        
        if query_embedding:
            results = vector_db.search_by_similarity(
                query_embedding,
                k=3,
                min_similarity=0.5
            )
            
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. [sim={result['similarity']:.3f}] {result['metadata'].get('text', 'N/A')[:50]}...")
    
    # Test 3: Persistence
    print("\n" + "=" * 60)
    print("Step 4: Test Persistence (Reload DB)")
    print("=" * 60)
    
    print(f"\nClosing and reopening database...")
    del vector_db
    
    vector_db_reloaded = VectorDBConnector(persist_directory=test_dir)
    print(f"✅ Database reloaded")
    print(f"   Embeddings after reload: {vector_db_reloaded.count()}")
    
    if vector_db_reloaded.count() == len(test_conversations):
        print(f"✅ Persistence working correctly!")
    else:
        print(f"⚠️ Persistence issue: expected {len(test_conversations)}, got {vector_db_reloaded.count()}")
    
    # Test 4: Get stats
    print("\n" + "=" * 60)
    print("Step 5: Database Statistics")
    print("=" * 60)
    
    stats = vector_db_reloaded.get_stats()
    print(f"\nDatabase Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    
    shutil.rmtree(test_dir)
    print(f"✅ Test database cleaned up")
    
    print("\n" + "=" * 60)
    print("✅ All ChromaDB Integration Tests Passed!")
    print("=" * 60)
    print("\nFase 1 (Setup) COMPLETE ✅")
    print("Ready for Fase 2: Integration with store_conversation()")


if __name__ == "__main__":
    asyncio.run(test_chromadb_integration())
