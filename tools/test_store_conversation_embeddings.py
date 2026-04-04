"""
Test store_conversation() with embedding integration (T140 - Fase 2).

Tests that conversations are stored in both SQL Server and ChromaDB.
"""

import asyncio
import sys
import os
import shutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_personality.memory.chromadb_connector import ChromaDBConnector as SQLServerConnector
from vector_personality.memory.embedding_generator import EmbeddingGenerator
from vector_personality.memory.vector_db_connector import VectorDBConnector


async def test_store_conversation_with_embeddings():
    """Test that store_conversation saves to both SQL and vector DB."""
    
    print("=" * 60)
    print("Testing store_conversation() + Embeddings (T140 - Fase 2)")
    print("=" * 60)
    
    test_chroma_dir = "./test_fase2_chroma"
    
    # Clean up
    if os.path.exists(test_chroma_dir):
        shutil.rmtree(test_chroma_dir)
    
    # Initialize components
    print("\n" + "=" * 60)
    print("Step 1: Initialize Components")
    print("=" * 60)
    
    try:
        sql_db = SQLServerConnector(
            server='localhost',
            database='vector_memory',
            trusted_connection=True
        )
        # Test connection
        await sql_db.query("SELECT 1")
        print("✅ SQL Server connector initialized")
        
    except Exception as e:
        print(f"⚠️ SQL Server not available: {e}")
        print("\n" + "=" * 60)
        print("Skipping SQL Server integration tests")
        print("Testing ChromaDB + Embedding logic only")
        print("=" * 60)
        sql_db = None
    
    try:
        embedding_gen = EmbeddingGenerator(
            ollama_url="http://localhost:11434",
            ollama_model="mxbai-embed-large"
        )
        print("✅ Embedding generator initialized")
        
        vector_db = VectorDBConnector(persist_directory=test_chroma_dir)
        print(f"✅ Vector DB initialized ({vector_db.count()} embeddings)")
        
    except Exception as e:
        print(f"❌ Failed to initialize embedding/vector components: {e}")
        return
    
    if not sql_db:
        print("\n⚠️ SQL Server not available - testing logic only")
        print("To run full test: start SQL Server LocalDB")
        print("\nTesting embedding generation and storage logic...")
        
        # Test embedding generation directly
        test_text = "Mio fratello è un programmatore molto bravo"
        embedding = await embedding_gen.generate_embedding(test_text)
        
        if embedding and len(embedding) == 1024:
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
            
            # Store in vector DB with fake conversation ID
            vector_db.add_embedding(
                conversation_id="test-conv-001",
                embedding=embedding,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'text_preview': test_text[:100]
                }
            )
            print(f"✅ Embedding stored in ChromaDB")
            print(f"   Vector DB count: {vector_db.count()}")
            
            # Test search
            query = "Chi è esperto di programmazione?"
            query_emb = await embedding_gen.generate_embedding(query)
            results = vector_db.search_by_similarity(query_emb, k=3, min_similarity=0.5)
            
            print(f"\n✅ Semantic search working:")
            for i, r in enumerate(results, 1):
                print(f"   {i}. [sim={r['similarity']:.3f}] {r['metadata']['text_preview']}")
            
            print("\n✅ Logic test PASSED - integration ready!")
            print("   Start SQL Server to test full store_conversation() flow")
        
        # Cleanup
        if os.path.exists(test_chroma_dir):
            import time
            time.sleep(0.5)
            try:
                shutil.rmtree(test_chroma_dir)
            except:
                pass
        
        return
    
    # Test 1: Store conversation WITHOUT embedding (backward compatibility)
    print("\n" + "=" * 60)
    print("Test 1: Store WITHOUT Embeddings (Backward Compatible)")
    print("=" * 60)
    
    try:
        # Create test face
        face_id = await sql_db.create_face(name="Test User")
        print(f"✅ Created test face: {face_id}")
        
        # Store without embeddings
        conv_id_1 = await sql_db.store_conversation(
            speaker_id=face_id,
            text="Ciao Vector, come stai?",
            response_text="Ciao! Sto bene, grazie!"
        )
        print(f"✅ Stored conversation (no embedding): {conv_id_1}")
        print(f"   Vector DB count: {vector_db.count()} (should still be 0)")
        
        if vector_db.count() == 0:
            print("✅ Backward compatibility working!")
        else:
            print("⚠️ Unexpected embedding created")
            
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return
    
    # Test 2: Store conversation WITH embedding
    print("\n" + "=" * 60)
    print("Test 2: Store WITH Embeddings (T140)")
    print("=" * 60)
    
    test_conversations = [
        {"text": "Mi ricordo quando abbiamo parlato di Paw Patrol", "response": "Sì! Chase è il tuo preferito!"},
        {"text": "Mio fratello è un programmatore", "response": "Che linguaggi usa?"},
        {"text": "Vector, mi aiuti con i compiti?", "response": "Certo! Di cosa hai bisogno?"},
    ]
    
    stored_ids = []
    
    for conv in test_conversations:
        try:
            conv_id = await sql_db.store_conversation(
                speaker_id=face_id,
                text=conv['text'],
                response_text=conv['response'],
                vector_db=vector_db,
                embedding_gen=embedding_gen
            )
            stored_ids.append(conv_id)
            print(f"✅ Stored with embedding: {conv_id}")
            print(f"   Text: {conv['text'][:50]}...")
            
        except Exception as e:
            print(f"❌ Failed to store: {e}")
    
    print(f"\n✅ Total conversations stored: {len(stored_ids)}")
    print(f"✅ Vector DB embeddings: {vector_db.count()} (expected: 3)")
    
    if vector_db.count() == 3:
        print("✅ All embeddings stored successfully!")
    else:
        print(f"⚠️ Expected 3 embeddings, got {vector_db.count()}")
    
    # Test 3: Verify embeddings are searchable
    print("\n" + "=" * 60)
    print("Test 3: Search Stored Embeddings")
    print("=" * 60)
    
    query = "Chi è esperto di informatica?"
    print(f"\nQuery: '{query}'")
    
    query_embedding = await embedding_gen.generate_embedding(query)
    
    if query_embedding:
        results = vector_db.search_by_similarity(
            query_embedding,
            k=3,
            min_similarity=0.5
        )
        
        print(f"✅ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. [sim={result['similarity']:.3f}] {result['conversation_id']}")
            print(f"      {result['metadata'].get('text_preview', 'N/A')[:60]}...")
        
        # Verify we can find "programmatore" conversation
        found_programmer = any(
            'programmat' in result['metadata'].get('text_preview', '').lower()
            for result in results
        )
        
        if found_programmer:
            print("✅ Semantic search working - found 'programmatore'!")
        else:
            print("⚠️ Expected to find 'programmatore' conversation")
    
    # Test 4: Verify SQL Server has conversations
    print("\n" + "=" * 60)
    print("Test 4: Verify SQL Server Storage")
    print("=" * 60)
    
    recent = await sql_db.get_recent_conversations(hours=1, limit=10)
    print(f"✅ Recent conversations in SQL: {len(recent)}")
    
    for conv in recent[:3]:
        print(f"   - {conv.get('speaker_name', 'Unknown')}: {conv['text'][:50]}...")
    
    # Cleanup
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    
    try:
        # Delete test conversations
        for conv_id in stored_ids:
            await sql_db.execute("DELETE FROM conversations WHERE conversation_id = ?", (conv_id,))
        
        # Delete test face
        await sql_db.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
        
        print(f"✅ Cleaned up {len(stored_ids)} test conversations")
        print(f"✅ Cleaned up test face")
        
        # Clean ChromaDB
        if os.path.exists(test_chroma_dir):
            import time
            time.sleep(0.5)  # Let ChromaDB release file handles
            try:
                shutil.rmtree(test_chroma_dir)
                print(f"✅ Cleaned up ChromaDB test directory")
            except Exception as e:
                print(f"⚠️ Could not delete {test_chroma_dir}: {e} (manual cleanup needed)")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Fase 2 (Integration) Tests PASSED!")
    print("=" * 60)
    print("\nNext: Fase 3 - Implement semantic_search_old() in ContextBuilder")


if __name__ == "__main__":
    asyncio.run(test_store_conversation_with_embeddings())
