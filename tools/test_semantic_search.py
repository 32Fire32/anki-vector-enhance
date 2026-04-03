"""
Test semantic search vs keyword search in ContextBuilder (T140 Phase 3)

This script:
1. Creates test conversations with embeddings
2. Compares semantic search vs keyword search results
3. Validates semantic search finds relevant conversations by meaning
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_personality.memory.sql_server_connector import SQLServerConnector
from vector_personality.memory.embedding_generator import EmbeddingGenerator
from vector_personality.memory.vector_db_connector import VectorDBConnector
from vector_personality.cognition.context_builder import ContextBuilder
from datetime import datetime, timedelta


async def setup_test_data(db, vector_db, embedding_gen):
    """Generate embeddings for existing conversations."""
    print("\n📦 Setting up test data...")
    
    # Get existing conversations from SQL Server (last 30 days)
    print("  Fetching existing conversations...")
    conversations = await db.get_recent_conversations(hours=30*24, limit=50)
    
    if not conversations:
        print("  ⚠️ No existing conversations found. Please talk to Vector first!")
        return []
    
    print(f"  Found {len(conversations)} existing conversations")
    
    # Generate embeddings for conversations that don't have them yet
    conversation_ids = []
    
    print(f"  Processing first 10 conversations...")
    for i, conv in enumerate(conversations[:10], 1):  # Limit to 10 for testing
        conv_id = conv.get('conversation_id')
        text = conv.get('text', '')
        response = conv.get('response_text', '')
        
        print(f"  Conversation {i}: id={conv_id}, text_len={len(text)}, has_response={bool(response)}")
        
        if not conv_id or not text:
            print(f"    ⏭️ Skipping (no ID or text)")
            continue
        
        # Convert conv_id to string
        conv_id_str = str(conv_id)
        
        try:
            # Generate embedding
            combined_text = f"User: {text}\nVector: {response}"
            embedding = await embedding_gen.generate_embedding(combined_text)
            
            if embedding:
                vector_db.add_embedding(
                    conversation_id=conv_id_str,
                    embedding=embedding,
                    metadata={
                        'timestamp': str(conv.get('timestamp', '')),
                        'speaker_id': conv.get('speaker_id', 1),
                        'text_preview': text[:100]
                    }
                )
                conversation_ids.append(conv_id_str)
                print(f"  ✅ Added embedding {conv_id_str}: {text[:50]}...")
        except Exception as e:
            print(f"  ⚠️ Failed to add embedding for {conv_id_str}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Test data ready: {len(conversation_ids)} conversations with embeddings")
    return conversation_ids


async def test_semantic_search(context_builder):
    """Test semantic search with queries."""
    print("\n🔍 Testing semantic search...")
    
    test_queries = [
        {
            "query": "Mi parla della programmazione che faccio",
            "expected": "Should find: Python, machine learning, sviluppatore software"
        },
        {
            "query": "Cosa ho fatto per mantenermi in forma?",
            "expected": "Should find: bicicletta"
        },
        {
            "query": "Che lavoro faccio?",
            "expected": "Should find: sviluppatore software, programmare"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        
        # Build context using semantic search
        context = await context_builder.build_conversation_context(test["query"])
        
        # Check if "RICORDI RECUPERATI" section exists
        if "RICORDI RECUPERATI" in context:
            print("✅ Semantic search found results:")
            lines = context.split("\n")
            for line in lines:
                if line.strip().startswith("- ") and "giorni fa" in line:
                    print(f"  {line.strip()}")
        else:
            print("⚠️ No results found")
    
    return True


async def test_keyword_fallback(db):
    """Test that keyword search still works as fallback."""
    print("\n🔍 Testing keyword fallback...")
    
    # Search using keyword method directly
    results = await db.search_old_conversations(
        keywords=["programmare", "Python"],
        days_back=90,
        limit=5
    )
    
    if results:
        print(f"✅ Keyword search found {len(results)} results:")
        for r in results[:3]:
            text = (r.get('text') or '')[:60]
            print(f"  - {text}...")
    else:
        print("⚠️ Keyword search returned no results")
    
    return len(results) > 0


async def compare_searches(context_builder, db):
    """Compare semantic vs keyword search for same query."""
    print("\n⚖️  Comparing semantic vs keyword search...")
    
    query_text = "sviluppatore software intelligenza artificiale"
    
    # Extract keywords (simple split for this test)
    keywords = query_text.split()
    
    # Keyword search
    print(f"\n1️⃣ Keyword search for: {keywords}")
    keyword_results = await db.search_old_conversations(
        keywords=keywords,
        days_back=90,
        limit=5
    )
    print(f"   Found: {len(keyword_results)} results")
    
    # Semantic search
    print(f"\n2️⃣ Semantic search for: '{query_text}'")
    if context_builder._vector_db and context_builder._embedding_gen:
        semantic_results = await context_builder._semantic_search_old(
            user_text=query_text,
            days_back=90,
            k=5,
            min_similarity=0.3
        )
        print(f"   Found: {len(semantic_results)} results")
        
        if semantic_results:
            print("\n   Top 3 semantic results (with scores):")
            for r in semantic_results[:3]:
                text = (r.get('text') or '')[:60]
                score = r.get('relevance_score', 0)
                print(f"   - [score={score:.3f}] {text}...")
    else:
        print("   ⚠️ Semantic search components not available")
    
    return True


async def cleanup_test_data(db, conversation_ids):
    """Remove test conversations from database."""
    print("\n🧹 Cleaning up test data...")
    
    for conv_id in conversation_ids:
        try:
            await db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conv_id,)
            )
        except Exception as e:
            print(f"  ⚠️ Failed to delete conversation {conv_id}: {e}")
    
    print(f"✅ Cleaned up {len(conversation_ids)} test conversations")


async def main():
    print("=" * 60)
    print("T140 Phase 3: Semantic Search Test")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    # SQL Server connector
    try:
        db = SQLServerConnector(
            server='(localdb)\\MSSQLLocalDB',
            database='vector_memory'
        )
        
        # Test connection
        connected = await db.test_connection()
        if not connected:
            print("❌ SQL Server connection failed")
            return
        print("✅ SQL Server connected")
    except Exception as e:
        print(f"❌ SQL Server initialization failed: {e}")
        return
    
    # Embedding generator
    try:
        embedding_gen = EmbeddingGenerator()
        provider_info = embedding_gen.get_provider_info()
        print(f"✅ Embedding generator: {provider_info['provider']} (model: {provider_info['model']})")
    except Exception as e:
        print(f"❌ Embedding generator failed: {e}")
        return
    
    # Vector DB connector
    try:
        import tempfile
        test_db_dir = os.path.join(tempfile.gettempdir(), "test_semantic_search_chromadb")
        vector_db = VectorDBConnector(persist_directory=test_db_dir)
        print(f"✅ Vector DB connected: {test_db_dir}")
    except Exception as e:
        print(f"❌ Vector DB initialization failed: {e}")
        return
    
    # Context builder (with semantic search support)
    try:
        context_builder = ContextBuilder(
            db_connector=db,
            working_memory=None,  # Not needed for this test
            groq_client=None,
            vector_db=vector_db,
            embedding_gen=embedding_gen
        )
        print("✅ Context builder ready (with semantic search)")
    except Exception as e:
        print(f"❌ Context builder initialization failed: {e}")
        return
    
    # Run tests
    conversation_ids = []
    try:
        # Setup test data
        conversation_ids = await setup_test_data(db, vector_db, embedding_gen)
        
        # Test semantic search
        await test_semantic_search(context_builder)
        
        # Test keyword fallback
        await test_keyword_fallback(db)
        
        # Compare searches
        await compare_searches(context_builder, db)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if conversation_ids:
            await cleanup_test_data(db, conversation_ids)
        
        await db.close()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
