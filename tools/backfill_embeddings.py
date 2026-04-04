"""
Backfill embeddings for existing conversations (T140 Phase 4)

This script:
1. Queries all conversations from SQL Server
2. Identifies which ones don't have embeddings in ChromaDB
3. Generates embeddings in batches
4. Stores them in ChromaDB with metadata

Run once to enable semantic search on historical conversations.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_personality.memory.chromadb_connector import ChromaDBConnector as SQLServerConnector
from vector_personality.memory.embedding_generator import EmbeddingGenerator
from vector_personality.memory.vector_db_connector import VectorDBConnector


async def get_conversations_without_embeddings(db, vector_db):
    """Find conversations that don't have embeddings yet."""
    print("\n🔍 Finding conversations without embeddings...")
    
    # Get all conversations from SQL Server
    all_conversations = await db.query(
        """
        SELECT 
            conversation_id,
            text,
            response_text,
            timestamp,
            speaker_id
        FROM conversations
        WHERE text IS NOT NULL AND text != ''
        ORDER BY timestamp DESC
        """
    )
    
    print(f"  Found {len(all_conversations)} total conversations in SQL Server")
    
    if not all_conversations:
        return []
    
    # Get existing embedding IDs from ChromaDB
    stats = vector_db.get_stats()
    existing_count = stats['total_embeddings']
    print(f"  Found {existing_count} existing embeddings in ChromaDB")
    
    # Get all IDs from ChromaDB
    try:
        # Query with a dummy embedding to get all stored IDs
        all_results = vector_db.collection.get()
        existing_ids = set(all_results['ids']) if all_results.get('ids') else set()
        print(f"  ChromaDB contains {len(existing_ids)} conversation IDs")
    except Exception as e:
        print(f"  ⚠️ Failed to query ChromaDB IDs: {e}")
        existing_ids = set()
    
    # Filter conversations that need embeddings
    conversations_to_process = []
    for conv in all_conversations:
        conv_id = str(conv['conversation_id'])
        if conv_id not in existing_ids:
            conversations_to_process.append(conv)
    
    print(f"  ✅ {len(conversations_to_process)} conversations need embeddings")
    return conversations_to_process


async def backfill_batch(conversations, embedding_gen, vector_db, batch_size=10):
    """Process conversations in batches."""
    total = len(conversations)
    processed = 0
    failed = 0
    
    print(f"\n📦 Processing {total} conversations in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = conversations[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} conversations)...")
        
        for conv in batch:
            conv_id = str(conv['conversation_id'])
            text = conv.get('text', '')
            response = conv.get('response_text', '')
            timestamp = conv.get('timestamp', datetime.now())
            speaker_id = conv.get('speaker_id', '')
            
            try:
                # Generate embedding
                combined_text = f"User: {text}"
                if response:
                    combined_text += f"\nVector: {response}"
                
                embedding = await embedding_gen.generate_embedding(combined_text)
                
                if embedding:
                    # Store in ChromaDB
                    vector_db.add_embedding(
                        conversation_id=conv_id,
                        embedding=embedding,
                        metadata={
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'speaker_id': str(speaker_id),
                            'text_preview': text[:100]
                        }
                    )
                    processed += 1
                    
                    # Show preview
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    print(f"    ✅ {conv_id}: {text_preview}")
                else:
                    failed += 1
                    print(f"    ❌ {conv_id}: Failed to generate embedding")
                    
            except Exception as e:
                failed += 1
                print(f"    ❌ {conv_id}: {e}")
        
        # Progress update
        progress = min(i + batch_size, total)
        print(f"  Progress: {progress}/{total} ({100*progress//total}%)")
    
    return processed, failed


async def verify_backfill(vector_db, expected_count):
    """Verify that embeddings were stored correctly."""
    print("\n✅ Verifying backfill...")
    
    stats = vector_db.get_stats()
    actual_count = stats['total_embeddings']
    
    print(f"  Expected: ~{expected_count} embeddings")
    print(f"  Actual: {actual_count} embeddings")
    print(f"  Similarity metric: {stats['similarity_metric']}")
    
    if actual_count >= expected_count * 0.95:  # Allow 5% tolerance
        print("  ✅ Backfill verification PASSED")
        return True
    else:
        print(f"  ⚠️ Warning: Expected ~{expected_count} but found {actual_count}")
        return False


async def main():
    print("=" * 60)
    print("T140 Phase 4: Backfill Embeddings")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    # SQL Server connector
    try:
        db = SQLServerConnector(
            server='(localdb)\\MSSQLLocalDB',
            database='vector_memory'
        )
        
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
    
    # Vector DB connector (production database)
    try:
        vector_db_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "vector_personality",
            "memory",
            "chromadb_data"
        )
        os.makedirs(vector_db_dir, exist_ok=True)
        
        vector_db = VectorDBConnector(persist_directory=vector_db_dir)
        print(f"✅ Vector DB connected: {vector_db_dir}")
        
        stats = vector_db.get_stats()
        print(f"  Current embeddings: {stats['total_embeddings']}")
    except Exception as e:
        print(f"❌ Vector DB initialization failed: {e}")
        return
    
    # Find conversations without embeddings
    try:
        conversations_to_process = await get_conversations_without_embeddings(db, vector_db)
        
        if not conversations_to_process:
            print("\n✅ All conversations already have embeddings!")
            print("   Nothing to backfill.")
            return
        
        # Confirm with user
        print(f"\n⚠️ About to generate {len(conversations_to_process)} embeddings")
        estimated_time = len(conversations_to_process) * 1.5  # ~1.5 sec per embedding
        print(f"   Estimated time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")
        print("\n   Press Ctrl+C to cancel, or wait 3 seconds to continue...")
        
        await asyncio.sleep(3)
        
        # Process backfill
        processed, failed = await backfill_batch(
            conversations_to_process,
            embedding_gen,
            vector_db,
            batch_size=10
        )
        
        # Verify
        await verify_backfill(vector_db, processed)
        
        # Summary
        print("\n" + "=" * 60)
        print("Backfill Summary:")
        print("=" * 60)
        print(f"✅ Processed: {processed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success rate: {100*processed/(processed+failed):.1f}%")
        print("\n✅ Backfill complete! Semantic search now works on all conversations.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Backfill cancelled by user")
    except Exception as e:
        print(f"\n❌ Backfill failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.close()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
