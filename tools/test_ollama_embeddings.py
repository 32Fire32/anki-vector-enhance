"""
Test Ollama embedding generation (T140).

Run after installing Ollama and downloading mxbai-embed-large:
  winget install Ollama.Ollama
  ollama pull mxbai-embed-large
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_personality.memory.embedding_generator import EmbeddingGenerator


async def test_ollama_embedding():
    """Test Ollama embedding generation."""
    
    print("=" * 60)
    print("Testing Ollama Embedding Generator (T140)")
    print("=" * 60)
    
    # Initialize (no OpenAI key needed)
    try:
        generator = EmbeddingGenerator(
            ollama_url="http://localhost:11434",
            ollama_model="mxbai-embed-large"
        )
        
        print(f"\n✅ Generator initialized")
        info = generator.get_provider_info()
        print(f"   Provider: {info['provider']}")
        print(f"   Model: {info['model']}")
        print(f"   Dimensions: {info['dimensions']}")
        print(f"   Cost: {info['cost']}")
        
    except RuntimeError as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("\nTo fix:")
        print("  1. Install Ollama: winget install Ollama.Ollama")
        print("  2. Download model: ollama pull mxbai-embed-large")
        print("  3. Verify: ollama list")
        return
    
    # Test single embedding
    print("\n" + "=" * 60)
    print("Test 1: Single Embedding")
    print("=" * 60)
    
    test_text = "Vector è un robot intelligente creato da Anki"
    print(f"\nText: '{test_text}'")
    
    embedding = await generator.generate_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Range: [{min(embedding):.4f}, {max(embedding):.4f}]")
    else:
        print("❌ Failed to generate embedding")
        return
    
    # Test batch embeddings
    print("\n" + "=" * 60)
    print("Test 2: Batch Embeddings")
    print("=" * 60)
    
    test_texts = [
        "Ciao Vector, come stai?",
        "Mi ricordo quando abbiamo parlato di Paw Patrol",
        "Mio fratello è un programmatore",
        "Il robot si muove con i cingoli",
        "Gli occhi di Vector cambiano colore"
    ]
    
    print(f"\nGenerating embeddings for {len(test_texts)} texts...")
    embeddings = await generator.generate_embeddings_batch(test_texts)
    
    success_count = sum(1 for e in embeddings if e is not None)
    print(f"✅ Generated {success_count}/{len(test_texts)} embeddings")
    
    # Test semantic similarity (simple cosine)
    print("\n" + "=" * 60)
    print("Test 3: Semantic Similarity")
    print("=" * 60)
    
    def cosine_similarity(a, b):
        """Simple cosine similarity."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b)
    
    # Compare related vs unrelated texts
    text_robot = "Vector è un robot intelligente"
    text_robot_similar = "Il robot si muove con i cingoli"
    text_unrelated = "Mi piace la pizza margherita"
    
    emb_robot = await generator.generate_embedding(text_robot)
    emb_similar = await generator.generate_embedding(text_robot_similar)
    emb_unrelated = await generator.generate_embedding(text_unrelated)
    
    if all([emb_robot, emb_similar, emb_unrelated]):
        sim_related = cosine_similarity(emb_robot, emb_similar)
        sim_unrelated = cosine_similarity(emb_robot, emb_unrelated)
        
        print(f"\nBase text: '{text_robot}'")
        print(f"\nSimilar text: '{text_robot_similar}'")
        print(f"  → Similarity: {sim_related:.4f}")
        
        print(f"\nUnrelated text: '{text_unrelated}'")
        print(f"  → Similarity: {sim_unrelated:.4f}")
        
        if sim_related > sim_unrelated:
            print(f"\n✅ Semantic similarity working correctly!")
            print(f"   Related texts are more similar ({sim_related:.4f} > {sim_unrelated:.4f})")
        else:
            print(f"\n⚠️ Unexpected similarity scores")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_ollama_embedding())
