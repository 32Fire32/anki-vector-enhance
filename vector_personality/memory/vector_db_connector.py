"""
Vector Database Connector using ChromaDB for semantic search.

T140: Vector Database with Embeddings
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorDBConnector:
    """
    Vector database connector using ChromaDB for semantic search.
    Stores conversation embeddings for fast similarity search.
    """

    def __init__(self, persist_directory: str = "./vector_memory_chroma"):
        """
        Initialize ChromaDB persistent client.

        Args:
            persist_directory: Path to ChromaDB persistent storage
        """
        self.persist_directory = persist_directory
        
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get collection for conversation embeddings
            self.collection = self.client.get_or_create_collection(
                name="conversation_embeddings",
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
            )

            count = self.collection.count()
            logger.info(f"✅ ChromaDB initialized: {count} embeddings loaded from {persist_directory}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise

    def add_embedding(
        self,
        conversation_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add single conversation embedding to database.

        Args:
            conversation_id: Unique conversation ID (from SQL Server)
            embedding: Vector embedding (1024 or 1536 dimensions)
            metadata: Optional metadata (timestamp, speaker_id, etc.)
        """
        try:
            # Ensure conversation_id is a string
            conv_id_str = str(conversation_id) if conversation_id is not None else None
            if not conv_id_str:
                raise ValueError("conversation_id cannot be None or empty")
            
            self.collection.add(
                ids=[conv_id_str],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else None
            )
            logger.debug(f"✅ Added embedding for conversation {conv_id_str}")
        except Exception as e:
            logger.error(f"❌ Failed to add embedding for {conversation_id}: {e}")
            raise

    def add_embeddings_batch(
        self,
        conversation_ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add multiple conversation embeddings in batch (faster).

        Args:
            conversation_ids: List of conversation IDs
            embeddings: List of embeddings (same order as IDs)
            metadatas: Optional list of metadata dicts
        """
        if len(conversation_ids) != len(embeddings):
            raise ValueError(f"IDs ({len(conversation_ids)}) and embeddings ({len(embeddings)}) length mismatch")
        
        try:
            self.collection.add(
                ids=conversation_ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"✅ Added {len(conversation_ids)} embeddings in batch")
        except Exception as e:
            logger.error(f"❌ Failed to add batch embeddings: {e}")
            raise

    def search_by_similarity(
        self,
        query_embedding: List[float],
        k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search conversations by semantic similarity.

        Args:
            query_embedding: Embedding of the query text
            k: Number of results to return
            min_similarity: Minimum cosine similarity (0.0-1.0)

        Returns:
            List of dicts with {conversation_id, similarity, metadata}
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "distances"]
            )

            # Convert distances to similarities (ChromaDB returns cosine distances)
            # For cosine: similarity = 1 - distance
            similar_convos = []
            
            if not results['ids'] or not results['ids'][0]:
                logger.debug("No results found in vector search")
                return []

            for i, conv_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                similarity = 1.0 - distance  # Cosine similarity

                if similarity >= min_similarity:
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    similar_convos.append({
                        'conversation_id': conv_id,
                        'similarity': similarity,
                        'metadata': metadata
                    })

            logger.debug(f"✅ Found {len(similar_convos)} conversations with similarity >= {min_similarity}")
            return similar_convos

        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []

    def get_embedding(self, conversation_id: str) -> Optional[List[float]]:
        """
        Retrieve embedding for specific conversation.

        Args:
            conversation_id: Conversation ID to lookup

        Returns:
            Embedding vector or None if not found
        """
        try:
            result = self.collection.get(
                ids=[conversation_id],
                include=["embeddings"]
            )
            if result['embeddings'] and len(result['embeddings']) > 0:
                return result['embeddings'][0]
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get embedding for {conversation_id}: {e}")
            return None

    def delete_embedding(self, conversation_id: str) -> None:
        """
        Delete embedding for conversation.

        Args:
            conversation_id: Conversation ID to delete
        """
        try:
            self.collection.delete(ids=[conversation_id])
            logger.debug(f"✅ Deleted embedding for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete embedding {conversation_id}: {e}")

    def count(self) -> int:
        """Get total number of embeddings in database."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete all embeddings (use with caution!).
        Useful for testing or re-indexing.
        """
        try:
            self.client.delete_collection("conversation_embeddings")
            self.collection = self.client.create_collection(
                name="conversation_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning("⚠️ ChromaDB collection reset - all embeddings deleted")
        except Exception as e:
            logger.error(f"❌ Failed to reset collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_embeddings': self.count(),
            'collection_name': self.collection.name,
            'persist_directory': self.persist_directory,
            'similarity_metric': 'cosine'
        }
