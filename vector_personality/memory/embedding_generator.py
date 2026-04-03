"""
Embedding Generator supporting Ollama (local) and OpenAI (fallback).

T140: Vector Database with Embeddings
"""

import logging
import requests
from typing import List, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings using Ollama (local, free) or OpenAI (fallback).
    
    Priority: Ollama > OpenAI
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mxbai-embed-large",
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small"
    ):
        """
        Initialize embedding generator.

        Args:
            ollama_url: Ollama API endpoint (default localhost)
            ollama_model: Ollama embedding model (mxbai-embed-large recommended)
            openai_api_key: OpenAI API key (optional fallback)
            openai_model: OpenAI model (text-embedding-3-small or 3-large)
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

        # Track which provider is available
        self.ollama_available = self._check_ollama()
        self.openai_available = bool(openai_api_key)

        if self.ollama_available:
            logger.info(f"✅ Ollama embeddings available: {ollama_model} (local, free)")
            self.dimensions = 1024  # mxbai-embed-large
        elif self.openai_available:
            logger.warning("⚠️ Ollama not available, using OpenAI fallback")
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            self.dimensions = 1536 if "small" in openai_model else 3072
        else:
            logger.error("❌ No embedding provider available (Ollama or OpenAI required)")
            raise RuntimeError("No embedding provider configured")

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check exact match or with :latest tag
                if self.ollama_model in model_names:
                    return True
                elif f"{self.ollama_model}:latest" in model_names:
                    # Update model name to include :latest tag
                    self.ollama_model = f"{self.ollama_model}:latest"
                    return True
                else:
                    logger.warning(
                        f"⚠️ Ollama running but model '{self.ollama_model}' not found. "
                        f"Available: {model_names}. Run: ollama pull {self.ollama_model}"
                    )
            return False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for single text.

        Args:
            text: Text to embed (max ~8000 tokens for OpenAI, unlimited for Ollama)

        Returns:
            List of floats (1024 for Ollama, 1536 for OpenAI-small)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Try Ollama first
        if self.ollama_available:
            try:
                return self._generate_ollama_embedding(text)
            except Exception as e:
                logger.error(f"Ollama embedding failed: {e}")
                # Try OpenAI fallback if available
                if self.openai_available:
                    logger.info("Falling back to OpenAI...")
                else:
                    return None

        # OpenAI fallback
        if self.openai_available:
            try:
                return await self._generate_openai_embedding(text)
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {e}")
                return None

        return None

    def _generate_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama API (synchronous)."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.ollama_model,
                "prompt": text[:8000]  # Reasonable limit
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()['embedding']

    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API (async)."""
        response = await self.openai_client.embeddings.create(
            model=self.openai_model,
            input=text[:8000],
            encoding_format="float"
        )
        return response.data[0].embedding

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (for OpenAI; Ollama processes one-by-one)

        Returns:
            List of embeddings (same order as input, None for failures)
        """
        if self.ollama_available:
            # Ollama: process one-by-one (fast enough for embeddings)
            embeddings = []
            for text in texts:
                try:
                    emb = self._generate_ollama_embedding(text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Failed to embed text: {e}")
                    embeddings.append(None)
            return embeddings

        elif self.openai_available:
            # OpenAI: batch processing
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    response = await self.openai_client.embeddings.create(
                        model=self.openai_model,
                        input=batch,
                        encoding_format="float"
                    )
                    embeddings.extend([d.embedding for d in response.data])
                except Exception as e:
                    logger.error(f"Batch {i}-{i+batch_size} failed: {e}")
                    embeddings.extend([None] * len(batch))
            return embeddings

        return [None] * len(texts)

    def get_provider_info(self) -> dict:
        """Get information about current embedding provider."""
        if self.ollama_available:
            return {
                'provider': 'Ollama',
                'model': self.ollama_model,
                'dimensions': self.dimensions,
                'cost': 'FREE (local)',
                'url': self.ollama_url
            }
        elif self.openai_available:
            return {
                'provider': 'OpenAI',
                'model': self.openai_model,
                'dimensions': self.dimensions,
                'cost': '$0.02/million tokens',
                'url': 'https://api.openai.com'
            }
        else:
            return {
                'provider': 'None',
                'error': 'No embedding provider available'
            }
