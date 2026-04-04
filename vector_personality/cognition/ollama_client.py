"""
Ollama Client Module

Local LLM client via Ollama HTTP API (OpenAI-compatible).
Replaces Groq/OpenAI cloud APIs with fully local inference.

Features:
- OpenAI-compatible chat completions (drop-in replacement)
- Streaming support for real-time TTS
- Zero cost, full privacy, no data leaves the machine
- Automatic model availability checking

Phase 11 - Local AI Migration
"""

import asyncio
import logging
import random
from typing import Optional, List, Dict, Any, AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Local LLM client using Ollama HTTP API.

    Drop-in replacement for GroqClient / OpenAIClient.
    Uses the ``/api/chat`` endpoint (OpenAI-compatible).

    Attributes:
        base_url: Ollama server URL (default http://localhost:11434)
        default_model: Model tag (e.g. 'mistral-small3.2:latest')
    """

    FALLBACK_RESPONSES = [
        "Sto avendo problemi con il mio cervello locale. Puoi riprovare?",
        "I miei circuiti sono un po' sovraccarichi. Dammi un momento!",
        "Hmm, devo pensarci su. Puoi chiedere di nuovo?",
        "Sto riscontrando alcune difficoltà tecniche. Mi dispiace!",
    ]

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "gemma3:4b",
        max_retries: int = 2,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        logger.info(
            f"OllamaClient initialized: model={default_model}, "
            f"url={self.base_url}"
        )

    # ── public: check availability ──────────────────────────

    async def check_model_available(self) -> bool:
        """Return True if the default model is pulled and Ollama is running."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    names = [m["name"] for m in data.get("models", [])]
                    available = self.default_model in names
                    if not available:
                        logger.warning(
                            f"Model '{self.default_model}' not found. "
                            f"Available: {names}"
                        )
                    return available
        except Exception as e:
            logger.error(f"Ollama connectivity check failed: {e}")
            return False

    # ── public: chat completion (non-streaming) ─────────────

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        Generate a chat completion via Ollama ``/api/chat``.

        Interface-compatible with GroqClient.chat_completion /
        OpenAIClient.chat_completion.
        """
        model = model or self.default_model
        retries = max_retries if max_retries is not None else self.max_retries

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        f"{self.base_url}/api/chat", json=payload
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.error(
                                f"Ollama HTTP {resp.status} "
                                f"(attempt {attempt + 1}/{retries}): {body}"
                            )
                            if attempt < retries - 1:
                                await asyncio.sleep(1)
                                continue
                            return self._get_fallback_response()

                        data = await resp.json()
                        content = (
                            data.get("message", {}).get("content", "").strip()
                        )
                        total_dur = data.get("total_duration", 0) / 1e9
                        eval_count = data.get("eval_count", 0)
                        logger.info(
                            f"✅ Ollama response: {len(content)} chars, "
                            f"{eval_count} tokens, {total_dur:.2f}s"
                        )
                        return content

            except asyncio.TimeoutError:
                logger.warning(
                    f"Ollama timeout (attempt {attempt + 1}/{retries})"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                return self._get_fallback_response()

        return self._get_fallback_response()

    # ── public: streaming chat completion ───────────────────

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ) -> Optional[AsyncIterator[str]]:
        """
        Stream chat tokens from Ollama (for TTS integration).

        Returns an async iterator that yields content chunks.
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }

        try:
            session = aiohttp.ClientSession(timeout=self.timeout)
            resp = await session.post(
                f"{self.base_url}/api/chat", json=payload
            )
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Ollama stream HTTP {resp.status}: {body}")
                await resp.release()
                await session.close()
                return None

            async def _stream():
                try:
                    import json as _json
                    async for line in resp.content:
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                        except ValueError:
                            continue
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                finally:
                    await resp.release()
                    await session.close()

            return _stream()

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            return None

    # ── internal ────────────────────────────────────────────

    def _get_fallback_response(self) -> str:
        return random.choice(self.FALLBACK_RESPONSES)

    async def close(self):
        """No persistent connection to close."""
        logger.info("OllamaClient closed")


# Backward compatibility alias
LLMClient = OllamaClient
