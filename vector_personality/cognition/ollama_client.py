"""
Ollama Client Module

Local LLM client via Ollama HTTP API (OpenAI-compatible).
Replaces Groq/OpenAI cloud APIs with fully local inference.

Features:
- OpenAI-compatible chat completions (drop-in replacement)
- Streaming support for real-time TTS
- Vision/multimodal support (base64 images)
- Zero cost, full privacy, no data leaves the machine
- Automatic model availability checking

Phase 11 - Local AI Migration
"""

import asyncio
import base64
import concurrent.futures
import io
import logging
import random
from typing import Optional, List, Dict, Any, AsyncIterator

# Dedicated executor for CPU-bound PIL image enhancement.
# Isolated from the asyncio loop's default executor so that loop lifecycle
# events (shutdown_default_executor, loop.close) never affect image processing.
_PIL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="ollama-img"
)

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

    @staticmethod
    def _pil_to_base64(image, max_size: int = 640) -> str:
        """Convert a PIL Image to a base64-encoded JPEG string, enhanced for VLM."""
        from PIL import Image as PILImage, ImageEnhance, ImageFilter

        if isinstance(image, PILImage.Image):
            img = image.copy()  # MUST copy — thumbnail() is in-place and would corrupt the SDK's cached image
        else:
            # numpy array → PIL (already a new object)
            img = PILImage.fromarray(image)

        # Resize keeping aspect ratio — keep max_size=640 to preserve Vector's native resolution
        img.thumbnail((max_size, max_size))

        # Enhance image quality to help the VLM handle Vector's low-quality camera
        img = ImageEnhance.Contrast(img).enhance(1.4)   # boost contrast
        img = ImageEnhance.Sharpness(img).enhance(2.5)  # sharpen edges
        img = ImageEnhance.Brightness(img).enhance(1.1) # slight brightness lift

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── public: vision completion ───────────────────────────

    async def vision_completion(
        self,
        prompt: str,
        image,
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 200,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """
        Send an image + text prompt to a multimodal Ollama model.

        Args:
            prompt: Text instruction (e.g. "Describe what you see")
            image: PIL Image or numpy array (will be resized & base64-encoded)
            model: Vision model tag (default: gemma3:12b)
            temperature: Sampling temperature
            max_tokens: Max response tokens
            timeout_seconds: Override default timeout for slow first-run

        Returns:
            Model's text description of the image
        """
        model = model or "gemma3:12b"
        # Run CPU-intensive PIL enhancement in a dedicated thread executor so the
        # event loop stays free.  Uses an isolated executor (not the loop default)
        # to avoid 'cannot schedule new futures after shutdown' errors.
        loop = asyncio.get_running_loop()
        img_b64 = await loop.run_in_executor(_PIL_EXECUTOR, self._pil_to_base64, image)
        logger.info(
            f"\U0001f441\ufe0f VLM request: model={model}, "
            f"image_b64_len={len(img_b64)}, prompt_len={len(prompt)}"
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }

        timeout = aiohttp.ClientTimeout(
            total=timeout_seconds or self.timeout.total
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Ollama vision HTTP {resp.status}: {body}")
                        return ""
                    data = await resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    total_dur = data.get("total_duration", 0) / 1e9
                    eval_count = data.get("eval_count", 0)
                    prompt_eval = data.get("prompt_eval_count", 0)
                    logger.info(
                        f"\U0001f441\ufe0f VLM response: {len(content)} chars, {total_dur:.2f}s, "
                        f"prompt_tokens={prompt_eval}, eval_tokens={eval_count}"
                    )
                    if total_dur < 1.0:
                        logger.warning(
                            f"\U0001f441\ufe0f VLM suspiciously fast ({total_dur:.2f}s) — "
                            f"image may not be processed. Full response keys: {list(data.keys())}"
                        )
                    return content
        except asyncio.TimeoutError:
            logger.warning("Ollama vision timeout")
            return ""
        except Exception as e:
            logger.error(f"Ollama vision error: {e}")
            return ""

    async def close(self):
        """No persistent connection to close."""
        logger.info("OllamaClient closed")


# Backward compatibility alias
LLMClient = OllamaClient
