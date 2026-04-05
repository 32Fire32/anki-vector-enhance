"""
Standalone Chat Handler
=======================
Handles text conversations without a connected Vector robot.
Works with just Ollama + ChromaDB so the user can always chat with
Vector's "mind", even when the physical robot is powered off.

Initialized once at application startup in execute.py and reused for
every web-chat request when the agent thread is not running.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StandaloneChatHandler:
    """Chat with Vector's AI without requiring the robot or the agent thread."""

    def __init__(self):
        self._ready = False
        self._llm = None
        self._personality = None
        self._context_builder = None
        self._db = None
        self._memory = None
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = 10

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialise AI components from environment variables.

        Returns True on success, False if critical components (LLM) are unavailable.
        Resets _ready to False on failure so the next call retries.
        """
        if self._ready:
            return True

        # --- LLM client ---
        try:
            from vector_personality.cognition.ollama_client import OllamaClient
            import requests as _req

            url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
            try:
                r = _req.get(f"{url}/api/tags", timeout=5)
                available = [m["name"] for m in r.json().get("models", [])]
                if model not in available:
                    logger.warning(
                        f"[StandaloneChat] model '{model}' not found; available: {available}"
                    )
                    if available:
                        model = available[0]
                        logger.info(f"[StandaloneChat] falling back to '{model}'")
                    else:
                        logger.error("[StandaloneChat] Ollama has no models loaded")
                        return False
            except Exception as e:
                logger.warning(f"[StandaloneChat] cannot reach Ollama: {e}")
                return False

            self._llm = OllamaClient(
                base_url=url,
                default_model=model,
                timeout_seconds=60,
            )
            logger.info(f"[StandaloneChat] LLM ready: {model}")
        except Exception as e:
            logger.error(f"[StandaloneChat] LLM init failed: {e}")
            return False

        # --- Working memory (lightweight, in-process) ---
        try:
            from vector_personality.memory.working_memory import WorkingMemory
            self._memory = WorkingMemory()
        except Exception as e:
            logger.warning(f"[StandaloneChat] WorkingMemory unavailable: {e}")
            self._memory = None

        # --- ChromaDB (persistent memory, optional) ---
        try:
            from vector_personality.memory.chromadb_connector import ChromaDBConnector
            chromadb_dir = os.getenv("CHROMADB_DIR", "./vector_memory_chroma")
            self._db = ChromaDBConnector(persist_directory=chromadb_dir)
            logger.info(f"[StandaloneChat] ChromaDB ready at {chromadb_dir}")
        except Exception as e:
            logger.warning(f"[StandaloneChat] ChromaDB unavailable (no persistent memory): {e}")
            self._db = None

        # --- Personality ---
        try:
            from vector_personality.core.personality import PersonalityModule
            self._personality = PersonalityModule(self._db)
        except Exception as e:
            logger.warning(f"[StandaloneChat] PersonalityModule unavailable: {e}")
            self._personality = None

        # --- Context builder (startup summary + semantic retrieval, optional) ---
        try:
            from vector_personality.cognition.context_builder import ContextBuilder
            if self._db and self._memory:
                self._context_builder = ContextBuilder(
                    db_connector=self._db,
                    working_memory=self._memory,
                    groq_client=self._llm,
                )
        except Exception as e:
            logger.warning(f"[StandaloneChat] ContextBuilder unavailable: {e}")
            self._context_builder = None

        self._ready = True
        logger.info("[StandaloneChat] ready — web chat available without robot")
        return True

    # ------------------------------------------------------------------
    # Public API (mirrors ChatHandler.handle_message signature)
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        text: str,
        *,
        channel: str = "web",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> str:
        if not self._ready:
            ok = self.initialize()
            if not ok:
                return (
                    "Il mio sistema di ragionamento non è disponibile. "
                    "Assicurati che Ollama sia in esecuzione e riprova."
                )
        text = (text or "").strip()
        if not text:
            return "Non ho capito nulla."

        chan_key = f"{channel}:{user_id or 'anon'}"
        history = self._histories.setdefault(chan_key, [])

        # Build context
        context: Dict[str, Any] = {}
        if self._context_builder:
            try:
                mem_ctx = await self._context_builder.build_conversation_context(
                    user_text=text
                )
                context["memory_context"] = mem_ctx
            except Exception as e:
                logger.debug(f"[StandaloneChat] context build error: {e}")

        # Channel note
        parts = [
            "Stai comunicando con l'utente tramite la chat web del dashboard.",
            "Il tuo corpo fisico (robot Vector) è spento. "
            "Rispondi come la tua mente digitale che vive sul PC.",
        ]
        if user_name:
            parts.append(f"L'utente si chiama {user_name}.")
        context["channel_note"] = " ".join(parts)

        # Generate response
        from vector_personality.cognition.response_generator import ResponseGenerator

        mood = getattr(self._memory, "current_mood", 50) if self._memory else 50
        # Personality is optional; ResponseGenerator handles None gracefully if
        # we give it a fallback object with an `effective_traits` property.
        personality = self._personality or _FallbackPersonality()

        response_gen = ResponseGenerator(
            openai_client=self._llm,
            personality_module=personality,
        )

        try:
            response = await response_gen.generate_response(
                user_input=text,
                conversation_history=history[-self._max_history:],
                context=context,
                mood=mood,
            )
        except Exception as e:
            logger.error(f"[StandaloneChat] response generation failed: {e}")
            response = (
                "Scusa, ho avuto un problema nel generare una risposta. Riprova."
            )

        # Update history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        # Persist to DB
        if self._db:
            speaker_id = user_id or str(uuid.uuid4())
            try:
                await self._db.store_conversation(
                    speaker_id=speaker_id,
                    text=text,
                    room_id=None,
                    response_text=response,
                )
            except Exception as e:
                logger.debug(f"[StandaloneChat] failed to persist conversation: {e}")

        logger.info(
            f"[StandaloneChat/{channel}] user={user_name or user_id or '?'}: "
            f'"{text}" → "{response}"'
        )
        return response


# ---------------------------------------------------------------------------
# Minimal fallback if PersonalityModule fails to load
# ---------------------------------------------------------------------------

class _FallbackPersonality:
    """Minimal personality stub used when PersonalityModule is unavailable."""

    class _Traits:
        curiosity = 0.7
        sassiness = 0.3
        friendliness = 0.7
        vitality = 0.8

    effective_traits = _Traits()
