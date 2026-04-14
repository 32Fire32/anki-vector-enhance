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
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Same patterns as chat_handler.py
_YES_PATTERNS = re.compile(
    r'^\s*(s[i\xec]|certo|ok|vai|cerca|esatto|perfetto|yes|sure|dai|fallo|per favore|'
    r'assolutamente|ovviamente|volentieri|yep|yup)\b',
    re.IGNORECASE,
)
_SEARCH_MARKER = re.compile(r'\[CERCA:\s*(.+?)\]', re.IGNORECASE)


class StandaloneChatHandler:
    """Chat with Vector's AI without requiring the robot or the agent thread."""

    def __init__(self):
        self._ready = False
        self._llm = None
        self._personality = None
        self._context_builder = None
        self._db = None
        self._memory = None
        self._embedder = None
        self._entity_memory = None
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = 10
        # Pending web-search queries: chan_key → query string
        self._pending_searches: Dict[str, str] = {}

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
            # Prefer STANDALONE_MODEL (larger, better quality) when robot is off;
            # fall back to OLLAMA_MODEL if not available.
            preferred = os.getenv("STANDALONE_MODEL") or os.getenv("OLLAMA_MODEL", "gemma3:4b")
            model = preferred
            try:
                r = _req.get(f"{url}/api/tags", timeout=5)
                available = [m["name"] for m in r.json().get("models", [])]
                if model not in available:
                    fallback = os.getenv("OLLAMA_MODEL", "gemma3:4b")
                    logger.warning(
                        f"[StandaloneChat] model '{model}' not found; trying fallback '{fallback}'"
                    )
                    if fallback in available:
                        model = fallback
                        logger.info(f"[StandaloneChat] using fallback model '{model}'")
                    elif available:
                        model = available[0]
                        logger.info(f"[StandaloneChat] using first available model '{model}'")
                    else:
                        logger.error("[StandaloneChat] Ollama has no models loaded")
                        return False
                else:
                    logger.info(f"[StandaloneChat] standalone model: {model}")
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

        # --- Embedding generator (optional, enables semantic search) ---
        try:
            from vector_personality.memory.embedding_generator import EmbeddingGenerator
            self._embedder = EmbeddingGenerator(ollama_url=url)
            logger.info("[StandaloneChat] EmbeddingGenerator ready")
        except Exception as e:
            logger.debug(f"[StandaloneChat] EmbeddingGenerator unavailable: {e}")
            self._embedder = None

        # --- Entity memory (structured profiles) ---
        try:
            from vector_personality.memory.entity_memory import EntityMemory
            if self._db:
                self._entity_memory = EntityMemory(self._db.client, self._embedder)
                logger.info("[StandaloneChat] EntityMemory ready")
        except Exception as e:
            logger.debug(f"[StandaloneChat] EntityMemory unavailable: {e}")
            self._entity_memory = None

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
                    entity_memory=self._entity_memory,
                    embedding_gen=self._embedder,
                )
        except Exception as e:
            logger.warning(f"[StandaloneChat] ContextBuilder unavailable: {e}")
            self._context_builder = None

        # --- User registry ---
        try:
            from vector_personality.memory.user_registry import UserRegistry
            self._user_registry = UserRegistry()
        except Exception as e:
            logger.debug(f"[StandaloneChat] UserRegistry unavailable: {e}")
            self._user_registry = None

        self._ready = True
        logger.info("[StandaloneChat] ready — web chat available without robot")
        return True

    # ------------------------------------------------------------------
    # Public API (mirrors ChatHandler.handle_message signature)
    # ------------------------------------------------------------------

    def has_pending_search(self, channel: str, user_id: Optional[str]) -> bool:
        """Return True if this channel has a pending web-search offer."""
        return f"{channel}:{user_id or 'anon'}" in self._pending_searches

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

        # ─ Web search: user is confirming a pending search offer ─────────────
        pending_query = self._pending_searches.get(chan_key)
        if pending_query and _YES_PATTERNS.match(text):
            self._pending_searches.pop(chan_key, None)
            return await self._execute_web_search(
                chan_key=chan_key,
                query=pending_query,
                history=history,
                user_id=user_id,
                channel=channel,
                mood=getattr(self._memory, "current_mood", 50) if self._memory else 50,
            )
        # Any non-yes reply clears the pending offer
        self._pending_searches.pop(chan_key, None)

        # Build context
        context: Dict[str, Any] = {}
        if self._context_builder:
            try:
                mem_result = await self._context_builder.build_conversation_context(
                    user_text=text
                )
                context["memory_context"] = mem_result.get("memory", "")
                if mem_result.get("user_facts"):
                    context["user_facts"] = mem_result["user_facts"]
                if mem_result.get("memory_recall_hint"):
                    context["memory_recall_hint"] = mem_result["memory_recall_hint"]
            except Exception as e:
                logger.debug(f"[StandaloneChat] context build error: {e}")

        # Channel note
        parts = [
            "Stai comunicando con l'utente tramite la chat del dashboard.",
            "Il tuo corpo fisico (robot Vector) è spento. "
            "Rispondi come la tua mente digitale che vive sul PC.",
        ]
        if user_name:
            parts.append(f"L'utente si chiama {user_name}.")
        else:
            parts.append(
                "Non sai ancora come si chiama l'utente. "
                "Se è naturale nel contesto della conversazione, chiedi il nome. "
                "Non farlo in modo robotico o ripetuto — solo una volta, quando ha senso."
            )
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
                response_style="chat",
            )
        except Exception as e:
            logger.error(f"[StandaloneChat] response generation failed: {e}")
            response = (
                "Scusa, ho avuto un problema nel generare una risposta. Riprova."
            )

        # ─ Parse [CERCA: query] marker ───────────────────────────────
        sm = _SEARCH_MARKER.search(response)
        if sm:
            search_query = sm.group(1).strip()
            response = _SEARCH_MARKER.sub("", response).strip()
            response = response.rstrip(".") + " — vuoi che cerchi?"
            self._pending_searches[chan_key] = search_query
            logger.info(f"[WebSearch][standalone] Pending search queued: {search_query!r}")

        # Update history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        # Persist to DB — resolve channel identity to canonical user ID
        if self._db:
            raw_id = user_id or str(uuid.uuid4())
            speaker_id = self._user_registry.resolve(raw_id) if self._user_registry else raw_id
            if speaker_id != raw_id:
                logger.debug(f"[UserRegistry] standalone {raw_id!r} → {speaker_id!r}")
            try:
                await self._db.store_conversation(
                    speaker_id=speaker_id,
                    text=text,
                    room_id=None,
                    response_text=response,
                )
            except Exception as e:
                logger.debug(f"[StandaloneChat] failed to persist conversation: {e}")

            # Extract and store personal facts (fire-and-forget)
            try:
                import asyncio as _aio
                from vector_personality.memory.fact_extractor import FactExtractor
                stable_id = user_id or f"{channel}_default"
                _extractor = FactExtractor(self._db, self._llm, self._entity_memory)
                _aio.create_task(
                    _extractor.extract_and_store(text, speaker_id=stable_id)
                )
            except Exception as e:
                logger.debug(f"[StandaloneChat] fact extraction skipped: {e}")

        logger.info(
            f"[StandaloneChat/{channel}] user={user_name or user_id or '?'}: "
            f'"{text}" → "{response}"'
        )
        return response

    # ------------------------------------------------------------------
    # Web search helper
    # ------------------------------------------------------------------

    async def _execute_web_search(
        self,
        *,
        chan_key: str,
        query: str,
        history: List[Dict[str, str]],
        user_id: Optional[str],
        channel: str,
        mood: int,
    ) -> str:
        from vector_personality.cognition.web_searcher import web_search
        from vector_personality.cognition.response_generator import ResponseGenerator

        logger.info(f"[WebSearch][standalone] Searching: {query!r}")
        web_results = await web_search(query)
        if not web_results:
            web_results = "(Nessun risultato trovato.)"
            logger.info("[WebSearch][standalone] No results returned")
        else:
            logger.info(f"[WebSearch][standalone] Got {len(web_results)} chars of results")

        context: Dict[str, Any] = {"web_results": web_results}
        if self._context_builder:
            try:
                mem_result = await self._context_builder.build_conversation_context(user_text=query)
                context["memory_context"] = mem_result.get("memory", "")
                if mem_result.get("user_facts"):
                    context["user_facts"] = mem_result["user_facts"]
                if mem_result.get("memory_recall_hint"):
                    context["memory_recall_hint"] = mem_result["memory_recall_hint"]
            except Exception as e:
                logger.debug(f"[WebSearch][standalone] context error: {e}")

        personality = self._personality or _FallbackPersonality()
        response_gen = ResponseGenerator(
            openai_client=self._llm,
            personality_module=personality,
        )
        try:
            response = await response_gen.generate_response(
                user_input=f"Cerca: {query}",
                conversation_history=history[-self._max_history:],
                context=context,
                mood=mood,
                response_style="chat",
            )
        except Exception as e:
            logger.error(f"[WebSearch][standalone] response generation failed: {e}")
            snippet = web_results.split("\n")[0][:200]
            response = f"Ho trovato questo: {snippet}"

        # Prefix so the user knows the search completed
        response = "✅ Ricerca completata!\n\n" + response

        history.append({"role": "user", "content": f"[cerca] {query}"})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        if self._db:
            try:
                raw_id = user_id or str(uuid.uuid4())
                await self._db.store_conversation(
                    speaker_id=raw_id,
                    text=f"[cerca] {query}",
                    room_id=None,
                    response_text=response,
                )
            except Exception as exc:
                logger.debug(f"[WebSearch][standalone] persist failed: {exc}")

        logger.info(f"[WebSearch][standalone] Response: {response!r}")
        return response

class _FallbackPersonality:
    """Minimal personality stub used when PersonalityModule is unavailable."""

    class _Traits:
        curiosity = 0.7
        sassiness = 0.3
        friendliness = 0.7
        vitality = 0.8

    effective_traits = _Traits()
