"""
Unified Chat Handler
====================
Single entry-point for text conversations from ANY channel:
  - physical  (Vector's microphone → STT)
  - web       (dashboard chat widget)
  - telegram  (Telegram bot)

Uses the same ResponseGenerator, ContextBuilder, memory DB and personality
engine as the physical robot so the user always talks to the same "entity".
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vector_personality.core.vector_agent import VectorAgent

logger = logging.getLogger(__name__)

# Words/phrases that mean "yes, go search"
_YES_PATTERNS = re.compile(
    r'^\s*(s[iì]|certo|ok|vai|cerca|esatto|perfetto|yes|sure|dai|fallo|per favore|'  # noqa: RUF001
    r'assolutamente|ovviamente|volentieri|claro|yep|yup|yas|sip)\b',
    re.IGNORECASE,
)

# Marker emitted by the LLM when it wants to offer a search
_SEARCH_MARKER = re.compile(r'\[CERCA:\s*(.+?)\]', re.IGNORECASE)


class ChatHandler:
    """Process text messages from any channel and return Vector's response."""

    def __init__(self, agent: VectorAgent):
        self.agent = agent
        # Per-channel conversation histories (keyed by channel_id)
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = 10  # 5 turns per channel
        # Pending web-search queries: chan_key → query string
        self._pending_searches: Dict[str, str] = {}

    def has_pending_search(self, channel: str, user_id: Optional[str]) -> bool:
        """Return True if *channel*/*user_id* has a pending web-search offer waiting for confirmation."""
        chan_key = f"{channel}:{user_id or 'anon'}"
        return chan_key in self._pending_searches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        text: str,
        *,
        channel: str = "web",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        notify_cb=None,
    ) -> str:
        """Generate a response to *text* coming from *channel*.

        Parameters
        ----------
        text : str
            The user's message.
        channel : str
            One of ``"physical"``, ``"web"``, ``"telegram"``.
        user_id : str | None
            Unique user identifier (Telegram user id, "web", face_id …).
        user_name : str | None
            Human-readable name (Telegram first_name, face name …).

        Returns
        -------
        str
            Vector's response text.
        """
        text = (text or "").strip()
        if not text:
            return "Non ho capito nulla."

        a = self.agent
        chan_key = f"{channel}:{user_id or 'anon'}"
        history = self._histories.setdefault(chan_key, [])

        # ── Web search: user is confirming a pending search offer ──────────
        pending_query = self._pending_searches.get(chan_key)
        if pending_query and _YES_PATTERNS.match(text):
            return await self._execute_pending_search(
                chan_key=chan_key,
                query=pending_query,
                history=history,
                context_builder=a.context_builder,
                reasoning_engine=a.reasoning_engine,
                llm_client=getattr(a, "llm_client", None),
                personality=a.personality,
                mood=a.memory.current_mood if a.memory else 50,
                db=a.db,
                user_id=user_id,
                channel=channel,
                user_name=user_name,
                notify_cb=notify_cb,
            )
        # Any non-yes reply clears the pending search offer
        self._pending_searches.pop(chan_key, None)

        # --- build context exactly like _handle_user_speech ---
        context: Dict[str, Any] = {}
        if a.reasoning_engine:
            context = await a.reasoning_engine.assemble_context()

        if a.context_builder:
            mem_result = await a.context_builder.build_conversation_context(user_text=text)
            context["memory_context"] = mem_result.get("memory", "")
            if mem_result.get("user_facts"):
                context["user_facts"] = mem_result["user_facts"]
            if mem_result.get("memory_recall_hint"):
                context["memory_recall_hint"] = mem_result["memory_recall_hint"]

        # Scene description (available only when Vector is ON)
        robot_online = self._robot_online()
        if robot_online and a.scene_descriptor and a.scene_descriptor.last_description:
            context["scene_description"] = a.scene_descriptor.last_description
            if a.scene_descriptor.last_change_description:
                context["scene_change"] = a.scene_descriptor.last_change_description
            if a.scene_descriptor.visual_memory:
                vm_lines = []
                for v in list(a.scene_descriptor.visual_memory.values())[-6:]:
                    desc = v["description"]
                    label = v.get("user_label", "")
                    if label:
                        vm_lines.append(f"{desc} (il proprietario mi ha detto che è: {label})")
                    else:
                        vm_lines.append(desc)
                context["visual_memory"] = "; ".join(vm_lines)

        # Channel awareness — let Vector know how the user is talking to him
        channel_note = self._channel_context(channel, user_name, robot_online)
        if channel_note:
            context["channel_note"] = channel_note

        # --- generate response ---
        llm_client = getattr(a, "llm_client", None)
        if not llm_client:
            return "I miei sistemi di ragionamento non sono disponibili al momento."

        from vector_personality.cognition.response_generator import ResponseGenerator

        response_gen = ResponseGenerator(
            openai_client=llm_client,
            personality_module=a.personality,
        )
        mood = a.memory.current_mood if a.memory else 50

        response = await response_gen.generate_response(
            user_input=text,
            conversation_history=history[-self._max_history:],
            context=context,
            mood=mood,
            response_style="chat",
        )

        # ── Parse [CERCA: query] marker from LLM response ─────────────────
        search_match = _SEARCH_MARKER.search(response)
        if search_match:
            search_query = search_match.group(1).strip()
            # Strip the marker from the visible response
            visible_response = _SEARCH_MARKER.sub("", response).strip()
            # Append a natural offer to search
            visible_response = visible_response.rstrip(".") + " — vuoi che cerchi?"
            self._pending_searches[chan_key] = search_query
            logger.info(f"[WebSearch] Pending search queued for {chan_key!r}: {search_query!r}")
            response = visible_response

        # Update per-channel history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        # Persist to DB — resolve channel identity to canonical user ID
        raw_id = user_id or str(uuid.uuid4())
        registry = getattr(a, "user_registry", None)
        speaker_id = registry.resolve(raw_id) if registry else raw_id
        if speaker_id != raw_id:
            logger.debug(f"[UserRegistry] {channel} {raw_id!r} → {speaker_id!r}")
        if a.db:
            try:
                await a.db.store_conversation(
                    speaker_id=speaker_id,
                    text=text,
                    room_id=getattr(a.memory, "current_room_id", None),
                    response_text=response,
                    vector_db=getattr(a, "vector_db", None),
                    embedding_gen=getattr(a, "embedding_gen", None),
                )
            except Exception as e:
                logger.warning(f"Failed to persist {channel} conversation: {e}")

            # Extract and store personal facts (fire-and-forget)
            try:
                from vector_personality.memory.fact_extractor import FactExtractor
                llm = getattr(a, "llm_client", None)
                if llm:
                    stable_id = user_id or f"{channel}_default"
                    extractor = FactExtractor(a.db, llm, getattr(a, "entity_memory", None))
                    asyncio.create_task(
                        extractor.extract_and_store(text, speaker_id=stable_id)
                    )
            except Exception as e:
                logger.debug(f"Fact extraction skipped: {e}")

        # If the physical robot is ON, optionally speak the response aloud
        if robot_online and channel in ("telegram", "web") and a.tts:
            try:
                if a.audio_processor:
                    a.audio_processor.discard_pending_utterances()
                await a._ensure_behavior_control()
                await a.tts.speak(response)
                if a.scene_descriptor:
                    a.scene_descriptor.mark_spoke()
            except Exception as e:
                logger.warning(f"TTS failed for {channel} message: {e}")

        logger.info(f"[{channel}] user={user_name or user_id or '?'}: \"{text}\" → \"{response}\"")
        return response

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    async def _execute_pending_search(
        self,
        *,
        chan_key: str,
        query: str,
        history: List[Dict[str, str]],
        context_builder: Any,
        reasoning_engine: Any,
        llm_client: Any,
        personality: Any,
        mood: int,
        db: Any,
        user_id: Optional[str],
        channel: str,
        user_name: Optional[str],
        notify_cb=None,
    ) -> str:
        """Perform the web search and generate a response with the results."""
        from vector_personality.cognition.web_searcher import web_search
        from vector_personality.cognition.response_generator import ResponseGenerator

        self._pending_searches.pop(chan_key, None)
        logger.info(f"[WebSearch] Searching: {query!r}")

        # Notify the caller that the search has started (e.g. Telegram sends "🔍 Sto cercando…")
        # For web we skip this because HTTP is single request/response — we prefix the final result instead.
        if notify_cb is not None:
            try:
                await notify_cb("🔍 Sto cercando, un momento...")
            except Exception as _e:
                logger.debug(f"notify_cb failed: {_e}")

        web_results = await web_search(query)
        if not web_results:
            web_results = "(Nessun risultato trovato.)"
            logger.info("[WebSearch] No results returned")
        else:
            logger.info(f"[WebSearch] Got {len(web_results)} chars of results")

        # Build context with web results injected
        context: Dict[str, Any] = {}
        if reasoning_engine:
            context = await reasoning_engine.assemble_context()
        if context_builder:
            mem_result = await context_builder.build_conversation_context(user_text=query)
            context["memory_context"] = mem_result.get("memory", "")
            if mem_result.get("user_facts"):
                context["user_facts"] = mem_result["user_facts"]
            if mem_result.get("memory_recall_hint"):
                context["memory_recall_hint"] = mem_result["memory_recall_hint"]
        context["web_results"] = web_results

        if not llm_client:
            clean = web_results.split("\n")[0] if web_results else "Nessun risultato."
            return f"Ho trovato questo: {clean}"

        response_gen = ResponseGenerator(
            openai_client=llm_client,
            personality_module=personality,
        )
        response = await response_gen.generate_response(
            user_input=f"Cerca: {query}",
            conversation_history=history[-self._max_history:],
            context=context,
            mood=mood,
            response_style="chat",
        )

        # Update history
        history.append({"role": "user", "content": f"[cerca] {query}"})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        # Persist
        if db:
            try:
                raw_id = user_id or str(uuid.uuid4())
                await db.store_conversation(
                    speaker_id=raw_id,
                    text=f"[cerca] {query}",
                    room_id=None,
                    response_text=response,
                )
            except Exception as exc:
                logger.debug(f"Failed to persist web-search conversation: {exc}")

        logger.info(f"[WebSearch] Response ({len(response)} chars): {response!r}")
        # For web channel (no notify_cb), prefix with a ✅ so the user knows the search is done
        if notify_cb is None and channel == "web":
            response = "✅ Ricerca completata!\n\n" + response
        return response

    # ------------------------------------------------------------------
    # Camera snapshot
    # ------------------------------------------------------------------

    def get_camera_snapshot_b64(self) -> Optional[str]:
        """Return a base64-encoded JPEG from Vector's camera, or None."""
        if not self._robot_online():
            return None
        try:
            cam_img = self.agent.robot.camera.latest_image
            if cam_img and cam_img.raw_image is not None:
                from vector_personality.cognition.ollama_client import OllamaClient
                return OllamaClient._pil_to_base64(cam_img.raw_image, max_size=640)
        except Exception as e:
            logger.warning(f"Camera snapshot failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _robot_online(self) -> bool:
        """Check if the physical robot is connected and running."""
        a = self.agent
        return bool(
            getattr(a, "running", False)
            and getattr(a, "robot", None)
            and getattr(a.robot, "conn", None)
        )

    @staticmethod
    def _channel_context(
        channel: str, user_name: Optional[str], robot_online: bool
    ) -> str:
        """Build a short note for the system prompt about the interaction channel."""
        parts = []
        if channel == "telegram":
            parts.append(
                "Stai comunicando con l'utente via Telegram (chat testuale remota)."
            )
            if not robot_online:
                parts.append(
                    "Il tuo corpo fisico (robot Vector) è spento. "
                    "Rispondi come la tua mente digitale che vive sul PC."
                )
        elif channel == "web":
            parts.append(
                "Stai comunicando con l'utente tramite la chat del dashboard."
            )
            if not robot_online:
                parts.append(
                    "Il tuo corpo fisico (robot Vector) è spento. "
                    "Rispondi come la tua mente digitale che vive sul PC."
                )
        if user_name:
            parts.append(f"L'utente si chiama {user_name}.")
        else:
            parts.append(
                "Non sai ancora come si chiama l'utente. "
                "Se è naturale nel contesto, chiedi il nome una volta sola — non in modo robotico."
            )
        return " ".join(parts) if parts else ""
