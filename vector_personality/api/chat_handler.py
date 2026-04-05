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
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vector_personality.core.vector_agent import VectorAgent

logger = logging.getLogger(__name__)


class ChatHandler:
    """Process text messages from any channel and return Vector's response."""

    def __init__(self, agent: VectorAgent):
        self.agent = agent
        # Per-channel conversation histories (keyed by channel_id)
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = 10  # 5 turns per channel

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

        # --- build context exactly like _handle_user_speech ---
        context: Dict[str, Any] = {}
        if a.reasoning_engine:
            context = await a.reasoning_engine.assemble_context()

        if a.context_builder:
            memory_ctx = await a.context_builder.build_conversation_context(user_text=text)
            context["memory_context"] = memory_ctx

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
        )

        # Update per-channel history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        if len(history) > self._max_history:
            history[:] = history[-self._max_history:]

        # Persist to DB
        speaker_id = user_id or str(uuid.uuid4())
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
                "Stai comunicando con l'utente tramite la chat web del dashboard."
            )
            if not robot_online:
                parts.append(
                    "Il tuo corpo fisico (robot Vector) è spento. "
                    "Rispondi come la tua mente digitale che vive sul PC."
                )
        if user_name:
            parts.append(f"L'utente si chiama {user_name}.")
        return " ".join(parts) if parts else ""
