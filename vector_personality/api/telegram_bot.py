"""
Telegram Bot for Vector
========================
Allows remote interaction with Vector's AI brain via Telegram.
Works regardless of whether the robot is on or off — uses ChatHandler
(full context) when the agent is running, or StandaloneChatHandler
(Ollama + ChromaDB only) when the robot is off.

Commands:
  /start   — Welcome message
  /photo   — Get a snapshot from Vector's camera (if online)
  /status  — Check if Vector is online
  (free text) — Chat with Vector

Configuration:
  Set TELEGRAM_BOT_TOKEN in api.env

Dependencies:
  pip install python-telegram-bot
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import TYPE_CHECKING, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

logger = logging.getLogger(__name__)


class VectorTelegramBot:
    """Async Telegram bot that works with or without the robot.

    Accepts the AgentBridge object and routes messages dynamically:
    - Agent running  → ChatHandler (full context, TTS, visual memory)
    - Agent stopped  → StandaloneChatHandler (Ollama + ChromaDB only)
    
    Access control: Only whitelisted Telegram user IDs can use the bot.
    Set TELEGRAM_ALLOWED_USER_IDS in api.env (comma-separated list of user IDs).
    """

    def __init__(self, token: str, bridge):
        self.token = token
        self.bridge = bridge
        self._app: Optional[Application] = None
        
        # Load whitelist from environment
        allowed_str = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        self._allowed_user_ids = set()
        if allowed_str:
            try:
                self._allowed_user_ids = set(int(uid.strip()) for uid in allowed_str.split(",") if uid.strip())
                logger.info(f"✅ Telegram whitelist: {len(self._allowed_user_ids)} user(s) allowed")
            except ValueError as e:
                logger.warning(f"⚠️ Invalid TELEGRAM_ALLOWED_USER_IDS: {e}")
    
    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is whitelisted."""
        if not self._allowed_user_ids:
            # No whitelist configured — allow all (backward compatibility)
            logger.warning("⚠️ TELEGRAM_ALLOWED_USER_IDS not set — bot is PUBLIC!")
            return True
        return user_id in self._allowed_user_ids

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the Application and start polling (non-blocking)."""
        self._app = (
            Application.builder()
            .token(self.token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("photo", self._cmd_photo))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True, timeout=5)
        logger.info("Telegram bot started — polling for messages")

    async def stop(self) -> None:
        """Gracefully shut down the bot."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning(f"Telegram bot shutdown error: {e}")
            logger.info("Telegram bot stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _robot_online(self) -> bool:
        ch = self.bridge._chat_handler
        return ch is not None and ch._robot_online()

    async def _resolve_message(
        self,
        text: str,
        channel: str,
        user_id: Optional[str],
        user_name: Optional[str],
    ) -> str:
        """Route to ChatHandler (agent running) or StandaloneChatHandler (agent stopped)."""
        chat_handler = self.bridge._chat_handler
        loop = self.bridge._loop

        if chat_handler and loop and loop.is_running():
            # Agent is running in a separate thread — submit to its loop and
            # await the result in the current (uvicorn) event loop.
            fut = asyncio.run_coroutine_threadsafe(
                chat_handler.handle_message(
                    text, channel=channel, user_id=user_id, user_name=user_name
                ),
                loop,
            )
            return await asyncio.wrap_future(fut)

        sc = self.bridge._standalone_chat
        if sc:
            return await sc.handle_message(
                text, channel=channel, user_id=user_id, user_name=user_name
            )

        return (
            "Il mio sistema di ragionamento non è disponibile. "
            "Assicurati che Ollama sia in esecuzione e riprova."
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        name = update.effective_user.first_name or "utente"
        online = "🟢 online" if self._robot_online() else "🔴 offline"
        await update.message.reply_text(
            f"Ciao {name}! Sono Vector 🤖\n"
            f"Corpo fisico: {online}\n\n"
            "Scrivimi qualsiasi cosa e ti risponderò.\n"
            "Comandi:\n"
            "  /photo — scatta una foto dalla mia telecamera\n"
            "  /status — controlla se sono acceso"
        )

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if self._robot_online():
            await update.message.reply_text(
                "🟢 Sono acceso e operativo! Il mio corpo fisico è attivo."
            )
        else:
            await update.message.reply_text(
                "🔴 Il mio corpo fisico è spento, ma la mia mente digitale "
                "è qui sul PC. Possiamo comunque parlare!"
            )

    async def _cmd_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id
        if not self._is_user_allowed(user_id):
            logger.warning(f"🚫 Unauthorized /photo attempt from user {user_id}")
            await update.message.reply_text("Non sei autorizzato a usare questo bot.")
            return
        
        ch = self.bridge._chat_handler
        b64 = ch.get_camera_snapshot_b64() if ch else None
        if b64 is None:
            await update.message.reply_text(
                "📷 Non posso scattare una foto — il mio corpo fisico è spento."
            )
            return
        try:
            image_bytes = base64.b64decode(b64)
            await update.message.reply_photo(
                photo=io.BytesIO(image_bytes),
                caption="📷 Ecco cosa vedo in questo momento!",
            )
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            await update.message.reply_text(
                "📷 Errore nell'invio della foto, riprova."
            )

    # ------------------------------------------------------------------
    # Free-text chat
    # ------------------------------------------------------------------

    async def _on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user = update.effective_user
        user_id = user.id
        
        # Check if user is allowed
        if not self._is_user_allowed(user_id):
            logger.warning(f"🚫 Unauthorized message from user {user_id}: {update.message.text!r}")
            await update.message.reply_text("Non sei autorizzato a usare questo bot.")
            return
        
        text = update.message.text or ""
        user_name = user.first_name or None

        # Resolve Telegram user ID to canonical ID via UserRegistry (if available)
        resolved_user_id = str(user_id)
        try:
            bridge = self.bridge
            agent = getattr(getattr(bridge, "_chat_handler", None), "agent", None)
            registry = getattr(agent, "user_registry", None)
            if registry is None:
                sc = getattr(bridge, "_standalone_chat", None)
                registry = getattr(sc, "_user_registry", None)
            if registry:
                resolved_user_id = registry.resolve(str(user_id))
        except Exception:
            pass

        # Show typing indicator while generating response
        await update.message.chat.send_action("typing")

        # Check if the user is confirming a pending web-search offer.
        # If so, send an immediate acknowledgement, then the actual result.
        import re as _re
        _yes_pat = _re.compile(
            r'^\s*(s[ì i]|certo|ok|vai|cerca|esatto|perfetto|yes|sure|dai|fallo|per favore|'
            r'assolutamente|ovviamente|volentieri|yep|yup)\b',
            _re.IGNORECASE,
        )
        chat_handler = self.bridge._chat_handler
        sc = getattr(self.bridge, "_standalone_chat", None)
        is_search_confirmation = _yes_pat.match(text) and (
            (chat_handler is not None and chat_handler.has_pending_search("telegram", resolved_user_id))
            or (sc is not None and sc.has_pending_search("telegram", resolved_user_id))
        )

        if is_search_confirmation:
            # Send an immediate "searching..." message so the user isn't left in silence
            await update.message.reply_text("🔍 Sto cercando, un momento...")
            await update.message.chat.send_action("typing")

        response = await self._resolve_message(
            text,
            channel="telegram",
            user_id=resolved_user_id,
            user_name=user_name,
        )
        await update.message.reply_text(response)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def start_telegram_bot(bridge) -> Optional[VectorTelegramBot]:
    """Create and start the Telegram bot if TELEGRAM_BOT_TOKEN is set.

    Accepts the AgentBridge object so the bot can route messages dynamically.
    Returns the bot instance, or None if no token is configured.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.info("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")
        return None

    bot = VectorTelegramBot(token, bridge)
    await bot.start()
    return bot
