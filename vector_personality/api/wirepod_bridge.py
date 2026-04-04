"""
Wire-pod ↔ Agent Bridge  (OpenAI-compatible HTTP server)
=========================================================
Wire-pod already does STT correctly via its own Vosk/Whisper pipeline.
This module exposes a minimal OpenAI-compatible endpoint so wire-pod
can forward the transcribed speech text to our personality engine.

How it works:
  1. User says "Hey Vector, <command>"
  2. Wire-pod does wake-word detection + STT → gets transcript
  3. Wire-pod's knowledge-graph sends POST /v1/chat/completions here
  4. We run the transcript through our reasoning/memory/emotion pipeline
  5. We return the response text in streaming SSE format
  6. Wire-pod has Vector speak the response via its native TTS

Configuration in wire-pod web UI (http://192.168.1.6:8080):
  Knowledge Graph → Provider: Custom
  URL: http://<this-PC-IP>:8181
  Key: (any non-empty string, e.g. "local")
  Enable Intent Graph: YES  ← routes all speech here, not just "I have a question"

Port: configurable via WIREPOD_BRIDGE_PORT env var (default 8181)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Optional

from aiohttp import web

if TYPE_CHECKING:
    pass  # avoid circular import

logger = logging.getLogger(__name__)

# ── module-level agent reference ────────────────────────────────────────────
_agent = None  # set by start_wirepod_bridge()


def _set_agent(agent):
    global _agent
    _agent = agent


# ── helpers ─────────────────────────────────────────────────────────────────

def _sse_chunk(content: str, completion_id: str) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vector-personality",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_done(completion_id: str) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vector-personality",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


# ── request handler ──────────────────────────────────────────────────────────

async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract the last user message (that's the speech transcript from wire-pod)
    user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_text = msg.get("content", "").strip()
            break

    if not user_text:
        raise web.HTTPBadRequest(text="No user message found")

    logger.info(f"[WirePodBridge] 🗣️ Received: '{user_text}'")

    # Generate response via agent (agent speaks via our Italian gTTS internally)
    response_text = "Non ho capito, puoi ripetere?"  # fallback
    try:
        if _agent is not None:
            response_text = await _agent.handle_speech_for_wirepod(user_text)
        else:
            logger.warning("[WirePodBridge] Agent not available — returning fallback")
    except Exception as exc:
        logger.error(f"[WirePodBridge] Agent error: {exc}", exc_info=True)
        response_text = "Ho avuto un problema. Riprova."

    # Return " " (space) to wire-pod so it does NOT double-speak via
    # Vector's English built-in TTS.  Our agent already spoke the
    # response in Italian via gTTS above.
    wire_pod_reply = " "

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if stream:
        # Wire-pod expects Server-Sent Events streaming
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(_sse_chunk(wire_pod_reply, completion_id))
        await resp.write(_sse_done(completion_id))
        await resp.write_eof()
        return resp
    else:
        # Non-streaming fallback
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "vector-personality",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": wire_pod_reply},
                    "finish_reason": "stop",
                }
            ],
        }
        return web.json_response(payload)


async def handle_models(request: web.Request) -> web.Response:
    """Return a dummy model list so wire-pod doesn't complain."""
    return web.json_response(
        {
            "object": "list",
            "data": [
                {"id": "vector-personality", "object": "model", "owned_by": "local"}
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "agent": _agent is not None})


async def handle_catchall(request: web.Request) -> web.Response:
    """Log any request that doesn't match a known route — helps debug wire-pod path."""
    body_bytes = await request.read()
    try:
        body_text = body_bytes.decode("utf-8")[:400]
    except Exception:
        body_text = repr(body_bytes[:100])
    logger.warning(
        f"[WirePodBridge] ⚠️ Unmatched request: {request.method} {request.path} "
        f"| body={body_text}"
    )
    # If it looks like a chat completions call, handle it regardless of path
    if b"messages" in body_bytes:
        logger.info("[WirePodBridge] 🔄 Looks like a chat request — handling anyway")
        try:
            body_json = json.loads(body_bytes)
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        messages = body_json.get("messages", [])
        stream = body_json.get("stream", False)
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "").strip()
                break
        if not user_text:
            return web.json_response({"error": "no user message"}, status=400)

        logger.info(f"[WirePodBridge] 🗣️ Received (fallback path): '{user_text}'")
        response_text = "Non ho capito, puoi ripetere?"
        try:
            if _agent is not None:
                response_text = await _agent.handle_speech_for_wirepod(user_text)
        except Exception as exc:
            logger.error(f"[WirePodBridge] Agent error: {exc}", exc_info=True)

        # Return " " to wire-pod — our agent already spoke via gTTS
        wire_pod_reply = " "
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if stream:
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            await resp.prepare(request)
            await resp.write(_sse_chunk(wire_pod_reply, completion_id))
            await resp.write(_sse_done(completion_id))
            await resp.write_eof()
            return resp
        else:
            payload = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "vector-personality",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": wire_pod_reply}, "finish_reason": "stop"}],
            }
            return web.json_response(payload)

    return web.json_response({"error": "not found", "path": request.path}, status=404)


# ── server lifecycle ─────────────────────────────────────────────────────────

_runner: Optional[web.AppRunner] = None


async def start_wirepod_bridge(agent, port: Optional[int] = None) -> int:
    """
    Start the OpenAI-compatible bridge server.

    :param agent: The VectorAgent instance.
    :param port:  Port to listen on (default: WIREPOD_BRIDGE_PORT env var or 8181).
    :returns:     The port the server is listening on.
    """
    global _runner

    _set_agent(agent)

    if port is None:
        port = int(os.environ.get("WIREPOD_BRIDGE_PORT", "8181"))

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    # Catch-all: log anything that doesn't match (helps debug wire-pod path)
    app.router.add_route("*", "/{path_info:.*}", handle_catchall)

    _runner = web.AppRunner(app)
    await _runner.setup()
    site = web.TCPSite(_runner, "0.0.0.0", port)
    await site.start()

    logger.info(
        f"[WirePodBridge] ✅ Listening on port {port}. "
        f"Configure wire-pod Knowledge Graph → Custom → http://<THIS-PC-IP>:{port}"
    )
    return port


async def stop_wirepod_bridge():
    global _runner
    if _runner:
        await _runner.cleanup()
        _runner = None
        logger.info("[WirePodBridge] Server stopped")
