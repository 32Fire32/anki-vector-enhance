"""
Vector Agent GUI Dashboard
===========================
Web-based control panel for the Vector Personality Agent.

Usage:
    python execute.py

Opens a browser to http://localhost:8765 with tabs:
  - Dashboard: Robot info, mood, START/STOP controls
  - Monitor:   Live logs, VLM context, camera frames
  - Settings:  Model selection, volume, scan interval
"""

import asyncio
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Ensure UTF-8 on Windows
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Load api.env
# ---------------------------------------------------------------------------
def _load_env():
    env_file = Path(__file__).parent / "api.env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ---------------------------------------------------------------------------
# Shared Bridge (thread-safe state between agent and GUI)
# ---------------------------------------------------------------------------
class AgentBridge:
    """Thread-safe state shared between the agent thread and FastAPI."""

    def __init__(self):
        self.lock = threading.Lock()
        # Agent lifecycle
        self.running = False
        self.status = "stopped"          # stopped | starting | running | error
        self.error_message = ""
        # Robot info
        self.robot_name = ""
        self.robot_serial = ""
        self.robot_ip = ""
        # Models
        self.llm_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.vlm_model = os.getenv("VLM_MODEL", "llava:7b")
        # Mood (0-100)
        self.mood = 50
        # VLM frames
        self.vlm_frame_b64: Optional[str] = None
        self.vlm_response: Optional[str] = None
        self.vlm_objects: list = []
        self.vlm_timestamp: Optional[str] = None
        # Live camera
        self.camera_frame_b64: Optional[str] = None
        self.camera_enabled = False
        # LLM context
        self.last_llm_context = ""
        self.last_llm_response = ""
        # Logs
        self.log_queue: queue.Queue = queue.Queue(maxsize=2000)
        # Agent & thread references
        self._agent = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks_to_cancel = []  # Async tasks to cancel on shutdown
        # Unified chat handler (shared by web chat + Telegram)
        self._chat_handler = None
        self._telegram_bot = None
        # Standalone chat — always available even without robot
        self._standalone_chat = None

    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot of current state."""
        with self.lock:
            return {
                "status": self.status,
                "error": self.error_message,
                "robot_name": self.robot_name,
                "robot_serial": self.robot_serial,
                "robot_ip": self.robot_ip,
                "llm_model": self.llm_model,
                "vlm_model": self.vlm_model,
                "mood": self.mood,
                "vlm_frame": self.vlm_frame_b64,
                "vlm_response": self.vlm_response,
                "vlm_objects": self.vlm_objects,
                "vlm_timestamp": self.vlm_timestamp,
                "camera_frame": self.camera_frame_b64 if self.camera_enabled else None,
                "camera_enabled": self.camera_enabled,
                "last_llm_context": self.last_llm_context,
                "last_llm_response": self.last_llm_response,
            }


bridge = AgentBridge()

# Initialize standalone chat immediately (no robot needed — just Ollama + DB)
def _init_standalone_chat():
    try:
        from vector_personality.api.standalone_chat import StandaloneChatHandler
        sc = StandaloneChatHandler()
        sc.initialize()          # Lazy-safe: will retry on first message if Ollama is not yet up
        bridge._standalone_chat = sc
    except Exception as e:
        logging.getLogger("execute").warning(f"StandaloneChat init skipped: {e}")

_init_standalone_chat()

# ---------------------------------------------------------------------------
# Logging handler that pushes records into the bridge's queue
# ---------------------------------------------------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put_nowait(msg)
        except queue.Full:
            pass  # Drop oldest if full

# ---------------------------------------------------------------------------
# Agent thread
# ---------------------------------------------------------------------------
def _run_agent_thread():
    """Run the VectorAgent in its own thread + event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bridge._loop = loop
    try:
        loop.run_until_complete(_run_agent_async())
    except Exception as e:
        # Log the error so it appears in the Monitor tab
        logging.getLogger("execute").error(f"Agent thread fatal error: {e}", exc_info=True)
        with bridge.lock:
            bridge.status = "error"
            bridge.error_message = str(e)
    finally:
        loop.close()
        bridge._loop = None
        with bridge.lock:
            if bridge.status != "error":
                bridge.status = "stopped"
            bridge.running = False


async def _run_agent_async():
    """Async agent lifecycle."""
    import anki_vector
    from vector_personality.core.vector_agent import VectorAgent

    # Configure logging with queue handler for GUI
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Queue handler (sends to GUI)
    qh = QueueLogHandler(bridge.log_queue)
    qh.setFormatter(formatter)
    root_logger.addHandler(qh)

    # File handler
    fh = logging.FileHandler("vector_agent.log", encoding="utf-8")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # Suppress noisy SDK loggers
    for name in ("events.EventHandler", "anki_vector.events", "anki_vector.connection"):
        logging.getLogger(name).setLevel(logging.ERROR)

    logger = logging.getLogger("execute")

    with bridge.lock:
        bridge.status = "starting"
        bridge.error_message = ""

    logger.info("Connecting to Vector...")
    vector_ip = os.environ.get("VECTOR_HOST") or os.environ.get("VECTOR_IP")
    robot_kwargs = {"enable_face_detection": True, "enable_audio_feed": True}
    if vector_ip:
        robot_kwargs["ip"] = vector_ip

    robot = anki_vector.Robot(**robot_kwargs)
    try:
        robot.connect(timeout=60)
    except anki_vector.exceptions.VectorTimeoutException as e:
        if "ListAnimations" not in str(e) and "ListAnimationTriggers" not in str(e):
            raise

    # Extract robot info
    with bridge.lock:
        bridge.robot_name = getattr(robot, "_name", "") or ""
        bridge.robot_ip = getattr(robot, "_ip", "") or ""
        # Serial comes from config section name
        try:
            cfg_path = Path.home() / ".anki_vector" / "sdk_config.ini"
            if cfg_path.exists():
                import configparser
                cp = configparser.ConfigParser(strict=False)
                cp.read(str(cfg_path))
                for section in cp.sections():
                    if cp.get(section, "name", fallback="") == bridge.robot_name:
                        bridge.robot_serial = section
                        break
        except Exception:
            pass

    logger.info(f"Connected to {bridge.robot_name} at {bridge.robot_ip}")

    try:
        agent = VectorAgent(robot)
        bridge._agent = agent
        with bridge.lock:
            bridge.status = "running"
            bridge.running = True

        # Install hooks BEFORE starting the agent loop
        _install_hooks(agent)

        # Unified chat handler (web + Telegram share the same brain)
        from vector_personality.api.chat_handler import ChatHandler
        bridge._chat_handler = ChatHandler(agent)

        await agent.start()
    finally:
        bridge._chat_handler = None
        bridge._agent = None
        try:
            robot.disconnect()
        except Exception:
            pass


def _install_hooks(agent):
    """Monkey-patch lightweight hooks into the agent for GUI data extraction."""
    _hook_log = logging.getLogger("execute.hooks")

    # 1. Hook into scene_descriptor to capture VLM frames + responses
    sd = getattr(agent, "scene_descriptor", None)
    if sd:
        original_extract = sd._extract_objects

        async def _hooked_extract(image):
            # Capture the frame as base64 for the GUI using a thread executor
            # so PIL enhancement doesn't block the event loop.
            try:
                from vector_personality.cognition.ollama_client import OllamaClient
                loop = asyncio.get_running_loop()
                b64 = await loop.run_in_executor(None, OllamaClient._pil_to_base64, image)
                with bridge.lock:
                    bridge.vlm_frame_b64 = b64
                    bridge.vlm_timestamp = datetime.now().strftime("%H:%M:%S")
                _hook_log.info(f"VLM frame captured ({len(b64)} chars)")
            except Exception as e:
                _hook_log.warning(f"VLM frame capture failed: {e}")
            result = await original_extract(image)
            with bridge.lock:
                bridge.vlm_objects = list(result) if result else []
            return result

        sd._extract_objects = _hooked_extract

        # Capture the raw VLM response by hooking vision_completion
        ollama = getattr(sd, "ollama", None)
        if ollama:
            original_vision = ollama.vision_completion

            async def _hooked_vision(*args, **kwargs):
                result = await original_vision(*args, **kwargs)
                with bridge.lock:
                    bridge.vlm_response = result
                return result

            ollama.vision_completion = _hooked_vision

    # 2. Hook into context builder to capture LLM context
    cb = getattr(agent, "context_builder", None)
    if cb and hasattr(cb, "build_context"):
        original_build = cb.build_context

        async def _hooked_build(*args, **kwargs):
            result = await original_build(*args, **kwargs)
            if isinstance(result, str):
                with bridge.lock:
                    bridge.last_llm_context = result
            elif isinstance(result, list):
                # Messages list - format for display
                with bridge.lock:
                    bridge.last_llm_context = "\n".join(
                        f"[{m.get('role','?')}] {m.get('content','')[:500]}"
                        for m in result
                        if isinstance(m, dict)
                    )
            return result

        cb.build_context = _hooked_build

    # 3. Periodically read mood from working memory
    memory = getattr(agent, "memory", None)
    if memory:
        original_iter = agent._main_loop_iteration

        async def _hooked_iter():
            await original_iter()
            with bridge.lock:
                bridge.mood = getattr(memory, "current_mood", 50)

        agent._main_loop_iteration = _hooked_iter

    # 4. Dedicated fast camera capture loop (every 2s, in thread executor).
    #    Updates both the live camera stream and the VLM frame preview independently
    #    of how slow or fast llava inference is running.  This ensures the GUI always
    #    shows a recent frame even during long VLM pauses.
    async def _camera_loop():
        from vector_personality.cognition.ollama_client import OllamaClient
        loop = asyncio.get_running_loop()
        while getattr(agent, "running", False):
            try:
                img_raw = None
                try:
                    cam_img = agent.robot.camera.latest_image
                    if cam_img and cam_img.raw_image is not None:
                        img_raw = cam_img.raw_image
                except Exception:
                    pass

                if img_raw is not None:
                    # Live camera preview (only when user enabled it).
                    # VLM Frame is intentionally NOT updated here — it only changes
                    # when llava actually runs a scan (see _hooked_extract).
                    if bridge.camera_enabled:
                        b64_live = await loop.run_in_executor(
                            None, lambda i=img_raw: OllamaClient._pil_to_base64(i, max_size=320)
                        )
                        with bridge.lock:
                            bridge.camera_frame_b64 = b64_live
            except Exception as e:
                _hook_log.debug(f"Camera loop error: {e}")

            await asyncio.sleep(2.0)

    # Store task reference so it can be cancelled on shutdown
    camera_task = asyncio.ensure_future(_camera_loop())
    with bridge.lock:
        bridge._tasks_to_cancel.append(camera_task)


def start_agent():
    """Start the agent in a background thread."""
    if bridge._thread and bridge._thread.is_alive():
        return False
    bridge._thread = threading.Thread(target=_run_agent_thread, daemon=True, name="VectorAgent")
    bridge._thread.start()
    return True


def stop_agent():
    """Signal the agent to stop."""
    agent = bridge._agent
    if agent:
        # 1. Signal agent to stop
        agent.running = False
        if agent.shutdown_event:
            agent.shutdown_event.set()
        
        # 2. Stop scene descriptor loop
        sd = getattr(agent, "scene_descriptor", None)
        if sd:
            sd.stop()
        
        # 3. Cancel pending async tasks (e.g., _camera_loop)
        with bridge.lock:
            tasks = bridge._tasks_to_cancel[:]
            bridge._tasks_to_cancel.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # 4. Update status
        with bridge.lock:
            bridge.status = "stopped"
            bridge.running = False
        
        # 5. Wait for agent thread to finish (up to 5 seconds)
        if bridge._thread:
            bridge._thread.join(timeout=5.0)
        
        return True
    return False


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------
def read_settings() -> dict:
    """Read current settings from api.env."""
    settings = {}
    env_file = Path(__file__).parent / "api.env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    settings[key.strip()] = value.strip()
    return settings


def write_settings(updates: dict):
    """Update specific keys in api.env without losing comments."""
    env_file = Path(__file__).parent / "api.env"
    if not env_file.exists():
        return

    lines = env_file.read_text(encoding="utf-8").splitlines()
    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Add any new keys not already in the file
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def get_available_models() -> list:
    """Query Ollama for available models."""
    try:
        import requests
        resp = requests.get(
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags",
            timeout=5,
        )
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
_log = logging.getLogger("execute")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start Telegram at uvicorn boot; stop it on shutdown."""
    try:
        from vector_personality.api.telegram_bot import start_telegram_bot
        bridge._telegram_bot = await start_telegram_bot(bridge)
        if bridge._telegram_bot:
            _log.info("Telegram bot started at app startup")
    except Exception as e:
        _log.warning(f"Telegram bot not started: {e}")

    yield  # application runs

    if bridge._telegram_bot:
        try:
            await bridge._telegram_bot.stop()
        except Exception:
            pass
        bridge._telegram_bot = None


app = FastAPI(title="Vector Dashboard", lifespan=_lifespan)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/state")
async def get_state():
    return bridge.snapshot()


@app.get("/api/settings")
async def get_settings():
    return read_settings()


@app.post("/api/settings")
async def update_settings(body: dict):
    allowed_keys = {
        "OLLAMA_MODEL", "VLM_MODEL", "VLM_INTERVAL", "WHISPER_MODEL",
        "AUDIO_SOURCE", "MICROPHONE_DEVICE_ID", "AMBIENT_MODE",
        "STARTUP_MEMORY_HOURS", "VECTOR_HOST",
    }
    filtered = {k: v for k, v in body.items() if k in allowed_keys}
    if not filtered:
        return {"ok": False, "error": "No valid settings provided"}
    write_settings(filtered)
    # Also update bridge display values
    with bridge.lock:
        if "OLLAMA_MODEL" in filtered:
            bridge.llm_model = filtered["OLLAMA_MODEL"]
        if "VLM_MODEL" in filtered:
            bridge.vlm_model = filtered["VLM_MODEL"]
    return {"ok": True, "updated": list(filtered.keys())}


@app.get("/api/models")
async def get_models():
    return {"models": get_available_models()}


@app.post("/api/start")
async def api_start():
    if bridge.running:
        return {"ok": False, "error": "Agent is already running"}
    ok = start_agent()
    return {"ok": ok}


@app.post("/api/stop")
async def api_stop():
    ok = stop_agent()
    return {"ok": ok}


@app.post("/api/camera/toggle")
async def toggle_camera():
    with bridge.lock:
        bridge.camera_enabled = not bridge.camera_enabled
        return {"camera_enabled": bridge.camera_enabled}


# ---------------------------------------------------------------------------
# Chat API (web chat + shared with Telegram via ChatHandler)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def api_chat(body: dict):
    """Handle a text chat message from the web dashboard.

    Works in two modes:
    - Agent running: uses ChatHandler (full context, TTS, visual memory)
    - Agent stopped: uses StandaloneChatHandler (Ollama + ChromaDB, no robot)
    """
    text = (body.get("message") or "").strip()
    if not text:
        return {"ok": False, "error": "Empty message"}

    agent_handler = bridge._chat_handler
    loop = bridge._loop

    if agent_handler and loop and loop.is_running():
        # Agent is running — route through its event loop for full context
        future = asyncio.run_coroutine_threadsafe(
            agent_handler.handle_message(text, channel="web", user_name="Web User"),
            loop,
        )
        try:
            response = future.result(timeout=60)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "response": response}

    # Agent not running — use standalone (robot-free) handler
    sc = bridge._standalone_chat
    if not sc:
        return {
            "ok": False,
            "error": "Nessun sistema AI disponibile. Assicurati che Ollama sia in esecuzione.",
        }
    try:
        response = await sc.handle_message(text, channel="web", user_name="Web User")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "response": response}


@app.get("/api/camera/snapshot")
async def camera_snapshot():
    """Return a single JPEG snapshot from Vector's camera."""
    handler = bridge._chat_handler
    if not handler:
        return {"ok": False, "error": "Agent not running"}
    b64 = handler.get_camera_snapshot_b64()
    if b64 is None:
        return {"ok": False, "error": "Vector is offline or camera unavailable"}
    return {"ok": True, "image": b64}


# ---------------------------------------------------------------------------
# WebSocket for real-time updates
# ---------------------------------------------------------------------------
connected_ws: list = []


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_ws.append(ws)
    try:
        while True:
            # Build update payload
            payload = bridge.snapshot()

            # Drain log queue
            logs = []
            while not bridge.log_queue.empty():
                try:
                    logs.append(bridge.log_queue.get_nowait())
                except queue.Empty:
                    break
            payload["logs"] = logs

            await ws.send_json(payload)
            await asyncio.sleep(1.0)  # 1 Hz updates
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in connected_ws:
            connected_ws.remove(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    port = int(os.getenv("DASHBOARD_PORT", "8765"))
    print(f"Vector Dashboard starting on http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
