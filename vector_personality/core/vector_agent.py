"""
Vector Agent Orchestrator - Main event loop and system integration.

Integrates all 6 modules (memory, perception, emotion, cognition, behavior, core)
into a unified autonomous agent with CLI monitoring and graceful shutdown.

Phase 5: Autonomous Behavior System
Author: Vector Personality Enhancement Team
Date: 2025-12-05
"""

import asyncio
import logging
import signal
import sys
import os
import uuid
import random
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector.exceptions import VectorPropertyValueNotReadyException

# Suppress unknown event warnings
logging.getLogger("events.EventHandler").setLevel(logging.ERROR)
logging.getLogger("anki_vector.events").setLevel(logging.ERROR)
logging.getLogger("anki_vector.connection").setLevel(logging.ERROR)

# Import all modules
from vector_personality.core.personality import PersonalityModule
from vector_personality.core.config import DEFAULT_CHROMADB_DIR
from vector_personality.memory.chromadb_connector import ChromaDBConnector, initialize_database
from vector_personality.memory.working_memory import WorkingMemory, TaskState

# Load environment variables from api.env
def load_env_file():
    """Load environment variables from api.env file."""
    env_file = Path(__file__).parent.parent.parent / 'api.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove inline comments (anything after #)
                    if '#' in value:
                        value = value.split('#')[0]
                    # Remove quotes and whitespace
                    value = value.strip('"').strip("'").strip()
                    os.environ[key] = value

load_env_file()

# Conditional imports for modules that may not exist yet
# Import each module separately to avoid cascading failures
try:
    from vector_personality.perception.face_detection import FaceDetectionHandler
except ImportError:
    FaceDetectionHandler = None

try:
    from vector_personality.perception.audio_processor import AudioProcessor
except ImportError:
    AudioProcessor = None

try:
    from vector_personality.perception.vector_mic import VectorMicProcessor
except ImportError:
    VectorMicProcessor = None

try:
    from vector_personality.perception.speech_recognition import SpeechRecognizer
except ImportError:
    SpeechRecognizer = None

try:
    from vector_personality.perception.local_whisper import LocalWhisperRecognizer
except ImportError:
    LocalWhisperRecognizer = None

try:
    from vector_personality.perception.object_detector import ObjectDetector
except ImportError:
    ObjectDetector = None

try:
    from vector_personality.perception.room_inference import RoomInference
except ImportError:
    RoomInference = None

try:
    from vector_personality.perception.text_to_speech import TextToSpeech
except ImportError:
    TextToSpeech = None

try:
    from vector_personality.emotion.mood_engine import MoodEngine
    from vector_personality.emotion.eye_color_mapper import EyeColorMapper
except ImportError:
    MoodEngine = None
    EyeColorMapper = None

try:
    from vector_personality.cognition.ollama_client import OllamaClient
except ImportError:
    OllamaClient = None

try:
    from vector_personality.cognition.groq_client import GroqClient
except ImportError:
    GroqClient = None

try:
    from vector_personality.cognition.openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from vector_personality.cognition.reasoning_engine import ReasoningEngine
except ImportError:
    ReasoningEngine = None

try:
    from vector_personality.cognition.context_builder import ContextBuilder
except ImportError:
    ContextBuilder = None

try:
    from vector_personality.behavior.task_manager import TaskManager, TaskPriority
    from vector_personality.behavior.autonomy_controller import AutonomyController
    from vector_personality.behavior.curiosity_engine import CuriosityEngine
    from vector_personality.behavior.idle_controller import IdleController
    from vector_personality.behavior.startup_controller import StartupController
    from vector_personality.behavior.animation_mapper import animation_mapper
except ImportError:
    TaskManager = None
    TaskPriority = None
    AutonomyController = None
    CuriosityEngine = None
    IdleController = None
    StartupController = None
    animation_mapper = None


logger = logging.getLogger(__name__)


class VectorAgent:
    """
    Main orchestrator for Vector's autonomous behavior system.
    
    Integrates all 6 modules:
    - Memory (persistent + working)
    - Perception (faces, objects, audio, room)
    - Emotion (mood tracking, eye color)
    - Cognition (GPT-4, budget, reasoning)
    - Behavior (tasks, autonomy, curiosity)
    - Core (personality, config, orchestration)
    
    Responsibilities:
    - Main event loop coordination
    - SDK event handling (face detection, cube events, etc.)
    - Task execution cycle
    - Graceful shutdown
    - CLI status monitoring
    """
    
    def __init__(self, robot: anki_vector.Robot):
        """Initialize the agent with all modules."""
        self.robot = robot
        self.running = False
        self.shutdown_event = None  # Will be created in start() when async loop is ready
        
        # Initialize modules
        logger.info("Initializing Vector Agent modules...")
        
        # Console input management (non-blocking)
        self.pending_console_input = None  # Stores (question, timestamp) tuple
        self.console_answer = None  # Stores user's typed answer
        self.console_input_task = None  # Background task for input
        
        # Conversation context tracking
        self.conversation_active = False  # True after wake word detected
        self.last_interaction_time = None  # Timestamp of last user speech
        self.conversation_timeout = 30.0  # Seconds before requiring wake word again (balanced: enough for follow-ups, not too long for background)
        self.conversation_history = []  # Short-term buffer: [{"role": "user"|"assistant", "content": ...}]
        self.unknown_face_id = None  # Reusable face_id for unknown users (avoids spam)
        self._speech_processing = False  # Guard: prevent concurrent _handle_user_speech calls
        
        # Startup tracking
        self.startup_time = None  # Set when startup completes
        self.startup_grace_period = 180.0  # 3 minutes - give Vector time to settle before curiosity kicks in
        self.startup_greeting_done = False  # Ensure we greet only after microphone is active
        
        # Core modules
        # Load database configuration from environment variables
        chromadb_dir = os.environ.get('CHROMADB_DIR', DEFAULT_CHROMADB_DIR)
        logger.info(f"Database: ChromaDB at {chromadb_dir}")
        self.db = ChromaDBConnector(persist_directory=chromadb_dir)
        self.personality = PersonalityModule(self.db)
        self.memory = WorkingMemory()
        
        # Get API keys for perception modules
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        # Perception modules (Phase 2)
        # Note: FaceDetectionHandler initialized later after TTS is available
        
        # Audio processing
        self.audio_processor = AudioProcessor() if AudioProcessor else None
        self.vector_mic = None  # Initialized in _process_audio once robot is connected
        
        # Speech recognition — Groq cloud first, then local Whisper fallback
        self.speech_recognizer = None
        if SpeechRecognizer and groq_api_key:
            self.speech_recognizer = SpeechRecognizer(
                api_key=groq_api_key,
                language="it"
            )
            logger.info("✅ Speech recognition initialized (Groq Whisper, Italian)")
        if not self.speech_recognizer:
            if LocalWhisperRecognizer:
                whisper_model = os.environ.get("WHISPER_MODEL", "small")
                self.speech_recognizer = LocalWhisperRecognizer(
                    model_size=whisper_model,
                    language="it",
                )
                logger.info(f"🎤 Local Whisper STT active (model={whisper_model}, Italian)")
            else:
                logger.warning("⚠️ No speech recognizer available — install faster-whisper or set GROQ_API_KEY")
        
        # Object detection (YOLOv5)
        # Note: Skip object detection initialization for now - it's causing startup delays
        # Will be lazy-loaded on first camera frame
        logger.info(f"ObjectDetector class available: {ObjectDetector is not None}")
        self.object_detector = None
        self._object_detector_class = ObjectDetector  # Save class for lazy init
        self._yolo_model_path = None
        self._last_object_persistence = {}  # Track last DB save time per object type

        
        # Check for YOLO model file availability
        if ObjectDetector:
            model_path = Path("yolov5n.pt")
            if not model_path.exists():
                model_path = Path("yolov5s.pt")
            if model_path.exists():
                self._yolo_model_path = str(model_path)
                logger.info(f"📷 Object detection will use: {model_path} (lazy-loaded on first frame)")
            else:
                logger.warning(f"⚠️ No YOLO model found - object detection disabled")
        else:
            logger.warning("⚠️ ObjectDetector class not imported")
        
        # Room inference
        self.room_inference = RoomInference(self.db) if RoomInference else None
        
        # Text-to-Speech (Google TTS - FREE) - 80s Robot Voice Mode! 🤖
        if TextToSpeech:
            self.tts = TextToSpeech(
                robot=robot,
                voice="italian",
                speed=1.15,  # Faster robotic speech
                monotone=True,  # Enable robotic filter for authentic 80s robot voice
                audio_processor=self.audio_processor  # Pass audio_processor for feedback prevention
            )
            logger.info("✅ Text-to-speech initialized (Google TTS - FREE) - 80s Robot Mode 🤖")
        else:
            self.tts = None
            logger.warning("⚠️ Text-to-speech disabled (module not available)")
        
        # Face detection (initialized after TTS for greeting capability)
        self.face_detector = FaceDetectionHandler(robot, self.db, self.memory, self.tts) if FaceDetectionHandler else None
        
        # Emotion modules (Phase 3)
        self.mood_engine = MoodEngine() if MoodEngine else None
        self.eye_color_mapper = EyeColorMapper() if EyeColorMapper else None
        
        # Cognition modules (Phase 11: Local AI via Ollama)
        # Priority: Ollama (local, free, private) → Groq → OpenAI
        ollama_model = os.environ.get('OLLAMA_MODEL', 'mistral-small3.2:latest')
        ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        self.llm_client = None
        self.chat_client = None  # Alias used by context_builder / summarizer
        self.groq_client = None  # Kept for backward compat with any remaining references
        self.openai_client = None

        # 1. Try local Ollama first (free, private, fast on local GPU)
        if OllamaClient:
            try:
                ollama = OllamaClient(
                    base_url=ollama_url,
                    default_model=ollama_model,
                    timeout_seconds=60,
                )
                # Quick synchronous connectivity check (no event loop needed)
                import requests as _req
                try:
                    r = _req.get(f"{ollama_url}/api/tags", timeout=5)
                    models = [m["name"] for m in r.json().get("models", [])]
                    available = ollama_model in models
                except Exception:
                    available = False

                if available:
                    self.llm_client = ollama
                    self.chat_client = ollama
                    self.groq_client = ollama  # compat alias for context_builder
                    logger.info(f"🏠 Using LOCAL Ollama LLM: {ollama_model}")
                    logger.info("✅ Zero cloud cost, full privacy, all data stays on this machine")
                else:
                    logger.warning(f"⚠️ Ollama running but model '{ollama_model}' not available")
            except Exception as e:
                logger.warning(f"⚠️ Ollama not reachable: {e}")

        # 2. Fall back to Groq (cloud, free tier)
        if not self.llm_client and GroqClient and groq_api_key:
            logger.info("☁️ Falling back to Groq cloud API (Ollama unavailable)")
            self.llm_client = GroqClient(
                api_key=groq_api_key,
                default_model="llama-3.3-70b-versatile"
            )
            self.chat_client = self.llm_client
            self.groq_client = self.llm_client
            logger.info("✅ Groq client initialized (Llama 3.3 70B)")

        # 3. Fall back to OpenAI (cloud, paid)
        if not self.llm_client and OpenAIClient and openai_api_key:
            logger.info("☁️ Falling back to OpenAI cloud API (Ollama/Groq unavailable)")
            self.llm_client = OpenAIClient(api_key=openai_api_key)
            self.chat_client = self.llm_client
            self.groq_client = None
            self.openai_client = self.llm_client
            logger.info("✅ OpenAI client initialized (GPT-4)")

        if not self.llm_client:
            logger.warning("⚠️ No LLM available (Ollama not running, no cloud API keys) - reasoning features disabled")
        
        # Budget enforcement system removed (T122)
        self.budget_enforcer = None

        if self.llm_client and ReasoningEngine:
            self.reasoning_engine = ReasoningEngine(
                working_memory=self.memory,
                db_connector=self.db
            )
        else:
            self.reasoning_engine = None

        # T140: Vector database + embedding generator for semantic search
        try:
            from vector_personality.memory.embedding_generator import EmbeddingGenerator
            from vector_personality.memory.vector_db_connector import VectorDBConnector
            
            # Initialize embedding generator (Ollama primary, OpenAI fallback)
            self.embedding_gen = EmbeddingGenerator()
            
            # Initialize vector database
            chromadb_dir = Path(__file__).parent.parent / "memory" / "chromadb_data"
            chromadb_dir.mkdir(parents=True, exist_ok=True)
            self.vector_db = VectorDBConnector(persist_directory=str(chromadb_dir))
            
            logger.info(f"✅ T140: Semantic search enabled ({self.embedding_gen.get_provider_info()['provider']})")
        except Exception as e:
            logger.warning(f"⚠️ T140: Semantic search disabled - {e}")
            self.embedding_gen = None
            self.vector_db = None

        # Context builder (T123) - now with startup summarization + semantic search (T140)
        if ContextBuilder and self.db and self.memory:
            self.context_builder = ContextBuilder(
                db_connector=self.db,
                working_memory=self.memory,
                groq_client=self.groq_client,  # NEW: Pass Groq for summarization
                vector_db=self.vector_db,      # T140: Vector database for semantic search
                embedding_gen=self.embedding_gen  # T140: Embedding generator
            )
        else:
            self.context_builder = None
        
        # Behavior modules (Phase 5)
        if TaskManager and AutonomyController and CuriosityEngine:
            # Import StateMachine for TaskManager
            try:
                from vector_personality.emotion.state_machine import StateMachine
                state_machine = StateMachine()
            except ImportError:
                state_machine = None
            
            if state_machine:
                self.task_manager = TaskManager(
                    working_memory=self.memory,
                    personality_module=self.personality,
                    state_machine=state_machine,
                    db_connector=self.db
                )
                self.autonomy_controller = AutonomyController(
                    working_memory=self.memory,
                    personality_module=self.personality,
                    reasoning_engine=self.reasoning_engine,
                    task_manager=self.task_manager,
                    robot=self.robot
                )
                self.curiosity_engine = CuriosityEngine(
                    working_memory=self.memory,
                    personality_module=self.personality,
                    reasoning_engine=self.reasoning_engine,
                    openai_client=self.openai_client,
                    db_connector=self.db,
                    min_interval_seconds=300  # 5 minutes between questions (user requested longer)
                )
            else:
                self.task_manager = None
                self.autonomy_controller = None
                self.curiosity_engine = None
        else:
            self.task_manager = None
            self.autonomy_controller = None
            self.curiosity_engine = None
        
        # Idle controller (keeps Vector looking alive)
        if IdleController:
            self.idle_controller = IdleController(
                robot=robot,
                min_interval_seconds=5.0,
                max_interval_seconds=10.0
            )
            logger.info("✅ Idle controller initialized")
        else:
            self.idle_controller = None
        
        # Startup controller (face-first initialization)
        if StartupController:
            self.startup_controller = StartupController(
                robot=robot,
                db_connector=self.db,
                tts=self.tts,
                face_detection_handler=self.face_detector,
                face_scan_timeout=30.0,  # 30 seconds to find face (3 angles × 5s each + buffer)
                head_scan_angle_range=(-10.0, 44.0)  # Near max range: -10° to 44° for normal face heights
            )
            logger.info("✅ Startup controller initialized")
        else:
            self.startup_controller = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ Vector Agent initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals (Ctrl+C, etc.)."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self.shutdown_event:
            self.shutdown_event.set()
        self.running = False
    
    async def start(self):
        """
        Start the agent's main event loop.
        
        Main loop consists of:
        1. Update sensors (camera, face detection, etc.)
        2. Update emotion state (mood calculation, eye color)
        3. Check for autonomous behavior triggers
        4. Execute pending tasks
        5. Display CLI status
        """
        # Create shutdown event now that we're in async context
        self.shutdown_event = asyncio.Event()
        self.running = True
        logger.info("🤖 Vector Agent starting up...")
        
        # Initialize database tables
        logger.info("Initializing database...")
        # Note: initialize_database is a standalone function, not a method
        # We skip it here since tables should already exist from Phase 1-4 testing
        # await initialize_database(self.db)
        
        # NEW: Generate startup memory summary (once at startup) - configurable via api.env
        if self.context_builder and hasattr(self.context_builder, 'generate_startup_summary'):
            try:
                startup_hours_env = os.getenv("STARTUP_MEMORY_HOURS", "72")
                try:
                    startup_hours = int(startup_hours_env)
                except ValueError:
                    startup_hours = 72
                    logger.warning(f"Invalid STARTUP_MEMORY_HOURS='{startup_hours_env}', falling back to 72")

                logger.info(f"🧠 Building startup memory summary (last {startup_hours}h)...")
                await self.context_builder.generate_startup_summary(hours=startup_hours)
                logger.info("✅ Startup memory summary ready")
            except Exception as e:
                logger.error(f"❌ Failed to generate startup summary: {e}")
                logger.info("⚠️ Continuing with fallback memory system")
        
        # Personality is already initialized in __init__
        logger.info("Personality profile loaded")
        
        # Enable camera for object detection
        logger.info("Enabling camera...")
        self.robot.camera.init_camera_feed()
        logger.info("✅ Camera feed enabled")
        
        # Start face detection
        if self.face_detector:
            logger.info("Starting face detection...")
            await self.face_detector.start()
        
        # Register SDK event handlers
        self._register_event_handlers()
        
        # Pre-load animation list to avoid timeout during wake word response
        logger.info("Loading animation triggers...")
        try:
            # This will populate the animation list cache
            anim_triggers = self.robot.anim.anim_trigger_list
            logger.info(f"✅ Loaded {len(anim_triggers)} animation triggers")
            # Dump the exact list to tools/available_animation_triggers.json for debugging and mapping
            try:
                from pathlib import Path
                import json
                out = Path(__file__).resolve().parents[2] / 'tools' / 'available_animation_triggers.json'
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(list(anim_triggers), f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Wrote available animation triggers to {out}")
            except Exception as fe:
                logger.warning(f"⚠️ Failed to write animation triggers to file: {fe}")
        except Exception as e:
            logger.warning(f"⚠️ Could not pre-load animations: {e}")
        
        logger.info("✅ Startup complete - entering main loop")
        logger.info("🎤 Voice interaction ENABLED - say 'Ciao Vector' to talk")
        self._display_status()
        
        # Startup greeting deferred until microphone is active.
        # The greeting will be played once the audio input is confirmed active
        # to avoid announcing readiness before audio/microphone subsystems are ready.

        # T109: Execute face-first startup sequence
        if self.startup_controller:
            logger.info("🚀 Starting face-first initialization sequence...")
            startup_result = await self.startup_controller.execute_startup_sequence()
            
            if startup_result['success']:
                logger.info(f"✅ Startup sequence complete: {startup_result['message']}")
                if startup_result['face_recognized'] and startup_result['face_name']:
                    logger.info(f"👋 Recognized and greeted: {startup_result['face_name']}")
                elif startup_result['face_found']:
                    logger.info(f"👤 Face found but not recognized (enrolled as: {startup_result['face_name'] or 'Unknown'})")
            else:
                logger.info(f"⚠️ Startup sequence incomplete: {startup_result['message']}")
        else:
            logger.warning("⚠️ StartupController not available - skipping face-first sequence")
        
        # Mark startup complete - this enables curiosity questions after grace period
        self.startup_time = datetime.now()
        logger.info(f"✅ Startup complete - curiosity grace period: {self.startup_grace_period}s")
        
        # Disable startup mode in face detection to enable continuous greeting
        if self.face_detector:
            self.face_detector.disable_startup_mode()
        
        # Start Audio Loop as a background task
        self.audio_task = asyncio.create_task(self._audio_loop())

        # Start wire-pod OpenAI-compatible bridge server
        try:
            from vector_personality.api.wirepod_bridge import start_wirepod_bridge
            bridge_port = await start_wirepod_bridge(self)
            logger.info(f"[WirePodBridge] ✅ Bridge active on :{bridge_port} — configure wire-pod KG → custom → http://<this-PC>:{bridge_port}")
        except Exception as _bridge_err:
            logger.warning(f"[WirePodBridge] Could not start bridge ({_bridge_err}) — wire-pod integration disabled")
        
        # Main event loop
        try:
            while self.running and not self.shutdown_event.is_set():
                await self._main_loop_iteration()
                await asyncio.sleep(0.5)  # 500ms cycle time
        
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
        
        finally:
            await self.shutdown()

    async def _run_startup_greeting(self, message: str):
        """Play the startup greeting via TTS or built-in speech.

        This is run as a background task to avoid blocking the audio loop.
        """
        try:
            if self.tts:
                await self.tts.speak(message)
                logger.info("✅ Startup greeting complete")
            elif self.robot:
                # built-in speak
                self.robot.behavior.say_text(message)
                logger.info("✅ Startup greeting (built-in) complete")
        except Exception as e:
            logger.error(f"❌ Startup greeting failed: {e}", exc_info=True)
    
    async def _main_loop_iteration(self):
        """Execute one iteration of the main event loop."""
        try:
            # 1. Process perception inputs (vision)
            await self._process_perception()
            # Note: Audio is now processed in self._audio_loop() background task
            
            # 2. Update emotion state
            await self._update_emotion()
            
            # 3. Check for autonomous behavior triggers
            await self._check_autonomy_triggers()
            
            # 4. Generate curiosity questions if appropriate
            await self._check_curiosity()
            
            # 5. Execute pending tasks
            await self._execute_tasks()
            
            # 6. Periodic status updates (every 20 cycles = 10 seconds)
            if not hasattr(self, '_loop_counter'):
                self._loop_counter = 0
            self._loop_counter += 1
            
            if self._loop_counter % 20 == 0:
                logger.info(f"📊 Status: Faces={len(self.memory.current_faces)}, Objects={len(self.memory.current_objects)}, Room={self.memory.current_room or 'Unknown'}, Mood={self.memory.current_mood}")
        
        except Exception as e:
            logger.error(f"Error in main loop iteration: {e}", exc_info=True)
    
    async def _process_perception(self):
        """Process camera frames and audio for object/room detection."""
        try:
            # Lazy-load object detector on first camera frame
            if self.object_detector is None and self._object_detector_class and self._yolo_model_path:
                try:
                    logger.info(f"📷 Loading YOLOv5 model: {self._yolo_model_path}")
                    self.object_detector = self._object_detector_class(
                        model_path=self._yolo_model_path,
                        confidence_threshold=0.5,
                        device="cpu"
                    )
                    logger.info(f"✅ Object detection loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load object detector: {e}")
                    self._object_detector_class = None  # Don't try again
            
            # Process camera frame for object detection
            if self.object_detector and self.robot.camera:
                try:
                    # Get latest camera image
                    try:
                        latest_image = self.robot.camera.latest_image
                    except VectorPropertyValueNotReadyException:
                        # Camera not ready yet
                        return

                    if latest_image and latest_image.raw_image is not None:
                        # Detect objects
                        detections = self.object_detector.detect(latest_image.raw_image)
                        
                        # Log all detections immediately
                        if detections:
                            obj_list = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                            logger.info(f"👁️ Detected {len(detections)} objects: {obj_list}")
                        
                        # Update working memory with detections
                        for detection in detections:
                            self.memory.observe_object(
                                object_type=detection['class'],
                                confidence=detection['confidence'],
                                location_description="in view"
                            )
                            
                            # Persist to database (Throttled: max once per 5s per object type)
                            obj_type = detection['class']
                            now = datetime.now()
                            last_save = self._last_object_persistence.get(obj_type)
                            
                            if not last_save or (now - last_save).total_seconds() > 5:
                                try:
                                    object_id = await self.db.store_object_detection(
                                        object_type=obj_type,
                                        confidence=detection['confidence'],
                                        room_id=self.memory.current_room_id,
                                        location_description="in view"
                                    )
                                    logger.info(f"💿 Persisted to database: {obj_type} (ID: {object_id})")
                                    self._last_object_persistence[obj_type] = now
                                except Exception as e:
                                    logger.error(f"Failed to persist object to database: {e}")
                        
                        # Infer room type from objects
                        if self.room_inference and detections:
                            room_type = self.room_inference.infer_room_type(detections)
                            if room_type != "unknown":
                                current_room = self.memory.current_room
                                if current_room != room_type:
                                    self.memory.set_room(room_type)
                                    logger.info(f"📍 Room changed: {current_room} → {room_type}")
                                    # Store in database
                                    await self.room_inference.store_room_visit(room_type)
                    else:
                        # Camera image not available
                        if not hasattr(self, '_camera_warning_shown'):
                            logger.warning("⚠️ Camera image not available (latest_image is None)")
                            self._camera_warning_shown = True
                
                except Exception as e:
                    logger.error(f"❌ Camera processing error: {e}", exc_info=True)
            
        
        except Exception as e:
            logger.error(f"Error processing perception: {e}", exc_info=True)
    
    async def _audio_loop(self):
        """
        Continuous background task for audio processing.
        Runs independently of the main loop to ensure responsiveness during long tasks.
        """
        logger.info("🎧 Audio processing loop started")
        while self.running:
            try:
                await self._process_audio()
                await asyncio.sleep(0.05)  # Small sleep to prevent CPU hogging
            except Exception as e:
                logger.error(f"Error in audio loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Backoff on error

    async def _process_audio(self):
        """
        Process audio from Vector's built-in microphones for speech recognition.
        
        Falls back to PC microphone if Vector mic feed is unavailable.
        
        Flow:
        1. Stream audio from Vector's microphone array via gRPC AudioFeed
        2. Use VAD to detect speech segments
        3. Check for wake word ("Ciao Vector" or Italian variants) 
        4. Transcribe speech after wake word detected
        5. Save to database and process response
        """
        if not self.audio_processor:
            return

        try:
            # Initialize audio counter if not exists
            if not hasattr(self, '_audio_counter'):
                self._audio_counter = 0
            self._audio_counter += 1
            
            # Ensure microphone listening is started
            if not self.audio_processor.is_recording and not getattr(self, '_mic_start_failed', False):
                audio_source = os.getenv('AUDIO_SOURCE', 'vector').lower()
                logger.info(f"[Audio] Initialising audio source: {audio_source!r}")

                if audio_source == 'pc':
                    # Legacy: use PC microphone via sounddevice
                    device_id = int(os.getenv('MICROPHONE_DEVICE_ID', '1'))
                    logger.info(f"🎤 Starting PC microphone (device {device_id})...")
                    self.audio_processor.start_listening(device=device_id)
                    if self.audio_processor.is_recording:
                        logger.info("🎤 PC microphone active")
                    else:
                        logger.error("🎤 PC microphone FAILED to open — will not retry. Check MICROPHONE_DEVICE_ID.")
                        self._mic_start_failed = True
                else:
                    # Default: use Vector's built-in microphone array
                    if VectorMicProcessor and self.robot:
                        try:
                            logger.info("[Audio] Creating VectorMicProcessor...")
                            self.vector_mic = VectorMicProcessor(self.robot, self.audio_processor)
                            self.vector_mic.start()
                            # Mark audio_processor as recording so this block doesn't re-enter
                            self.audio_processor.is_recording = True
                            self._vector_mic_start_time = datetime.now()
                            logger.info("🎤 Vector microphone array start requested — waiting for first chunk...")
                            # Launch sliding-window STT loop (bypasses webrtcvad for noisy mic)
                            asyncio.create_task(self._vector_mic_stt_loop())
                            logger.info("[SlidingSTT] Background Whisper VAD task created")
                        except Exception as e:
                            logger.warning(f"⚠️ Vector mic init failed ({e}), falling back to PC microphone")
                            self._fallback_to_pc_mic()
                    else:
                        logger.warning("⚠️ VectorMicProcessor not available, using PC microphone")
                        self._fallback_to_pc_mic()

                # Play deferred startup greeting once microphone is active
                try:
                    if not getattr(self, 'startup_greeting_done', False):
                        self.startup_greeting_done = True
                        greetings = [
                            "Eccomi, sono pronto!",
                            "Ciao, mondo!",
                            "Che bella dormita!",
                            "Un attimo che mi stiracchio i circuiti!"
                        ]
                        message = random.choice(greetings)
                        asyncio.create_task(self._run_startup_greeting(message))
                except Exception:
                    logger.exception("Error scheduling startup greeting")

            # --- Vector mic health check: if no chunks after 10s, fall back to PC mic ---
            if (self.vector_mic and self.audio_processor.is_recording
                    and self.vector_mic.total_chunks_received == 0):
                elapsed = (datetime.now() - getattr(self, '_vector_mic_start_time', datetime.now())).total_seconds()
                if elapsed > 10.0:
                    logger.warning(
                        f"[Audio] ⚠️ Vector mic started {elapsed:.0f}s ago but ZERO chunks received. "
                        "Wire-pod likely does not support AudioFeed gRPC. "
                        "Falling back to PC microphone (set AUDIO_SOURCE=pc to suppress this)."
                    )
                    self.vector_mic.stop()
                    self.vector_mic = None
                    self.audio_processor.is_recording = False
                    self._fallback_to_pc_mic()

            # Check signal level periodically to help user debug microphone issues
            if self._audio_counter % 50 == 0:  # Every ~2.5s (assuming 20Hz loop)
                if self.vector_mic and self.vector_mic.is_active:
                    direction = self.vector_mic.direction_label()
                    logger.info(
                        f"🎤 Vector mic: chunks={self.vector_mic.total_chunks_received} "
                        f"dir={direction} energy={self.audio_processor.max_energy:.0f} "
                        f"vad_active={self.audio_processor.speech_detected} ✅"
                    )
                elif self.audio_processor.max_energy < 100:
                    logger.info(f"🎤 Signal check: energy={self.audio_processor.max_energy:.2f} "
                                f"vad={self.audio_processor.speech_detected} (low - speak louder)")
                else:
                    logger.info(f"🎤 Mic active: energy={self.audio_processor.max_energy:.2f} "
                                f"vad={self.audio_processor.speech_detected} ✅")
                self.audio_processor.max_energy = 0.0  # Reset for next window

            # Check for complete utterance (speech segment ended)
            # Process ALL queued utterances to prevent backlog
            while True:
                utterance_data = self.audio_processor.get_last_utterance()
                
                if not utterance_data:
                    break
                
                # Include direction info if using Vector mic
                direction_info = ""
                if self.vector_mic and self.vector_mic.is_active:
                    direction_info = f" from {self.vector_mic.direction_label()}"

                # Calculate audio duration (sample_rate is typically 16000 Hz, 2 bytes per sample)
                audio_duration = len(utterance_data) / (self.audio_processor.sample_rate * 2)
                logger.info(f"🎤 Speech detected ({audio_duration:.2f}s{direction_info})")

                if not self.speech_recognizer:
                    # Save audio to temp WAV so it can be inspected manually
                    import tempfile, wave as _wave
                    _debug_wav = Path(tempfile.gettempdir()) / f"vector_debug_{datetime.now().strftime('%H%M%S')}.wav"
                    with _wave.open(str(_debug_wav), 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.audio_processor.sample_rate)
                        wf.writeframes(utterance_data)
                    logger.info(
                        f"⚠️  No speech recognizer — mic IS triggering VAD! "
                        f"Saved audio to {_debug_wav} — open this WAV to verify audio quality. "
                        "Add GROQ_API_KEY to api.env or install faster-whisper for transcription."
                    )
                    continue

                # Skip if audio is too short — real utterances are at least 1s.
                # Fragments below this are VAD micro-triggers (silence transitions, noise bursts).
                if audio_duration < 1.0:
                    logger.debug(f"⚠️ Audio too short ({audio_duration:.2f}s), skipping transcription")
                    continue

                # Save to temp WAV file
                import tempfile
                import wave
                temp_wav = Path(tempfile.gettempdir()) / f"vector_speech_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
                with wave.open(str(temp_wav), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.audio_processor.sample_rate)
                    wf.writeframes(utterance_data)

                # Transcribe audio to text
                try:
                    logger.info("🔊 Transcribing audio...")
                    result = await self.speech_recognizer.transcribe(str(temp_wav))
                    
                    if result and result.get('text'):
                        text = result['text'].strip()
                        confidence = result.get('confidence', 1.0)
                        
                        # Skip empty transcriptions (background noise)
                        if len(text) == 0:
                            logger.debug("Empty transcription result (background noise)")
                            continue
                        
                        # Filter 1: Confidence threshold (prevent garbled transcriptions)
                        # Relaxed from 0.7 to 0.5 - Whisper is conservative with confidence scores
                        if confidence < 0.5:
                            logger.debug(f"⚠️ Low confidence ({confidence:.2f}), ignoring: \"{text}\"")
                            continue
                        
                        logger.info(f"📝 Transcribed (confidence={confidence:.2f}): \"{text}\"")
                        
                        # Check for wake word in transcribed text (Italian variants)
                        # NOTE: Vector's native intent system only triggers on "Hey Vector" (English)
                        # Our Italian wake words won't activate Vector's built-in commands
                        wake_words = ["ciao vector", "ehi vector", "salve vector", "hey vector", "vector"]
                        text_lower = text.lower()
                        has_wake_word = any(wake in text_lower for wake in wake_words)
                        
                        # Filter 2: Intelligent hallucination detection (T146)
                        # Uses multi-level analysis: lexical patterns, repetitions, confidence
                        # Pass conversation_active flag to filter single-word responses during active conversation
                        is_conversation_active = self.conversation_active or has_wake_word
                        is_hallucination, reason = self.speech_recognizer.is_likely_hallucination(
                            text, confidence, is_conversation_active
                        )
                        
                        if is_hallucination:
                            logger.info(f"🚫 Hallucination detected ({reason}), ignoring: \"{text}\"")
                            continue
                        
                        # Filter 3: Minimum word count (prevent single-word background noise)
                        # Skip filter if wake word is present (allow "Vector!" or "Ciao Vector")
                        # Reduced from 3 to 2 words - allows "no grazie", "va bene", etc.
                        word_count = len(text.split())
                        if not has_wake_word and word_count < 2:
                            logger.debug(f"⚠️ Too short ({word_count} words), ignoring: \"{text}\"")
                            continue
                        
                        # Check if conversation is still active (within timeout)
                        conversation_expired = False
                        time_since_last_interaction = None
                        if self.last_interaction_time:
                            time_since_last = datetime.now() - self.last_interaction_time
                            time_since_last_interaction = time_since_last.total_seconds()
                            conversation_expired = time_since_last_interaction > self.conversation_timeout
                        
                        # Debug logging for conversation state
                        time_str = f"{time_since_last_interaction:.1f}s" if time_since_last_interaction is not None else "N/A"
                        logger.info(f"🔍 Conversation check: active={self.conversation_active}, expired={conversation_expired}, has_wake={has_wake_word}, time_since_last={time_str} (timeout={self.conversation_timeout}s)")
                        
                        if has_wake_word:
                            # Wake word detected - activate conversation
                            self.conversation_active = True
                            self.last_interaction_time = datetime.now()
                            logger.info("💬 Conversation started - exploration behavior paused")
                            
                            # Play greeting animation to acknowledge wake word (non-blocking)
                            try:
                                if self.robot:
                                    # Start animation in background without blocking conversation flow
                                    asyncio.create_task(
                                        self.robot.anim.play_animation_trigger('PutDownBlockPutDown', ignore_body_track=True)
                                    )
                                    logger.info("👋 Wake word animation started")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not play wake word animation: {e}")
                            
                            # Cancel any ongoing exploration behavior to avoid control conflicts
                            if self.task_manager:
                                if self.task_manager.current_state.value == "exploring":
                                    await self.task_manager.manual_override("user_conversation")
                                    logger.debug("🛑 Exploration behavior cancelled due to conversation start")
                                
                            
                            # Remove wake word from text (preserve original capitalization)
                            import re
                            remaining_text = text
                            for wake in wake_words:
                                # Case-insensitive replacement using word boundaries
                                pattern = r'\b' + re.escape(wake) + r'\b'
                                remaining_text = re.sub(pattern, '', remaining_text, flags=re.IGNORECASE)
                            
                            # Clean up extra whitespace and punctuation
                            remaining_text = remaining_text.strip().strip(',').strip()
                            
                            if remaining_text:
                                # Process the command/question
                                logger.info(f"🎤 Heard: '{remaining_text}'")
                                await self._handle_user_speech(remaining_text)
                            else:
                                logger.info("👋 Wake word only - no command given")
                                # Just acknowledge (Italian)
                                if self.tts:
                                    if self.audio_processor:
                                        self.audio_processor.discard_pending_utterances()
                                    await self.tts.speak("Sì? Come posso aiutarti?")
                        elif self.conversation_active and not conversation_expired:
                            # No wake word, but conversation still active - process as follow-up
                            self.last_interaction_time = datetime.now()
                            logger.info(f"💬 Follow-up (no wake word needed): '{text}'")
                            await self._handle_user_speech(text)
                        elif conversation_expired and self.conversation_active and word_count >= 3:
                            # Conversation expired BUT user is clearly speaking to Vector (3+ words)
                            # Auto-renew conversation instead of requiring wake word
                            self.last_interaction_time = datetime.now()
                            logger.info(f"💬 Auto-renewing conversation (expired but user speaking): '{text}'")
                            await self._handle_user_speech(text)
                        else:
                            # No wake word and conversation expired/inactive
                            if conversation_expired:
                                self.conversation_active = False
                                logger.info("🔄 Conversation ended - resuming autonomous exploration")

                            # AMBIENT MODE: ask LLM if Vector should join this conversation
                            if os.getenv('AMBIENT_MODE', 'false').lower() == 'true' and word_count >= 2:
                                should_respond = await self._should_respond_ambient(text)
                                if should_respond:
                                    logger.info(f"[Ambient] 💡 Vector joins conversation: '{text}'")
                                    self.conversation_active = True
                                    self.last_interaction_time = datetime.now()
                                    await self._handle_user_speech(text)
                                    continue

                            logger.debug(f"❌ No wake word in transcription: '{text}'")
                    else:
                        logger.debug("No text in transcription result")
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}", exc_info=True)
                    
                finally:
                    # Clean up temp file
                    try:
                        if temp_wav.exists():
                            temp_wav.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp audio file: {e}")
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)

    async def _should_respond_ambient(self, text: str) -> bool:
        """
        Ask the LLM whether Vector should spontaneously join a room conversation.
        Used only in AMBIENT_MODE=true. Returns True if Vector should respond.
        """
        try:
            has_face = bool(self.memory and self.memory.current_faces)
            face_ctx = "Una persona sta guardando Vector." if has_face else "Nessuna persona è visualmente prominente."
            mood = self.memory.current_mood if self.memory else 50
            mood_str = "di buon umore" if mood >= 60 else ("di cattivo umore" if mood <= 35 else "neutro")
            prompt = (
                f"{face_ctx}\n"
                f"Vector è un robot curioso, amichevole e {mood_str}.\n"
                f"Una persona nella stanza ha detto: \"{text}\"\n"
                f"Vector deve intervenire nella conversazione? "
                f"Considera se la persona parla di lui, gli fa una domanda indiretta, "
                f"o se Vector ha qualcosa di genuinamente utile/divertente da aggiungere.\n"
                f"Rispondi con UNA SOLA PAROLA: SI oppure NO"
            )
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.chat_completion(messages, max_tokens=5, temperature=0.2)
            return "SI" in response.upper()
        except Exception as exc:
            logger.debug(f"[Ambient] Relevance check failed: {exc}")
            return False

    def _fallback_to_pc_mic(self):
        """Start PC microphone as fallback when Vector's AudioFeed is unavailable."""
        device_id = int(os.getenv('MICROPHONE_DEVICE_ID', '1'))
        logger.info(f"[Audio] Starting PC microphone fallback (device {device_id})...")
        try:
            self.audio_processor.start_listening(device=device_id)
            logger.info(f"🎤 PC microphone active (device {device_id})")
        except Exception as e:
            logger.error(f"[Audio] PC microphone fallback also failed: {e}")

    async def _vector_mic_stt_loop(self):
        """
        Sliding-window speech recognition for Vector's built-in microphone.

        Bypasses webrtcvad (which cannot cope with Vector's motor noise floor).
        Every SLIDE_SEC seconds we grab WINDOW_SEC of raw audio from the ring
        buffer, run Whisper with its own Silero-based VAD, and check for the
        wake word.  Whisper returns an empty string when no speech is present,
        so in practice this is zero-cost during silence.
        """
        import tempfile
        import wave as _wave

        WINDOW_SEC = 4.0   # Duration of each Whisper window
        SLIDE_SEC  = 1.5   # How often we re-run Whisper
        MIN_BUF_SEC = 2.0  # Don't start until we have this much audio buffered

        last_text = ""           # Dedup: skip if Whisper returns the same text twice
        last_processed_end = 0.0  # epoch time when last command was dispatched

        logger.info("[SlidingSTT] Sliding-window STT loop started (Whisper VAD, no webrtcvad)")

        while self.running:
            await asyncio.sleep(SLIDE_SEC)

            try:
                if not self.vector_mic or not self.speech_recognizer:
                    continue

                # Skip while TTS is speaking to avoid transcribing Vector's own voice
                if self.audio_processor and self.audio_processor.is_muted:
                    continue

                buffered = self.vector_mic.buffered_seconds
                if buffered < MIN_BUF_SEC:
                    continue  # Not enough audio yet

                audio_data = self.vector_mic.get_audio_window(WINDOW_SEC)
                if not audio_data:
                    continue

                # Write to temp WAV
                tmp = Path(tempfile.gettempdir()) / f"vec_slide_{datetime.now().strftime('%H%M%S_%f')}.wav"
                try:
                    with _wave.open(str(tmp), 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)

                    result = await self.speech_recognizer.transcribe(str(tmp))
                    text = (result.get('text') or '').strip()

                    if not text:
                        last_text = ""  # Reset dedup so next real speech isn't blocked
                        continue

                    # Deduplicate: skip if Whisper returned the exact same sentence again
                    if text == last_text:
                        continue
                    last_text = text

                    confidence = result.get('confidence', 0.5)
                    logger.info(f"[SlidingSTT] 📝 '{text}' (conf={confidence:.2f})")

                    # --- Hallucination filter ---
                    if hasattr(self.speech_recognizer, 'is_likely_hallucination'):
                        is_conv = self.conversation_active
                        bad, reason = self.speech_recognizer.is_likely_hallucination(text, confidence, is_conv)
                        if bad:
                            logger.debug(f"[SlidingSTT] 🚫 Hallucination ({reason}): '{text}'")
                            continue

                    # --- Wake word check ---
                    wake_words = ["ciao vector", "ehi vector", "salve vector", "hey vector", "vector"]
                    text_lower = text.lower()
                    has_wake = any(w in text_lower for w in wake_words)

                    # Cooldown: don't dispatch again within 3s of last command
                    now = datetime.now().timestamp()
                    if now - last_processed_end < 3.0 and not has_wake:
                        continue

                    if has_wake:
                        self.conversation_active = True
                        self.last_interaction_time = datetime.now()
                        last_processed_end = now

                        import re
                        remaining = text
                        for w in wake_words:
                            remaining = re.sub(r'\b' + re.escape(w) + r'\b', '', remaining, flags=re.IGNORECASE)
                        remaining = remaining.strip().strip(',').strip()

                        logger.info(f"[SlidingSTT] 💬 Wake word detected. Command: '{remaining or '(none)'}'")
                        if remaining:
                            await self._handle_user_speech(remaining)
                        else:
                            if self.tts:
                                if self.audio_processor:
                                    self.audio_processor.discard_pending_utterances()
                                await self.tts.speak("Sì? Come posso aiutarti?")

                    elif self.conversation_active:
                        expire_ok = (self.last_interaction_time is None or
                                     (datetime.now() - self.last_interaction_time).total_seconds() < self.conversation_timeout)
                        if expire_ok:
                            self.last_interaction_time = datetime.now()
                            last_processed_end = now
                            logger.info(f"[SlidingSTT] 💬 Follow-up: '{text}'")
                            await self._handle_user_speech(text)
                        else:
                            self.conversation_active = False
                            logger.info("[SlidingSTT] Conversation timed out")

                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass

            except Exception as exc:
                logger.error(f"[SlidingSTT] Error: {exc}", exc_info=True)
                await asyncio.sleep(2.0)

    def _trigger_emotion_animation(self, emotion: str, intensity: float = 0.5):
        """
        Trigger an animation based on emotion and intensity.
        Fire-and-forget: never blocks the response pipeline.
        
        Args:
            emotion: Emotion name (joy, curiosity, confusion, etc.)
            intensity: Emotion intensity 0.0-1.0 (affects animation selection)
        """
        if not animation_mapper or not self.robot:
            return
            
        try:
            if not animation_mapper.should_trigger(emotion):
                logger.debug(f"Animation probability check skipped for: {emotion}")
                return

            trigger = animation_mapper.pick_animation(emotion, intensity=intensity)
            if not trigger:
                logger.debug(f"No animation trigger found for emotion: {emotion}")
                return

            logger.info(f"🎭 Playing {emotion} animation: {trigger} (intensity: {intensity:.2f})")

            try:
                result = self.robot.anim.play_animation_trigger(trigger, ignore_body_track=True)

                # If the SDK returned a coroutine/awaitable, schedule it fire-and-forget
                if asyncio.iscoroutine(result) or hasattr(result, '__await__'):
                    asyncio.create_task(result)
                else:
                    logger.debug(f"Animation play returned status: {result}")

            except Exception as play_ex:
                # Single attempt only — never retry/block
                logger.warning(f"Animation '{trigger}' failed (non-blocking): {play_ex}")
        except Exception as e:
            logger.warning(f"Unexpected error in emotion animation {emotion}: {e}")
    
    async def handle_speech_for_wirepod(self, text: str) -> str:
        """
        Process transcribed speech from wire-pod's STT pipeline.

        Called by the WirePodBridge HTTP server when wire-pod forwards a
        user utterance.  Runs the full reasoning / memory / emotion pipeline,
        speaks the response via our Italian gTTS, and returns the response text.

        Wire-pod will receive back a " " (space) from the bridge so it does
        NOT double-speak the response via Vector's English built-in TTS.

        :param text: Transcribed speech from wire-pod.
        :returns:    Italian response text.
        """
        logger.info(f"[WirePodBridge] 🎤 Processing: '{text}'")
        try:
            llm_client = self.llm_client if hasattr(self, "llm_client") else getattr(self, "openai_client", None)
            if not self.reasoning_engine or not llm_client:
                return "Ho avuto un problema tecnico."

            from vector_personality.cognition.response_generator import ResponseGenerator

            context = await self.reasoning_engine.assemble_context()
            if self.context_builder:
                context["memory_context"] = await self.context_builder.build_conversation_context(user_text=text)

            response_gen = ResponseGenerator(
                openai_client=llm_client,
                personality_module=self.personality,
            )
            response = await response_gen.generate_response(
                user_input=text,
                conversation_history=self.conversation_history[-10:],
                context=context,
                mood=self.memory.current_mood if self.memory else 50,
            )

            # Update short-term conversation history
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": response})
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Persist to memory / embeddings
            if self.db:
                try:
                    face_id = None
                    if self.memory and self.memory.current_faces:
                        face_id = list(self.memory.current_faces.keys())[0]
                    await self.db.store_conversation(
                        speaker_id=face_id,
                        text=text,
                        room_id=self.memory.current_room_id if self.memory else None,
                        response_text=response,
                        vector_db=self.vector_db,
                        embedding_gen=self.embedding_gen,
                    )
                    if self.context_builder:
                        self.context_builder.invalidate_cache()
                except Exception as exc:
                    logger.warning(f"[WirePodBridge] DB save failed: {exc}")

            # Update conversation state so follow-ups via AudioFeed (if it works) work too
            self.conversation_active = True
            self.last_interaction_time = datetime.now()

            logger.info(f"[WirePodBridge] 🤖 Response: '{response}'")

            # Speak via our Italian gTTS (wire-pod will get " " back to avoid double-speak)
            if self.tts:
                if self.audio_processor:
                    self.audio_processor.discard_pending_utterances()
                await self.tts.speak(response)
            
            return response

        except Exception as exc:
            logger.error(f"[WirePodBridge] Error generating response: {exc}", exc_info=True)
            return "Mi dispiace, si è verificato un errore."

    async def _handle_user_speech(self, text: str):
        """
        Handle transcribed user speech (T085).
        
        Args:
            text: Transcribed speech text
        """
        # Prevent concurrent responses — discard incoming if already processing
        if self._speech_processing:
            logger.debug(f"⏭️ Already processing speech, skipping: '{text}'")
            return
        self._speech_processing = True
        try:
            # Note: Logging already done by caller (avoid duplicate "🎤 Heard:" messages)
            
            # Transition to PROCESSING state
            if self.task_manager:
                await self.task_manager.transition_to(TaskState.PROCESSING)
            
            # Get current face (if any)
            face_id = None
            if self.memory.current_faces:
                face_id = list(self.memory.current_faces.keys())[0]
            
            # If no face detected, reuse a single 'Unknown' face_id
            if not face_id:
                if not self.unknown_face_id:
                    self.unknown_face_id = str(uuid.uuid4())
                    logger.info(f"Using unknown face_id: {self.unknown_face_id}")
                face_id = self.unknown_face_id

            # Generate LLM response (T083 + T121 Groq)
            llm_client = self.llm_client if hasattr(self, 'llm_client') else self.openai_client
            
            if self.reasoning_engine and llm_client:
                from vector_personality.cognition.response_generator import ResponseGenerator
                
                response_gen = ResponseGenerator(
                    openai_client=llm_client,  # Works with both Groq and OpenAI
                    personality_module=self.personality
                )
                
                context = await self.reasoning_engine.assemble_context()

                # Build memory-grounded context string (T123)
                if self.context_builder:
                    memory_context = await self.context_builder.build_conversation_context(user_text=text)
                    context['memory_context'] = memory_context

                # Include announce_faces flag if startup greeting requested it (one-time)
                try:
                    if self.memory and hasattr(self.memory, 'pop_announce_faces') and self.memory.pop_announce_faces():
                        context['announce_faces'] = True
                        logger.info('📣 announce_faces flag set in context (one-time)')
                except Exception as e:
                    logger.debug(f'Could not set announce_faces flag in context: {e}')

                response = await response_gen.generate_response(
                    user_input=text,
                    conversation_history=self.conversation_history[-10:],  # last 5 turns
                    context=context,
                    mood=self.memory.current_mood
                )
                
                # Update short-term conversation history
                self.conversation_history.append({"role": "user", "content": text})
                self.conversation_history.append({"role": "assistant", "content": response})
                # Keep max 10 entries (5 turns)
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                
                logger.info(f"🤖 Response: {response}")
                
                # Trigger animation based on mood before speaking (T128)
                current_mood = self.memory.current_mood if self.memory else 50.0
                if current_mood >= 70:
                    self._trigger_emotion_animation('joy', intensity=0.8)
                elif current_mood >= 55:
                    self._trigger_emotion_animation('satisfied', intensity=0.5)
                elif current_mood <= 30:
                    self._trigger_emotion_animation('sadness', intensity=0.7)
                elif current_mood <= 45:
                    self._trigger_emotion_animation('confusion', intensity=0.4)
                else:
                    # Neutral mood - use context-based animation
                    if any(word in text.lower() for word in ['grazie', 'gracias', 'thank']):
                        self._trigger_emotion_animation('thanks', intensity=0.5)
                    elif any(word in text.lower() for word in ['chi', 'cosa', 'perché', 'dove', 'come']):
                        self._trigger_emotion_animation('thinking', intensity=0.6)
                
                # Speak response via TTS
                if self.tts:
                    # Discard stale queued utterances without wiping the pre-roll ring buffer
                    if self.audio_processor:
                        self.audio_processor.discard_pending_utterances()
                    await self.tts.speak(response)
                else:
                    self.robot.behavior.say_text(response)
            else:
                # Debug logging to help diagnose issue
                if not self.reasoning_engine:
                    logger.error("❌ reasoning_engine is None - check database connection")
                if not llm_client:
                    logger.error("❌ LLM client is None - check Ollama is running or set GROQ_API_KEY/OPENAI_API_KEY in api.env")
                    logger.error(f"❌ OllamaClient={OllamaClient is not None}, GroqClient={GroqClient is not None}, OpenAIClient={OpenAIClient is not None}")
                
                response = "I heard you, but my reasoning systems are not available."
                logger.warning(response)
            
            # Save conversation to database (T085) + embeddings (T140)
            if self.db:
                try:
                    # T140: Use store_conversation() to generate embeddings automatically
                    await self.db.store_conversation(
                        speaker_id=face_id,
                        text=text,
                        room_id=self.memory.current_room_id,
                        response_text=response,
                        vector_db=self.vector_db,       # T140: Pass vector DB for semantic search
                        embedding_gen=self.embedding_gen  # T140: Pass embedding generator
                    )
                    logger.info("💿 Conversation saved to database")

                    # Invalidate cached base context so the next user turn can recall this memory.
                    if self.context_builder:
                        try:
                            self.context_builder.invalidate_cache()
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Failed to save conversation: {e}")
        
        except Exception as e:
            logger.error(f"Error handling user speech: {e}", exc_info=True)
        finally:
            self._speech_processing = False

    async def _update_emotion(self):
        """Update mood and eye color based on current state."""
        if not self.mood_engine or not self.eye_color_mapper:
            return
            
        try:
            # Update eye color to reflect current mood from working memory
            current_mood = self.memory.current_mood
            if current_mood is not None:
                color = self.eye_color_mapper.mood_to_rgb(current_mood)
                await self.eye_color_mapper.set_vector_eyes(self.robot, color)
        
        except Exception as e:
            logger.error(f"Error updating emotion: {e}", exc_info=True)
    
    async def _check_autonomy_triggers(self):
        """Check if autonomous behaviors should be triggered."""
        if not self.autonomy_controller:
            return
            
        try:
            # DISABLED: Duplicate curiosity question system - CuriosityEngine handles this
            # if await self.autonomy_controller.should_initiate_interaction():
            #     await self.autonomy_controller.initiate_interaction()
            
            # Check for exploration trigger (but not during active conversation)
            # When user says "Ciao Vector", we pause exploration to focus on conversation
            if self.conversation_active:
                # Conversation in progress - skip exploration to avoid behavior control conflicts
                return
            
            if await self.autonomy_controller.should_explore_environment():
                await self.autonomy_controller.start_exploration()
            
            # Get attention target (face, object, or room)
            target = await self.autonomy_controller.get_attention_target()
            if target:
                logger.debug(f"Attention target: {target['type']}")
        
        except Exception as e:
            logger.error(f"Error checking autonomy triggers: {e}", exc_info=True)
    
    async def _check_curiosity(self):
        """Generate curiosity questions if conditions are met."""
        if not self.curiosity_engine:
            return
        
        # Don't ask questions too soon after startup
        if self.startup_time:
            time_since_startup = (datetime.now() - self.startup_time).total_seconds()
            if time_since_startup < self.startup_grace_period:
                logger.debug(f"⏳ Startup grace period active ({time_since_startup:.1f}s / {self.startup_grace_period}s)")
                return
        
        # Don't interrupt active conversations with curiosity questions
        if self.conversation_active:
            logger.debug("💬 Conversation active - skipping curiosity question")
            return
        
        # Don't interrupt when user is speaking
        if self.audio_processor and self.audio_processor.speech_detected:
            logger.debug("🗣️ User is speaking - skipping curiosity question")
            return
            
        try:
            logger.debug("🤔 Checking curiosity engine...")
            question = await self.curiosity_engine.generate_curiosity_question()
            if question:
                logger.info(f"💭 Curiosity question generated: {question}")
                
                # Trigger curiosity animation before asking question (T128)
                self._trigger_emotion_animation('curiosity', intensity=0.7)
                
                # Speak the question via TTS (T082)
                if self.tts:
                    # Discard stale queued utterances without wiping the pre-roll ring buffer
                    if self.audio_processor:
                        self.audio_processor.discard_pending_utterances()
                    logger.info("🔊 Speaking via TTS...")
                    await self.tts.speak(question)
                    logger.info("✅ TTS completed")
                else:
                    # Fallback to Vector's built-in TTS
                    logger.info("🔊 Speaking via built-in TTS...")
                    self.robot.behavior.say_text(question)
                    logger.info("✅ Built-in TTS completed")
                
                # Curiosity question spoken - user can respond naturally via always-on microphone
            else:
                logger.debug("No curiosity question generated (conditions not met)")
        
        except Exception as e:
            logger.error(f"Error checking curiosity: {e}", exc_info=True)
    
    async def _console_input_handler(self):
        """
        Background task to handle console input without blocking main loop.
        
        Continuously waits for console input and processes it when a question is pending.
        This allows Vector to keep moving and executing behaviors while waiting for user input.
        """
        try:
            while self.running:
                # Wait for console input (blocking, but in background task)
                user_answer = await asyncio.to_thread(input, "Your answer: ")
                
                # Check if we have a pending question
                if self.pending_console_input:
                    question, timestamp = self.pending_console_input
                    
                    # Check if answer is not empty
                    if user_answer and len(user_answer.strip()) > 0:
                        logger.info(f"✅ Received console answer: {user_answer.strip()}")
                        await self._handle_user_speech(user_answer.strip())
                    else:
                        logger.info("⏭️ Empty answer - skipping")
                    
                    # Clear pending question
                    self.pending_console_input = None
                else:
                    logger.debug("No pending question - ignoring input")
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Console input handler stopped")
        except Exception as e:
            logger.error(f"Error in console input handler: {e}", exc_info=True)
    
    async def _execute_tasks(self):
        """Execute pending tasks from the task queue."""
        if not self.task_manager:
            return
            
        try:
            # Check if should trigger exploration if idle too long
            await self.task_manager.trigger_exploration_if_idle()
            
            # Execute next task if available
            result = await self.task_manager.execute_next_task()
            if result:
                logger.debug(f"Task executed: {result}")
                # Reset idle timer when task completes
                if self.idle_controller:
                    self.idle_controller.reset()
            else:
                # No tasks in queue - execute idle behavior to keep Vector active
                if self.idle_controller:
                    if self.task_manager.current_state == TaskState.IDLE:
                        await self.idle_controller.execute_idle_behavior()
                    else:
                        logger.debug(f"Skipping idle behavior - state is {self.task_manager.current_state.value}, not IDLE")
        
        except Exception as e:
            logger.error(f"Error executing tasks: {e}", exc_info=True)
    
    def _register_event_handlers(self):
        """Register SDK event handlers for robot events."""
        # Face detection events (already handled by FaceDetectionManager)
        # Other events can be registered here
        
        # Subscribe to Wake Word events ("Ciao Vector" or Italian variants detected)
        self.robot.events.subscribe(self._on_wake_word, Events.wake_word)
        
        # Subscribe to User Intent events (Voice Commands after wake word)
        self.robot.events.subscribe(self._on_user_intent, Events.user_intent)
        
        # Example: Cube tap event
        async def on_cube_tap(robot, event, done):
            logger.info("🎲 Cube tapped!")
            await self.task_manager.manual_override("cube_tapped")
            # Could add more cube interaction logic here
        
        # Uncomment to enable cube events:
        # self.robot.events.subscribe(on_cube_tap, Events.cube_tapped)
        
        logger.info("Event handlers registered")

    async def _on_wake_word(self, robot, event_type, event):
        """Handle wake word detection (Ciao Vector or Italian variants)."""
        try:
            logger.info("👂 Wake word detected! Vector is listening...")
            # Transition to LISTENING state
            if self.task_manager:
                await self.task_manager.transition_to(TaskState.LISTENING)
        except Exception as e:
            logger.error(f"Error handling wake word: {e}", exc_info=True)

    async def _on_user_intent(self, robot, event_type, event):
        """Handle UserIntent events (Voice Commands)."""
        # Wire-Pod intent feature → Italian user-text mapping
        _FEATURE_TO_TEXT = {
            "ReactToHello": "ciao",
            "ReactToGoodMorning": "buongiorno",
            "ReactToGoodNight": "buonanotte",
            "ReactToHowAreYou": "come stai",
            "ReactToThanks": "grazie",
            "ReactToOk": "ok",
            "ReactToHoldOn": "aspetta",
            "ReactToEyeContact": "ciao",
            "HeyVector": "hey vector",
        }
        try:
            logger.info(f"🎤 User Intent Event Received")
            logger.info(f"   Event Type: {event_type}")

            # Extract text if available
            text = None

            # Try multiple attribute names (Wire-Pod, DDL, Anki formats)
            for attr in ['text', 'transcription', 'speech_text', 'query_text', 'user_text']:
                if hasattr(event, attr):
                    val = getattr(event, attr)
                    if val and isinstance(val, str) and len(val.strip()) > 0:
                        text = val
                        logger.info(f"   ✅ Found text in '{attr}': {text}")
                        break

            # Try json_data field — Wire-Pod sends intent details here
            if not text and hasattr(event, 'json_data'):
                import json
                try:
                    data = json.loads(event.json_data)
                    # 1. Explicit speech text
                    text = data.get('text') or data.get('transcription')
                    if text:
                        logger.info(f"   ✅ Found transcription in json_data: {text}")
                    else:
                        # 2. Map active_feature to a known Italian utterance
                        feature = data.get('active_feature', '')
                        mapped = _FEATURE_TO_TEXT.get(feature)
                        if mapped:
                            text = mapped
                            logger.info(f"   ✅ Mapped intent '{feature}' → '{text}'")
                        elif feature:
                            # Unknown feature — use a sanitised version as fallback
                            text = feature.replace('ReactTo', '').lower()
                            logger.info(f"   ⚠️ Unknown intent '{feature}', using: '{text}'")
                except Exception:
                    pass

            # If no text, log the intent name for debugging
            if not text and hasattr(event, 'intent'):
                intent_name = event.intent.name if hasattr(event.intent, 'name') else str(event.intent)
                logger.warning(f"   ⚠️ No text found. Intent name: {intent_name}")
                text = f"User triggered intent: {intent_name}"

            if text:
                logger.info(f"   📝 Processing speech: '{text}'")
                await self._handle_user_speech(text)
            else:
                logger.warning("   ❌ Could not extract any text from UserIntent event")
                logger.debug(f"   Event details: {event}")

        except Exception as e:
            logger.error(f"Error handling user intent: {e}", exc_info=True)
    
    def _display_status(self):
        """Display current agent status in CLI."""
        print("\n" + "="*60)
        print(f"🤖 VECTOR AGENT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Task state
        state_emoji = {
            TaskState.IDLE: "😴",
            TaskState.LISTENING: "👂",
            TaskState.PROCESSING: "🧠",
            TaskState.EXPLORING: "🔍",
            TaskState.LEARNING: "📚",
            TaskState.PAUSED: "⏸️"
        }
        current_state = self.task_manager.current_state
        print(f"State: {state_emoji.get(current_state, '❓')} {current_state.value}")
        
        # Mood
        mood = self.memory.current_mood
        if mood is not None:
            mood_emoji = "😊" if mood >= 70 else "😐" if mood >= 40 else "😔"
            print(f"Mood: {mood_emoji} {mood:.1f}/100")
        
        # Personality traits (effective values)
        traits = self.personality.effective_traits
        print(f"\nPersonality (effective):")
        print(f"  Curiosity: {traits.curiosity:.2f} | Courage: {traits.courage:.2f}")
        print(f"  Vitality: {traits.vitality:.2f} | Friendliness: {traits.friendliness:.2f}")
        print(f"  Touchiness: {traits.touchiness:.2f} | Sassiness: {traits.sassiness:.2f}")
        
        # Memory stats
        print(f"\nMemory:")
        print(f"  Faces seen: {len(self.memory.current_faces)}")
        print(f"  Objects detected: {len(self.memory.current_objects)}")
        print(f"  Current room: {self.memory.current_room or 'Unknown'}")
        
        # Perception stats
        if self.object_detector:
            fps = 1.0 / (sum(self.object_detector.inference_times[-10:]) / len(self.object_detector.inference_times[-10:])) if self.object_detector.inference_times else 0
            print(f"\nPerception:")
            print(f"  Object detection: {self.object_detector.total_detections} objects in {self.object_detector.total_frames} frames")
            print(f"  Detection FPS: {fps:.1f}")
            if self.room_inference and self.memory.current_room:
                print(f"  Room confidence: {self.room_inference.last_room_type or 'unknown'}")
        
        # Budget status removed (T122)
        
        # Recent emotion events (last 3)
        if hasattr(self.memory, 'emotion_history') and self.memory.emotion_history:
            print(f"\nRecent emotion events:")
            for event in list(self.memory.emotion_history)[-3:]:
                timestamp = event.timestamp.strftime("%H:%M:%S") if hasattr(event, 'timestamp') else 'unknown'
                mood_change = getattr(event, 'mood_delta', 0)
                print(f"  [{timestamp}] Mood change: {mood_change:+.1f}")
        
        print("="*60)
        print("Press Ctrl+C to shutdown gracefully\n")
    
    async def shutdown(self):
        """Gracefully shutdown all modules and save state."""
        if not self.running:
            return
        
        logger.info("🛑 Initiating graceful shutdown...")
        self.running = False
        
        try:
            # Cancel console input task
            if self.console_input_task and not self.console_input_task.done():
                logger.info("Stopping console input handler...")
                self.console_input_task.cancel()
                try:
                    await self.console_input_task
                except asyncio.CancelledError:
                    pass
            
            # Stop Vector microphone feed
            if self.vector_mic:
                logger.info("Stopping Vector microphone feed...")
                self.vector_mic.stop()

            # Stop PC microphone (if it was used)
            if self.audio_processor and self.audio_processor.stream:
                logger.info("Stopping PC microphone...")
                self.audio_processor.stop_listening()
            
            # Stop face detection
            if self.face_detector:
                logger.info("Stopping face detection...")
                await self.face_detector.stop()
            
            # Save personality state
            logger.info("Saving personality state...")
            # Personality auto-saves to database, no explicit save needed
            
            # Close database connections
            logger.info("Closing database connections...")
            await self.db.close()
            
            # Display final status
            self._display_status()
            
            logger.info("✅ Shutdown complete - goodbye!")
        
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


async def main():
    """
    Main entry point for the Vector Agent.
    
    Usage:
        python -m vector_personality.core.vector_agent
    
    Or from another script:
        from vector_personality.core.vector_agent import VectorAgent
        
        async def run():
            async with anki_vector.AsyncRobot() as robot:
                agent = VectorAgent(robot)
                await agent.start()
        
        asyncio.run(run())
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vector_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("="*60)
    logger.info("VECTOR PERSONALITY ENHANCEMENT - AGENT STARTUP")
    logger.info("="*60)

    try:
        # Connect to Vector
        logger.info("Connecting to Vector...")
        # Allow overriding the target Vector IP via environment (e.g. api.env -> VECTOR_HOST)
        vector_ip = os.environ.get('VECTOR_HOST') or os.environ.get('VECTOR_IP')
        robot_kwargs = {
            'enable_face_detection': True,
            'enable_audio_feed': True,  # Enable microphone streaming from Vector
        }
        if vector_ip:
            robot_kwargs['ip'] = vector_ip
            logger.info(f"Using explicit Vector IP from env: {vector_ip}")

        # Don't specify behavior_control_level to allow Vector's natural idle behaviors
        # (movements, beeps, eye animations) while still allowing SDK control when needed.
        # Vector will automatically pause his idle behavior when we give commands.
        robot = anki_vector.Robot(**robot_kwargs)
        logger.info("Establishing connection (this may take a moment)...")

        try:
            robot.connect(timeout=60)  # Increased timeout for animation loading
        except anki_vector.exceptions.VectorTimeoutException as e:
            error_msg = str(e)
            # Animation list loading can timeout - this is non-critical
            if "ListAnimations" in error_msg or "ListAnimationTriggers" in error_msg:
                logger.warning(f"⚠️ Animation list loading timed out - continuing without animation support")
                logger.info("✅ Basic connection established")
            else:
                # Other timeouts are critical
                raise

        logger.info("✅ Connected to Vector")
        
        try:
            # Create and start agent
            agent = VectorAgent(robot)
            await agent.start()
        finally:
            try:
                robot.disconnect()
            except Exception as e:
                # The SDK can raise CancelledError during shutdown if internal tasks
                # are cancelled while closing the control stream. This should not
                # be fatal for our process.
                logger.warning(f"⚠️ robot.disconnect() raised during shutdown: {e}")
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Run the agent
    asyncio.run(main())
