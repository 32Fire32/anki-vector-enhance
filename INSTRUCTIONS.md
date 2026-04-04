# Anki Vector Enhance — Development Instructions

> Comprehensive reference for developing, extending, and maintaining the enhanced AI personality system for the Anki Vector robot.

---

ACTUAL IP: 192.168.1.13
NAME: Vector_E1F1
SERIAL: 00701d95

## Table of Contents

1. [How to Use](#how-to-use)
2. [Project Overview](#project-overview)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Language & Runtime](#language--runtime)
6. [Dependencies](#dependencies)
7. [Environment Setup](#environment-setup)
8. [Configuration](#configuration)
9. [Code Style Guide](#code-style-guide)
10. [Module Reference](#module-reference)
11. [Personality & Emotion System](#personality--emotion-system)
12. [Vision & Photo System](#vision--photo-system)
13. [Text-to-Speech System](#text-to-speech-system)
14. [Cognition & LLM Integration](#cognition--llm-integration)
15. [Database Layer](#database-layer)
16. [Testing](#testing)
17. [Build & Distribution](#build--distribution)
18. [Entry Points & Running](#entry-points--running)
19. [SDK Modifications Strategy](#sdk-modifications-strategy)
20. [Error Handling Patterns](#error-handling-patterns)
21. [Security & Privacy](#security--privacy)
22. [Key Design Decisions](#key-design-decisions)

---

n

## How to Use

### Prerequisites

- **Python 3.6.1+** (3.8+ recommended for full feature support)
- **Anki Vector robot** on the same local network as the development machine
- **ChromaDB** (installed via pip, runs locally — no external database server needed)
- **ffmpeg** installed and on PATH (required by `pydub` for audio conversion)

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> anki-vector-enhance
cd anki-vector-enhance

# 2. Install base SDK dependencies
pip install -r requirements.txt

# 3. Install personality system dependencies
pip install -r vector_personality/requirements.txt

# 4. Authenticate with your Vector robot (first time only)
python -m anki_vector.configure

# 5. Create the api.env file in the project root
#    (see Configuration section below)

# 6. Run the Italian experience
python start_ita.py

# 7. Run the full personality agent
python -m vector_personality.core.vector_agent

# 8. Run tests
pytest tests/ -v
```

### Common Tasks

| Task                                  | Command                                             |
| ------------------------------------- | --------------------------------------------------- |
| Capture a photo with object detection | `python vector_photo.py --model yolov5n --conf 0.5` |
| Run Italian TTS standalone            | `python vector_tts_it.py`                           |
| Test camera connection                | `python test_camera.py`                             |
| YOLOv5 inference test (no robot)      | `python yolo_test.py`                               |
| Start event supervisor service        | `python supervisor.py`                              |
| Install Windows service               | `python install_service.py install`                 |
| Build distribution                    | `make dist`                                         |
| Run a single test phase               | `pytest tests/test_phase3_emotion.py -v`            |

---

## Project Overview

This project transforms the Anki Vector robot from a basic companion into an **autonomous AI agent** with:

- **Persistent memory** (ChromaDB — unified vector + metadata storage)
- **6-dimensional personality** (curiosity, touchiness, vitality, friendliness, courage, sassiness)
- **Emotion engine** with mood-driven eye color and animation selection
- **Multi-modal perception** (face detection, YOLOv5 object recognition, speech recognition)
- **LLM-powered cognition** (Groq Llama 3.3 70B primary, OpenAI GPT-4 fallback)
- **Italian language support** (Google Translate + gTTS)
- **Autonomous behavior** (curiosity-driven exploration, task scheduling)

The system is built as a **wrapper/extension layer** over the original Anki Vector Python SDK — the SDK itself is minimally modified, preserving forward compatibility.

---

## Architecture

The system follows a **six-module modular architecture**:

```
┌─────────────────────────────────────────────────────┐
│                  VectorAgent (Orchestrator)          │
│                  vector_personality/core/            │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  Memory  │Perception│ Emotion  │Cognition │Behavior │
│  Phase 1 │ Phase 2  │ Phase 3  │ Phase 4  │ Phase 5 │
├──────────┼──────────┼──────────┼──────────┼─────────┤
│ChromaDB  │ Faces    │ Mood     │ Groq LLM │ Tasks   │
│ Vector   │ Objects  │ Eye RGB  │ OpenAI   │Autonomy │
│ Working  │ Speech   │ State    │ Context  │Curiosity│
│ Memory   │ TTS      │ Machine  │ Reasoning│Animation│
└──────────┴──────────┴──────────┴──────────┴─────────┘
                         │
              ┌──────────┴──────────┐
              │   Anki Vector SDK   │
              │    anki_vector/     │
              └─────────────────────┘
```

**Data flow:** Robot events → Perception → Memory + Emotion update → Cognition (if needed) → Behavior response → SDK actions

**Module independence:** Each module can fail gracefully without crashing the agent (conditional imports with try/except).

---

## Directory Structure

```
anki-vector-enhance/
├── anki_vector/                  # Modified Anki Vector Python SDK
│   ├── __init__.py               # SDK package root
│   ├── robot.py                  # Core Robot class (connection, lifecycle)
│   ├── behavior.py               # BehaviorComponent (movement, speech)
│   ├── camera.py                 # Camera feed interface
│   ├── events.py                 # Event subscription system
│   ├── connection.py             # gRPC connection, ControlPriorityLevel
│   ├── animation.py              # Animation trigger playback
│   ├── audio.py                  # Audio streaming (WAV playback)
│   ├── faces.py                  # Face recognition data
│   ├── motors.py                 # Motor control
│   ├── lights.py                 # LED control
│   ├── screen.py                 # Display/screen interface
│   ├── status.py                 # Robot status queries
│   ├── proximity.py              # Proximity sensor
│   ├── touch.py                  # Touch sensor
│   ├── nav_map.py                # Navigation map
│   ├── vision.py                 # Vision processing
│   ├── world.py                  # World state
│   ├── configure/                # Authentication setup scripts
│   ├── messaging/                # Protobuf message definitions
│   ├── opengl/                   # 3D viewer assets
│   └── camera_viewer/            # Camera viewer utility
│
├── vector_personality/           # Enhanced personality system (6 modules)
│   ├── __init__.py               # Package exports (VectorAgent, PersonalityTraits)
│   ├── requirements.txt          # Extended dependencies
│   ├── core/                     # Module 6: Core orchestration
│   │   ├── vector_agent.py       # Main event loop, module init, shutdown
│   │   ├── personality.py        # PersonalityTraits dataclass (6D)
│   │   └── config.py             # Defaults, room modifiers, cost tiers
│   ├── memory/                   # Module 1: Persistence layer
│   │   ├── working_memory.py     # Session-scoped volatile memory
│   │   ├── sql_server_connector.py  # Async SQL Server (Windows Auth)
│   │   ├── vector_db_connector.py   # ChromaDB semantic search
│   │   ├── embedding_generator.py   # Embedding computation
│   │   ├── context_summarizer.py    # Conversation history compression
│   │   └── schema.sql               # Database schema definition
│   ├── perception/               # Module 2: Sensor processing
│   │   ├── face_detection.py     # SDK face event handler
│   │   ├── object_detector.py    # YOLOv5 inference
│   │   ├── speech_recognition.py # Groq Whisper transcription
│   │   ├── text_to_speech.py     # gTTS Italian/English
│   │   ├── audio_processor.py    # VAD, audio buffering
│   │   └── room_inference.py     # Scene understanding
│   ├── emotion/                  # Module 3: Affective computing
│   │   ├── mood_engine.py        # 6 emotion drivers, decay
│   │   ├── eye_color_mapper.py   # Mood → RGB color
│   │   └── state_machine.py      # TaskState transitions
│   ├── cognition/                # Module 4: LLM integration
│   │   ├── groq_client.py        # Llama 3.3 70B (primary)
│   │   ├── openai_client.py      # GPT-4 (fallback)
│   │   ├── context_builder.py    # Prompt construction
│   │   ├── reasoning_engine.py   # Multi-step reasoning
│   │   └── response_generator.py # Post-processing for Vector
│   └── behavior/                 # Module 5: Action selection
│       ├── task_manager.py       # Priority queue task scheduling
│       ├── autonomy_controller.py # Autonomous vs. reactive decisions
│       ├── curiosity_engine.py   # Exploration task generation
│       ├── idle_controller.py    # Idle animations, eye color
│       ├── startup_controller.py # Boot sequence, greeting
│       └── animation_mapper.py   # Mood → animation triggers
│
├── tools/                        # Utility scripts
│   ├── animation_mapping.schema.json
│   ├── emotion_animation_mapping.json
│   ├── animation_triggers.json
│   ├── extract_animation_triggers.py
│   ├── build_emotion_mappings.py
│   ├── backfill_embeddings.py
│   └── ...
│
├── tests/                        # Pytest test suite (phase-based)
│   ├── test_phase1_memory.py
│   ├── test_phase2_perception.py
│   ├── test_phase3_emotion.py
│   ├── test_phase4_cognition.py
│   ├── test_phase5_behavior.py
│   ├── test_context_builder.py
│   ├── test_face_embeddings.py
│   ├── test_groq_fallback.py
│   ├── test_hallucination_filter.py
│   └── animation_tests/
│
├── vector-supervisor/            # Event logging service
├── RoboVec/                      # Legacy YOLOv5 training project
├── vector_docs/                  # Anki documentation cache
├── yolov5/                       # YOLOv5 source (ultralytics)
│
├── vector_ita.py                 # Italian behavior wrapper
├── start_ita.py                  # Italian experience entry point
├── vector_photo.py               # Capture + analyze from camera
├── vector_photo_saver.py         # Robust camera capture with retry
├── vector_photo_analyzer.py      # YOLOv5 image analysis
├── vector_tts_it.py              # Standalone Italian TTS
├── test_camera.py                # Camera validation
├── yolo_test.py                  # YOLOv5 test without robot
├── supervisor.py                 # Event logger to SQL Server
├── install_service.py            # Windows service installer
│
├── requirements.txt              # Base SDK dependencies
├── setup.py                      # Package build configuration
├── Makefile                      # Build targets (dist, wheel, clean)
├── CODESTYLE.md                  # Code style reference (PEP8/Google)
├── CONTRIBUTING.md               # Contributor license agreement
├── LICENSE.txt                   # Apache License 2.0
└── api.env                       # API keys (not committed)
```

---

## Language & Runtime

| Attribute           | Value                                       |
| ------------------- | ------------------------------------------- |
| **Language**        | Python 3 (3.6.1 minimum, 3.8+ recommended)  |
| **Async Framework** | `asyncio` (event loop, async/await)         |
| **SDK Protocol**    | gRPC via `aiogrpc` for robot communication  |
| **Serialization**   | Protocol Buffers (googleapis-common-protos) |
| **ML Runtime**      | PyTorch (CPU mode, YOLOv5 inference)        |
| **Database**        | ChromaDB (local, PersistentClient)          |
| **Package Format**  | setuptools (sdist + wheel)                  |
| **OS Target**       | Windows (primary), Linux (SDK compatible)   |

---

## Dependencies

### Base SDK (`requirements.txt`)

```
aiogrpc>=1.4              # Async gRPC client for Vector
cryptography              # SSL/TLS for SDK connection
flask                     # Web server (optional test endpoints)
googleapis-common-protos  # Protobuf definitions
numpy>=1.11               # Numerical computing
Pillow>=3.3               # Image capture and processing
requests                  # HTTP client
```

### Personality System (`vector_personality/requirements.txt`)

```
# Database
chromadb>=0.4.0           # Vector database for all persistence
# sqlalchemy>=1.4         # Optional ORM (not needed with ChromaDB)

# AI/ML
openai>=1.0.0             # GPT-4 API client
torch>=1.9.0              # YOLOv5 inference
torchvision>=0.10.0       # Image transforms

# Audio & Speech
pydub>=0.25.1             # Audio format conversion (MP3↔WAV)
gTTS>=2.2.4               # Google Text-to-Speech
librosa>=0.9.0            # Audio processing
webrtcvad>=4.0.0          # Voice Activity Detection
PyAudio>=0.2.11           # Microphone input

# Utilities
python-dotenv>=0.19.0     # .env file loading
Pillow>=8.0.0             # Image processing
dataclasses-json>=0.5     # Serialization

# Development
pytest>=6.0.0             # Unit testing
pytest-asyncio>=0.18.0    # Async tests
black>=21.0               # Code formatting
flake8>=3.9.0             # Linting
mypy>=0.910               # Type checking
```

### External (not pip-installable)

- **ffmpeg** — required by `pydub` for MP3↔WAV conversion
- **YOLOv5 models** — `yolov5n.pt` (nano) and `yolov5s.pt` (small) included in repo root

### External API Services

| Service                     | Library       | Purpose                 | Cost             |
| --------------------------- | ------------- | ----------------------- | ---------------- |
| **Groq** (Llama 3.3 70B)    | `groq`        | Primary LLM reasoning   | Free tier        |
| **Groq Whisper** (Large V3) | `groq`        | Speech-to-text          | Free tier        |
| **OpenAI** (GPT-4)          | `openai`      | Fallback LLM            | ~$0.03/1K tokens |
| **Google Translate**        | `googletrans` | EN→IT translation       | Free (limited)   |
| **Google TTS** (gTTS)       | `gtts`        | Text-to-speech          | Free             |
| **ChromaDB**                | `chromadb`    | Local vector embeddings | Free/local       |

---

## Environment Setup

### 1. Vector Robot Authentication

```bash
python -m anki_vector.configure
```

This stores credentials in `~/.anki_vector/` (serial number, IP, certificate, GUID).

The robot and development machine **must be on the same network**.

### 2. API Keys — `api.env`

Create an `api.env` file in the project root (this file is **NOT committed** to version control):

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
CHROMADB_DIR=./vector_memory_chroma
```

The `vector_agent.py` loads this file at startup via a custom `load_env_file()` function that reads key-value pairs and sets `os.environ`.

### 3. ChromaDB Database

The system uses **ChromaDB** as its sole persistence layer with `PersistentClient`:

```python
import chromadb
client = chromadb.PersistentClient(path="./vector_memory_chroma")
```

Data is stored in the following ChromaDB collections:

- `faces` — known face records (name, relationship, interaction counts)
- `face_embeddings` — face recognition embedding vectors (128/512-dim)
- `conversations` — conversation logs (text, response, speaker, timestamp)
- `objects` — detected objects with positions and confidence
- `rooms` — room/scene context and detected objects
- `personality_learned` — personality trait adjustments over time
- `supervisor_events` — robot event logs (supervisor.py)

Collections that store metadata-only records use dummy embeddings `[0.0, 0.0, 0.0]`, while face_embeddings and the semantic search collection (`vector_db_connector.py`) use real vector embeddings.

The `CHROMADB_DIR` environment variable (default: `./vector_memory_chroma`) controls the database location.

---

## Configuration

### Personality Defaults (`vector_personality/core/config.py`)

```python
DEFAULT_PERSONALITY_TRAITS = {
    'curiosity': 0.7,       # Moderately curious
    'touchiness': 0.6,      # Medium touch sensitivity
    'vitality': 0.8,        # High energy
    'friendliness': 0.7,    # Warm and engaging
    'courage': 0.5,         # Cautious but willing
    'sassiness': 0.3,       # Subtle sarcasm
}
```

### Room-Specific Modifiers

```python
ROOM_BEHAVIOR_ADJUSTMENTS = {
    'working_desk': { 'curiosity_reduction': 0.3, 'expressiveness_reduction': 0.4 },
    'kids_room':    { 'playfulness_boost': 0.9, 'vocalization_increase': 0.3 },
    'bedroom':      { 'quietness_multiplier': 0.8, 'curiosity_reduction': 0.4 },
    'kitchen':      { 'curiosity_increase': 0.2 },
    'living_room':  { 'friendliness_boost': 0.2 },
}
```

### API Cost Tiers

```python
API_COST_TIERS = {
    'free': 0.0,          # SDK-only actions
    'cheap': 0.002,       # Context-aware responses
    'moderate': 0.01,     # Complex reasoning
    'expensive': 0.05,    # Novel problem-solving
}
```

### Emotion Drivers

```python
EMOTION_DRIVERS = {
    'face_recognized': +10,
    'face_new': +5,
    'petted': +10,            # Up to +30 (touchiness-dependent)
    'moved_roughly': -20,     # Down to -50 (touchiness-dependent)
    'ignored_per_minute': -10,
    'task_success': +15,
    'curiosity_satisfied': +20,
    'contradiction': -5,
}
```

### Mood → Eye Color Mapping

```python
MOOD_TO_COLOR = {
    0:   (255, 0, 0),       # Angry: red
    20:  (255, 64, 0),      # Upset: red-orange
    40:  (255, 128, 0),     # Grumpy: orange
    60:  (255, 255, 0),     # Neutral: yellow
    80:  (0, 255, 0),       # Content: green
    100: (0, 255, 255),     # Joyful: cyan
}
```

---

## Code Style Guide

The project follows **PEP 8** with **Google-style docstrings**. Full reference: [CODESTYLE.md](CODESTYLE.md).

### Summary of Rules

| Rule                    | Convention                                                                   |
| ----------------------- | ---------------------------------------------------------------------------- |
| **Line length**         | 80 characters maximum (soft limit)                                           |
| **Indentation**         | 4 spaces (no tabs)                                                           |
| **Blank lines**         | 2 between top-level definitions; 1 between methods                           |
| **Constants**           | `UPPER_CASE`                                                                 |
| **Classes**             | `CamelCase`                                                                  |
| **Functions/variables** | `lowercase_with_underscores`                                                 |
| **Private members**     | Leading underscore: `_private_method`                                        |
| **Docstrings**          | Google style (Sphinx-extractable)                                            |
| **Documentation**       | Required for all public classes, methods, and functions                      |
| **Type hints**          | Used throughout (`Optional[str]`, `List[Dict[str, Any]]`, `Tuple[Any, Any]`) |

### Import Order

Three blocks, each sorted alphabetically, separated by blank lines:

```python
# 1. Standard library
import asyncio
import os
import sys

# 2. Third-party packages
import numpy
import torch
from PIL import Image

# 3. Local packages
from vector_personality.core.personality import PersonalityModule
from vector_personality.memory.working_memory import WorkingMemory
```

**Avoid:** wildcard imports (`from module import *`) outside tests.

**Prefer:** qualified names (`import module` → `module.object`) over direct imports for better mockability in tests.

### String Formatting

Long strings use parenthesized continuation:

```python
long_string = ('First long line...'
    'Second long line')
```

### Logging

- Per-module loggers: `logger = logging.getLogger(__name__)`
- Suppress noisy libraries at module level:

```python
logging.getLogger('aiogrpc').setLevel(logging.WARNING)
logging.getLogger('anki_vector.connection').setLevel(logging.ERROR)
```

### Dataclasses

Used for structured data (personality traits, observations, history entries):

```python
@dataclass
class PersonalityTraits:
    curiosity: float = 0.7
    touchiness: float = 0.6
    # ... with __post_init__ for clamping
```

---

## Module Reference

### Core (`vector_personality/core/`)

| File              | Purpose                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `vector_agent.py` | **Main orchestrator** — event loop, module init, CLI monitoring, signal handling, graceful shutdown |
| `personality.py`  | `PersonalityTraits` (6D dataclass) + `PersonalityModule` (base + learned deltas)                    |
| `config.py`       | All configuration defaults: traits, rooms, costs, drivers, colors, priorities                       |

### Memory (`vector_personality/memory/`)

| File                      | Purpose                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `working_memory.py`       | Session-scoped volatile state: mood (0-100), `TaskState`, face/object observations, emotion history, speech buffer |
| `chromadb_connector.py`   | Async ChromaDB connector — unified persistence for faces, conversations, objects, rooms, personality               |
| `sql_server_connector.py` | **(Deprecated)** Legacy SQL Server connector, kept for reference only                                              |
| `vector_db_connector.py`  | ChromaDB integration for semantic search over conversation history (separate from main persistence)                |
| `embedding_generator.py`  | Compute vector embeddings for conversation storage                                                                 |
| `context_summarizer.py`   | Compress long conversation histories for LLM context window                                                        |
| `schema.sql`              | Database DDL for all tables                                                                                        |

### Perception (`vector_personality/perception/`)

| File                    | Purpose                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `face_detection.py`     | Subscribe to SDK `robot_observed_face` events, store in ChromaDB, trigger greetings |
| `object_detector.py`    | YOLOv5 inference (COCO 80 classes), confidence filtering                            |
| `speech_recognition.py` | Groq Whisper Large V3 transcription with confidence scoring and retry               |
| `text_to_speech.py`     | gTTS (Italian/English), WAV conversion, Vector speaker playback                     |
| `audio_processor.py`    | Voice Activity Detection (VAD), audio buffering                                     |
| `room_inference.py`     | Infer room/scene from visual context                                                |

### Emotion (`vector_personality/emotion/`)

| File                  | Purpose                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `mood_engine.py`      | 6 emotion drivers (curiosity, satisfaction, excitement, loneliness, confusion, frustration) with configurable decay rates |
| `eye_color_mapper.py` | Smooth interpolation from mood (0-100) → RGB eye color                                                                    |
| `state_machine.py`    | `TaskState` transitions with validation (IDLE → LISTENING → PROCESSING → EXPLORING)                                       |

### Cognition (`vector_personality/cognition/`)

| File                    | Purpose                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `groq_client.py`        | Llama 3.3 70B (primary) + 3.1 8B-instant (fallback), with OpenAI fallback on error |
| `openai_client.py`      | GPT-4 wrapper (secondary fallback, cost-tracked)                                   |
| `context_builder.py`    | Construct LLM prompts with working memory state, personality, conversation history |
| `reasoning_engine.py`   | Multi-step reasoning chains, fact verification against memory                      |
| `response_generator.py` | Post-process LLM output for Vector (extract animation triggers, emotion markers)   |

### Behavior (`vector_personality/behavior/`)

| File                     | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `task_manager.py`        | Priority queue: critical(100), high(75), medium(50), low(25) |
| `autonomy_controller.py` | Decide autonomous action vs. wait for user input             |
| `curiosity_engine.py`    | Generate exploration tasks (ask questions, investigate room) |
| `idle_controller.py`     | Idle animations, eye color refresh, smooth state transitions |
| `startup_controller.py`  | Boot initialization sequence and greeting                    |
| `animation_mapper.py`    | Map mood/emotion → Anki animation triggers                   |

---

## Personality & Emotion System

### Six-Dimensional Personality Model

Each trait is a float in `[0.0, 1.0]`:

| Trait            | Default | Effect                                                           |
| ---------------- | ------- | ---------------------------------------------------------------- |
| **Curiosity**    | 0.7     | Drive to explore, ask questions, trigger autonomous tasks        |
| **Touchiness**   | 0.6     | Sensitivity to physical contact (amplifies pet/rough mood delta) |
| **Vitality**     | 0.8     | Energy level, influences API budget willingness                  |
| **Friendliness** | 0.7     | Warmth in responses, greeting enthusiasm                         |
| **Courage**      | 0.5     | Willingness to try unfamiliar actions                            |
| **Sassiness**    | 0.3     | Boldness and sarcasm in generated responses                      |

Traits have **two components**:

- **Base traits** — configuration defaults
- **Learned deltas** — accumulated from user feedback over time

Combined via `PersonalityTraits.__add__()` with automatic clamping in `__post_init__`.

### Mood Engine

- Mood is a single value `[0, 100]` with baseline 50
- Six **emotion drivers** each contribute to mood with weighted averaging:
  - **Positive:** Curiosity (+0.20 weight), Satisfaction (+0.25), Excitement (+0.15)
  - **Negative:** Loneliness (-0.15), Confusion (-0.15), Frustration (-0.10)
- Each driver **decays toward baseline** at configurable rates per second:
  - Curiosity: 0.05/s (slow decay)
  - Excitement: 0.15/s (fast decay)
  - Satisfaction: 0.08/s

### Eye Color Mapping

Mood value smoothly maps to RGB via interpolation:

```
0 (Red) → 20 (Red-Orange) → 40 (Orange) → 60 (Yellow) → 80 (Green) → 100 (Cyan)
  Angry      Upset            Grumpy        Neutral        Content       Joyful
```

### Task State Machine

States: `IDLE` → `LISTENING` → `PROCESSING` → `EXPLORING`

Transitions are validated — invalid transitions are blocked.

---

## Vision & Photo System

### Capture Pipeline (`vector_photo_saver.py`)

1. Connect to Vector robot
2. Initialize camera feed
3. Poll frames at 0.5s intervals (20s max timeout)
4. Export as PIL Image (RGB)
5. Cleanup: `close_camera_feed()` + `disconnect()`

**Robustness features:**

- Retry loop with exponential backoff (default 5 retries, 2s initial)
- `behavior_activation_timeout` configurable (300s default)
- Fallback to existing local image if `fallback_to_local=True`

### Object Detection (`vector_photo_analyzer.py`)

- **Models:** `yolov5n.pt` (nano, fast) or `yolov5s.pt` (small, more accurate)
- **Framework:** PyTorch `torch.hub.load('ultralytics/yolov5', ...)`
- **Input:** PIL Image or file path
- **Output:** Pandas DataFrame (xmin, ymin, xmax, ymax, confidence, class_name) + annotated image
- **Classes:** COCO 80 (person, dog, cat, cup, bottle, laptop, phone, etc.)
- **Inference:** CPU-friendly, `torch.no_grad()` context for efficiency

### CLI Options (`vector_photo.py`)

```bash
python vector_photo.py --ip <robot_ip> --model yolov5n --conf 0.5 --timeout 15 --retries 3
```

---

## Text-to-Speech System

### Pipeline

```
English text → Google Translate (EN→IT) → gTTS MP3 → pydub → PCM WAV (16kHz, 16-bit, mono) → Vector speaker
```

### Audio Format Requirements

Vector SDK requires: **WAV, PCM 16-bit, 16000 Hz, mono**

```python
audio = AudioSegment.from_mp3(mp3_file)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)
audio.export(wav_file, format="wav")
```

### Playback

```python
robot.audio.stream_wav_file(wav_file)  # Synchronous, do NOT await
```

### Cleanup

Temporary MP3/WAV files are **always** removed in `finally` blocks.

---

## Cognition & LLM Integration

### Fallback Chain

```
Groq (Llama 3.3 70B)  →  Groq (Llama 3.1 8B-instant)  →  OpenAI (GPT-4)  →  Hardcoded responses
       PRIMARY                   FAST FALLBACK                EXPENSIVE           OFFLINE FALLBACK
```

### Context Building

The `ContextBuilder` assembles LLM prompts from:

- Working memory state (mood, current faces/objects, task)
- Personality traits (influences tone and style)
- Conversation history (summarized if too long)
- ChromaDB semantic search results (relevant past conversations)

### Budget Enforcement

- `BudgetEnforcer` tracks API costs per hour
- Default limit: **€2/hour**
- Cost tiers route requests to appropriate LLM:
  - Free: SDK-only (no LLM call)
  - Cheap (€0.002): Groq (actually free-tier)
  - Moderate (€0.01): Complex reasoning via Groq
  - Expensive (€0.05): GPT-4 for novel problems

### Hallucination Filter

The `reasoning_engine.py` validates LLM responses against memory to catch factual contradictions.

---

## Database Layer

### ChromaDB (Unified Persistence)

All data is stored in a local ChromaDB instance using `PersistentClient`. No external database server required.

- **Connector:** `chromadb_connector.py` — `ChromaDBConnector` class with async methods
- **Async pattern:** `ThreadPoolExecutor` wraps synchronous ChromaDB calls to avoid blocking asyncio:

```python
async def store_conversation(self, text, response_text, speaker_id):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(self._executor, self._sync_store_conversation, ...)
```

- **Dummy embeddings:** Collections that only need metadata storage use `[0.0, 0.0, 0.0]` embeddings (ChromaDB requires embeddings for all documents)
- **Backward compatibility:** `query()` and `execute()` methods parse common SQL patterns used by `face_detection.py` and route them to the appropriate ChromaDB collection methods

### ChromaDB Semantic Search (`vector_db_connector.py`)

- Separate ChromaDB connector specifically for conversation embedding search
- Stores real vector embeddings generated by `embedding_generator.py`
- Used by `ContextBuilder` to retrieve semantically relevant past interactions

### Collections

| Collection            | Key Metadata Fields                                     |
| --------------------- | ------------------------------------------------------- |
| `faces`               | name, relationship, interaction_count, last_seen        |
| `face_embeddings`     | face_id, dimension (real embeddings, cosine similarity) |
| `conversations`       | text, response_text, speaker_id, timestamp              |
| `objects`             | object_type, pos_x, pos_y, pos_z, confidence, room      |
| `rooms`               | room_name, observed_objects (JSON), last_visit          |
| `personality_learned` | trait_name, delta_value, reason, timestamp              |
| `supervisor_events`   | event_type, timestamp, plus event-specific fields       |

---

## Testing

### Framework

- **pytest** with `pytest-asyncio` for async tests
- **unittest.mock** (`Mock`, `AsyncMock`, `patch`, `MagicMock`)
- **Phase-based test organization** matching module phases

### Running Tests

```bash
# All tests
pytest tests/ -v

# Single phase
pytest tests/test_phase3_emotion.py -v

# With coverage
pytest tests/ -v --cov=vector_personality
```

### Test Patterns

**1. Phase-based test files** — one file per architecture phase:

```
test_phase1_memory.py      → WorkingMemory, SQL connector, face tracking
test_phase2_perception.py  → Face detection, speech recognition, object detection
test_phase3_emotion.py     → MoodEngine, EyeColorMapper, StateMachine
test_phase4_cognition.py   → BudgetEnforcer, LLM clients, ContextBuilder
test_phase5_behavior.py    → Task scheduling, autonomy, animation mapping
```

**2. Conditional skipmodule imports** — tests skip gracefully if the module under test is not yet implemented:

```python
try:
    from vector_personality.emotion.mood_engine import MoodEngine
except ImportError:
    MoodEngine = None

@pytest.mark.skipif(MoodEngine is None, reason="MoodEngine not yet implemented")
class TestMoodEngine:
    ...
```

**3. Setup pattern** — `setup_method()` for per-test initialization:

```python
class TestWorkingMemory:
    def test_initialization(self):
        wm = WorkingMemory(initial_mood=60)
        assert wm.current_mood == 60
```

**4. Mock database calls:**

```python
self.mock_db = MagicMock()
self.mock_db.query = AsyncMock(return_value=[{'total': 0.0}])
self.mock_db.execute = AsyncMock()
```

**5. `sys.path` manipulation** for imports from parent directory:

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

---

## Build & Distribution

### Makefile Targets

| Target          | Action                                  |
| --------------- | --------------------------------------- |
| `make dist`     | Build sdist + wheel + examples          |
| `make wheel`    | Build wheel only                        |
| `make examples` | Package example scripts as tar.gz + zip |
| `make license`  | Copy LICENSE.txt to output directories  |
| `make clean`    | Remove `dist/` directory                |

### setup.py Configuration

- **Package name:** `anki_vector`
- **Version:** Read from `anki_vector/version.py` (`__version__ = "0.6.1.dev0"`)
- **Packages:** `anki_vector` + submodules (`camera_viewer`, `configure`, `messaging`, `opengl`, `reserve_control`)
- **License:** Apache License 2.0

### Optional Extras

```bash
pip install .[3dviewer]       # PyOpenGL for 3D viewer
pip install .[experimental]   # Keras, TensorFlow, scikit-learn
pip install .[test]           # pytest, requests_toolbelt
pip install .[docs]           # Sphinx documentation
```

---

## Entry Points & Running

### Italian Experience

```bash
python start_ita.py
```

Flow: retry logic (3 attempts, 5s backoff) → Vector connection → `RobotItaliano` wrapper → Italian speech → event loop.

### Full Personality Agent

```bash
python -m vector_personality.core.vector_agent
```

Flow: load `api.env` → connect to Vector → init database → create all 6 modules → subscribe SDK events → main loop with CLI status → graceful shutdown on SIGINT.

### Event Supervisor

```bash
python supervisor.py
```

Subscribes to all Vector events and logs them to ChromaDB `supervisor_events` collection (RobotObservedFace, ObservedObjects, NavMap, RobotState).

### Windows Service

```bash
python install_service.py install
python install_service.py start
```

Runs the supervisor as the `VectorSupervisorService` Windows service.

---

## SDK Modifications Strategy

The project uses a **wrapper/extension pattern** rather than forking the SDK:

1. **`ItalianBehavior`** extends `BehaviorComponent` — intercepts `say_text()` for translation + TTS
2. **`RobotItaliano`** wraps the original `Robot` — delegates via `__getattr__`, swaps only the `behavior` component
3. **Event subscriptions** in `VectorAgent` listen to stock SDK events and route data through the personality pipeline
4. **No core SDK files are modified** — this preserves compatibility with future SDK updates

### Key SDK APIs Used

```python
robot = anki_vector.Robot()
robot.connect()

# Events
robot.events.subscribe(handler, Events.robot_observed_face)

# Camera
robot.camera.init_camera_feed()
robot.camera.latest_image  # PIL Image

# Audio
robot.audio.stream_wav_file(wav_path)  # Synchronous

# Animation
robot.anim.play_animation_trigger('anim_greeting_happy')

# Behavior
robot.behavior.say_text("Hello")
robot.behavior.set_head_angle(degrees(10))
robot.behavior.set_lift_height(0.0)

# Control
ControlPriorityLevel  # Behavior arbitration priority
```

---

## Error Handling Patterns

### Retry with Exponential Backoff

Used in camera capture, robot connection, API calls:

```python
for attempt in range(max_retries):
    try:
        result = perform_action()
        break
    except TransientError:
        wait = initial_backoff * (2 ** attempt)
        await asyncio.sleep(wait)
```

### Fallback Chain

```python
try:
    response = await groq_client.chat(prompt)
except GroqError:
    try:
        response = await openai_client.chat(prompt)
    except OpenAIError:
        response = HARDCODED_FALLBACK
```

### Graceful Module Failure

```python
try:
    from vector_personality.perception.face_detection import FaceDetectionHandler
except ImportError:
    FaceDetectionHandler = None  # Module not available, agent continues
```

### Resource Cleanup

Temp files always cleaned in `finally` blocks. Camera feeds always closed. Database connections properly disposed.

---

## Security & Privacy

- **API keys** stored in `api.env` (excluded from version control)
- **Groq API** — no data retention policy (privacy-first)
- **ChromaDB** — local SQLite-backed storage, no cloud upload, no external server
- **Temp files** — always cleaned up in `finally` blocks
- **No credentials logged** — logging suppresses sensitive data

---

## Key Design Decisions

| Decision                          | Rationale                                                         |
| --------------------------------- | ----------------------------------------------------------------- |
| **Wrapper pattern over SDK fork** | Maintains compatibility with future SDK updates                   |
| **Groq as primary LLM**           | Free tier, fast inference, no data retention                      |
| **ChromaDB-only persistence**     | Eliminates SQL Server dependency; single local store for all data |
| **Phase-based development**       | Each module is independently testable and deployable              |
| **Conditional imports**           | Agent doesn't crash when optional modules are missing             |
| **ThreadPoolExecutor for DB**     | Prevents synchronous ChromaDB from blocking the asyncio loop      |
| **gTTS over paid TTS**            | Free, good quality, Italian support                               |
| **YOLOv5 nano model default**     | CPU-friendly, fast inference on limited hardware                  |
| **Six-dimensional personality**   | Rich enough for nuanced behavior, simple enough to tune           |
| **Mood decay toward baseline**    | Prevents emotional extremes, natural-feeling recovery             |
| **Room-specific behavior**        | Context-appropriate personality modulation                        |
| **Budget enforcement**            | Prevents runaway API costs during autonomous operation            |
| **Italian as default language**   | Primary user locale; easily extensible to other languages         |
