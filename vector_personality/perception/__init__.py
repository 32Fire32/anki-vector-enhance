"""
Perception Module

Phase 2: Perception Enhancement - Complete Implementation

Handles all sensory input processing:
- Audio processing with Voice Activity Detection (VAD)
- Speech recognition via OpenAI Whisper API
- Face detection (Phase 1 - already implemented)
- Room inference from object patterns
- Scene description via VLM (Phase 3)

Usage:
    from vector_personality.perception import (
        AudioProcessor,
        SpeechRecognizer,
        RoomInference,
        FaceDetectionHandler
    )
"""

# Phase 2 modules (NEW) - with graceful import handling
try:
    from .audio_processor import AudioProcessor, create_audio_processor
except ImportError as e:
    AudioProcessor = None
    create_audio_processor = None
    print(f"Warning: AudioProcessor not available: {e}")

try:
    from .speech_recognition import (
        SpeechRecognizer,
        create_speech_recognizer,
        estimate_whisper_cost
    )
except ImportError as e:
    SpeechRecognizer = None
    create_speech_recognizer = None
    estimate_whisper_cost = None
    print(f"Warning: SpeechRecognizer not available: {e}")

try:
    from .room_inference import (
        RoomInference,
        create_room_inference,
        room_type_to_emoji,
        detect_and_infer_room
    )
except ImportError as e:
    RoomInference = None
    create_room_inference = None
    room_type_to_emoji = None
    detect_and_infer_room = None
    print(f"Warning: RoomInference not available: {e}")

# Phase 1 module (existing)
try:
    from .face_detection import FaceDetectionHandler
except ImportError as e:
    FaceDetectionHandler = None
    print(f"Warning: FaceDetectionHandler not available: {e}")

__all__ = [
    # Audio
    'AudioProcessor',
    'create_audio_processor',
    
    # Speech
    'SpeechRecognizer',
    'create_speech_recognizer',
    'estimate_whisper_cost',
    
    # Room Inference
    'RoomInference',
    'create_room_inference',
    'room_type_to_emoji',
    'detect_and_infer_room',
    
    # Face Detection (Phase 1)
    'FaceDetectionHandler',
]
