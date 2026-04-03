"""
Working memory module for Vector Personality Project.
Manages session-scoped state (resets when Vector is idle/turned off).

Principle II: Persistent Memory Architecture (Tier 1: Working Memory)
This is the fast, volatile memory that exists only during active sessions.

Phase 3 Integration: Can optionally integrate with MoodEngine for
advanced emotion tracking with decay and personality influence.
"""

import logging
from typing import Optional, List, Dict, Any, Set, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from vector_personality.emotion.mood_engine import MoodEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Current task state (Principle IV: Autonomous Curiosity)"""
    IDLE = "idle"  # No active task
    LISTENING = "listening"  # Waiting for speech input
    PROCESSING = "processing"  # Processing speech/API call in progress
    EXPLORING = "exploring"  # Autonomous exploration mode
    LEARNING = "learning"  # Processing new information
    PAUSED = "paused"  # User requested pause


@dataclass
class FaceObservation:
    """Represents a face seen in the current session."""
    face_id: str
    name: Optional[str]
    first_seen_session: datetime
    last_seen_session: datetime
    interaction_count: int = 0
    last_mood_impact: int = 0  # -100 to +100


@dataclass
class ObjectObservation:
    """Represents an object detected in the current session."""
    object_type: str
    confidence: float
    first_detected_session: datetime
    last_detected_session: datetime
    detection_count: int = 1
    location_description: Optional[str] = None


@dataclass
class EmotionHistoryEntry:
    """Single emotion event in working memory."""
    timestamp: datetime
    event_type: str  # 'face_recognized', 'moved_roughly', 'pet_head', etc.
    mood_delta: int  # -100 to +100
    new_mood: int  # 0-100


class WorkingMemory:
    """
    Session-scoped working memory (Tier 1 of dual-tier architecture).
    Clears when Vector is idle or turned off.
    
    Principle II: Persistent Memory Architecture
    - Fast access for current session data
    - Flushes to SQL Server periodically
    - Resets on task state transitions
    
    Principle III: Emotional Authenticity
    - Tracks mood history for current session
    - Maintains emotion event log
    """
    
    def __init__(self, initial_mood: int = 50, mood_engine: Optional['MoodEngine'] = None):
        """
        Initialize working memory.
        
        Args:
            initial_mood: Starting mood (0-100, default: 50 = neutral)
            mood_engine: Optional MoodEngine for advanced emotion tracking (Phase 3)
        """
        # Current session state
        self.current_mood: int = initial_mood
        self.current_task: TaskState = TaskState.IDLE
        self.session_start: datetime = datetime.now()
        
        # Phase 3: Optional MoodEngine integration
        self.mood_engine: Optional['MoodEngine'] = mood_engine
        
        # Observations
        self.current_faces: Dict[str, FaceObservation] = {}  # face_id -> observation
        self.current_objects: Dict[str, ObjectObservation] = {}  # object_type -> observation
        self.current_room: Optional[str] = None  # room_name
        self.current_room_id: Optional[str] = None  # room UUID
        
        # Emotion tracking (Principle III)
        self.emotion_history: List[EmotionHistoryEntry] = []
        self.max_emotion_history: int = 100  # Keep last 100 events
        
        # Speech buffer (Phase 2)
        self.recent_speech_buffer: List[str] = []
        self.max_speech_buffer: int = 10
        
        # Context flags
        self.is_being_held: bool = False
        self.is_on_charger: bool = False
        self.last_touch_time: Optional[datetime] = None

        # Face announcement control: temporarily announce faces in responses
        self._announce_faces_until: Optional[datetime] = None
        
        logger.info(f"Working memory initialized at mood {initial_mood}")
    
    # ========== Face Management ==========
    
    def observe_face(
        self,
        face_id: str,
        name: Optional[str] = None,
        mood_impact: int = 10
    ) -> FaceObservation:
        """
        Record face observation in working memory.
        
        Args:
            face_id: UUID from database
            name: Person's name (if known)
            mood_impact: Mood delta for this face (+10 for recognized face)
        
        Returns:
            FaceObservation instance
        """
        now = datetime.now()
        
        # Cooldown per faccia: solo 1 mood update ogni 30 secondi per la stessa faccia
        mood_cooldown_seconds = 30
        should_update_mood = False
        
        if face_id in self.current_faces:
            # Update existing
            obs = self.current_faces[face_id]
            
            # Check if enough time has passed since last mood update for this face
            time_since_last = (now - obs.last_seen_session).total_seconds()
            if time_since_last >= mood_cooldown_seconds:
                should_update_mood = True
            
            obs.last_seen_session = now
            obs.interaction_count += 1
            obs.last_mood_impact = mood_impact
        else:
            # Create new - always update mood for first observation
            should_update_mood = True
            obs = FaceObservation(
                face_id=face_id,
                name=name,
                first_seen_session=now,
                last_seen_session=now,
                interaction_count=1,
                last_mood_impact=mood_impact
            )
            self.current_faces[face_id] = obs
        
        # Update mood only if cooldown passed
        if should_update_mood:
            self.update_mood(mood_impact, event_type='face_recognized')
            logger.info(f"Face observed: {name or 'Unknown'} (mood delta: {mood_impact:+d})")
        else:
            logger.debug(f"Face observed: {name or 'Unknown'} (mood cooldown active, no update)")
        
        return obs
    
    def get_active_faces(self, max_age_seconds: int = 300) -> List[FaceObservation]:
        """
        Get faces seen recently in this session.
        
        Args:
            max_age_seconds: Only include faces seen within this timeframe
        
        Returns:
            List of FaceObservation instances
        """
        now = datetime.now()
        active = []
        
        for obs in self.current_faces.values():
            age = (now - obs.last_seen_session).total_seconds()
            if age <= max_age_seconds:
                active.append(obs)
        
        return sorted(active, key=lambda x: x.last_seen_session, reverse=True)

    # ========== Face announcement control ==========
    def set_announce_faces(self, duration_seconds: int = 180):
        """Temporarily allow face mentions in the next responses.

        Args:
            duration_seconds: How long to allow automatic face mentions (default: 180s)
        """
        from datetime import datetime, timedelta
        self._announce_faces_until = datetime.now() + timedelta(seconds=duration_seconds)
        logger.info(f"Announce faces enabled for next {duration_seconds}s")

    def should_announce_faces(self) -> bool:
        """Return True if face announcement window is active."""
        from datetime import datetime
        if self._announce_faces_until is None:
            return False
        return datetime.now() <= self._announce_faces_until

    def pop_announce_faces(self) -> bool:
        """Return True if announcement was active and clear the flag (one-time)."""
        active = self.should_announce_faces()
        self._announce_faces_until = None
        if active:
            logger.debug("Face announcement flag consumed; will not repeat")
        return active

    # ========== Object Detection ==========

    def observe_object(
        self,
        object_type: str,
        confidence: float,
        location_description: Optional[str] = None
    ) -> ObjectObservation:
        """
        Record object detection in working memory.
        
        Args:
            object_type: Class name (e.g., "fridge", "desk")
            confidence: Detection confidence (0.0-1.0)
            location_description: Optional location text
        
        Returns:
            ObjectObservation instance
        """
        now = datetime.now()
        
        if object_type in self.current_objects:
            # Update existing
            obs = self.current_objects[object_type]
            obs.last_detected_session = now
            obs.detection_count += 1
            obs.confidence = max(obs.confidence, confidence)  # Keep highest confidence
            if location_description:
                obs.location_description = location_description
        else:
            # Create new
            obs = ObjectObservation(
                object_type=object_type,
                confidence=confidence,
                first_detected_session=now,
                last_detected_session=now,
                detection_count=1,
                location_description=location_description
            )
            self.current_objects[object_type] = obs
        
        logger.debug(f"Object observed: {object_type} ({confidence:.2f})")
        return obs
    
    def get_stable_objects(self, min_detections: int = 3) -> List[ObjectObservation]:
        """
        Get objects that have been consistently detected.
        
        Args:
            min_detections: Minimum detection count to be considered stable
        
        Returns:
            List of ObjectObservation instances
        """
        return [
            obs for obs in self.current_objects.values()
            if obs.detection_count >= min_detections
        ]
    
    # ========== Room Context ==========
    
    def set_room(self, room_name: str, room_id: Optional[str] = None):
        """
        Update current room context.
        
        Args:
            room_name: Human-readable room name
            room_id: Optional room UUID from database
        """
        if self.current_room != room_name:
            logger.info(f"Room changed: {self.current_room or 'Unknown'} → {room_name}")
            self.current_room = room_name
            self.current_room_id = room_id
    
    def get_room_context(self) -> Dict[str, Any]:
        """
        Get current room context including detected objects.
        
        Returns:
            Dict with room_name, room_id, object_types
        """
        return {
            'room_name': self.current_room,
            'room_id': self.current_room_id,
            'object_types': list(self.current_objects.keys()),
            'stable_objects': [obs.object_type for obs in self.get_stable_objects()]
        }
    
    def update_mood(self, delta: int, event_type: str):
        """
        Update current mood based on event.
        
        Phase 3: If MoodEngine is integrated, updates emotion drivers
        instead of direct mood changes for more realistic decay.
        
        Args:
            delta: Mood change (-100 to +100)
            event_type: Event that caused change (for logging)
        """
        old_mood = self.current_mood
        
        # Phase 3: Use MoodEngine if available
        if self.mood_engine:
            # Map event types to emotion drivers
            driver_map = {
                'face_recognized': ('satisfaction', delta * 0.5),
                'face_new': ('curiosity', delta * 0.6),
                'object_detected': ('curiosity', delta * 0.4),
                'room_transition': ('curiosity', delta * 0.5),
                'pet_head': ('satisfaction', delta * 0.8),
                'task_complete': ('satisfaction', delta * 0.7),
                'task_failed': ('frustration', abs(delta) * 0.6),
                'timeout': ('frustration', abs(delta) * 0.4),
                'learning': ('curiosity', delta * 0.5),
            }
            
            if event_type in driver_map:
                driver, amount = driver_map[event_type]
                self.mood_engine.update_driver(driver, amount)
                self.current_mood = int(self.mood_engine.calculate_mood())
            else:
                # Fallback to satisfaction driver
                self.mood_engine.update_driver('satisfaction', delta * 0.5)
                self.current_mood = int(self.mood_engine.calculate_mood())
        else:
            # Simple mood update (Phase 1 behavior)
            self.current_mood = max(0, min(100, self.current_mood + delta))
        
        # Record in emotion history
        entry = EmotionHistoryEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            mood_delta=delta,
            new_mood=self.current_mood
        )
        self.emotion_history.append(entry)
        
        # Trim history if too long
        if len(self.emotion_history) > self.max_emotion_history:
            self.emotion_history = self.emotion_history[-self.max_emotion_history:]
        
        if delta != 0:
            logger.info(f"Mood update: {old_mood} → {self.current_mood} ({delta:+d}) [event: {event_type}]")
            self.emotion_history = self.emotion_history[-self.max_emotion_history:]
        
        if delta != 0:
            logger.info(f"Mood update: {old_mood} → {self.current_mood} ({delta:+d}) [event: {event_type}]")
    
    def get_mood(self) -> int:
        """Get current mood (0-100)."""
        return self.current_mood
    
    def get_recent_emotion_events(self, max_events: int = 10) -> List[EmotionHistoryEntry]:
        """Get recent emotion events from history."""
        return self.emotion_history[-max_events:]
    
    def get_mood_trend(self, lookback_events: int = 20) -> float:
        """
        Calculate mood trend over recent events.
        
        Args:
            lookback_events: Number of recent events to analyze
        
        Returns:
            Average mood delta per event (negative = declining mood)
        """
        if not self.emotion_history:
            return 0.0
        
        recent = self.emotion_history[-lookback_events:]
        if len(recent) < 2:
            return 0.0
        
        total_delta = sum(event.mood_delta for event in recent)
        return total_delta / len(recent)
    
    # ========== Task State Management (Principle IV) ==========
    
    def set_task_state(self, new_state: TaskState):
        """
        Update task state.
        
        Args:
            new_state: New TaskState value
        """
        if self.current_task != new_state:
            logger.info(f"Task state: {self.current_task.value} → {new_state.value}")
            self.current_task = new_state
    
    def get_task_state(self) -> TaskState:
        """Get current task state."""
        return self.current_task
    
    def is_available_for_interaction(self) -> bool:
        """Check if Vector is available for user interaction."""
        return self.current_task in [TaskState.IDLE, TaskState.LISTENING, TaskState.EXPLORING]
    
    # ========== Speech Buffer (Phase 2) ==========
    
    def add_speech(self, text: str):
        """
        Add speech text to recent buffer.
        
        Args:
            text: Transcribed speech text
        """
        self.recent_speech_buffer.append(text)
        
        # Trim if too long
        if len(self.recent_speech_buffer) > self.max_speech_buffer:
            self.recent_speech_buffer = self.recent_speech_buffer[-self.max_speech_buffer:]
    
    def get_recent_speech(self, max_count: int = 5) -> List[str]:
        """Get recent speech buffer (most recent first)."""
        return list(reversed(self.recent_speech_buffer[-max_count:]))
    
    # ========== Context Flags ==========
    
    def update_physical_context(
        self,
        is_being_held: Optional[bool] = None,
        is_on_charger: Optional[bool] = None,
        was_touched: bool = False
    ):
        """
        Update physical context flags.
        
        Args:
            is_being_held: Whether Vector is being held
            is_on_charger: Whether Vector is on charger
            was_touched: Whether Vector was just touched
        """
        if is_being_held is not None:
            self.is_being_held = is_being_held
        
        if is_on_charger is not None:
            self.is_on_charger = is_on_charger
        
        if was_touched:
            self.last_touch_time = datetime.now()
            # Touch triggers positive mood
            self.update_mood(15, event_type='pet_head')
    
    def get_physical_context(self) -> Dict[str, Any]:
        """Get current physical context."""
        return {
            'is_being_held': self.is_being_held,
            'is_on_charger': self.is_on_charger,
            'last_touch_time': self.last_touch_time,
            'seconds_since_touch': (datetime.now() - self.last_touch_time).total_seconds() if self.last_touch_time else None
        }
    
    # ========== Session Management ==========
    
    def clear_observations(self):
        """Clear all observations (faces, objects)."""
        self.current_faces.clear()
        self.current_objects.clear()
        self.current_room = None
        self.current_room_id = None
        logger.info("Observations cleared")
    
    def clear_emotion_history(self):
        """Clear emotion history."""
        self.emotion_history.clear()
        logger.info("Emotion history cleared")
    
    def reset_for_new_session(self, initial_mood: int = 50):
        """
        Full reset for new session (Vector turned on or woke from sleep).
        
        Args:
            initial_mood: Starting mood for new session
        """
        self.current_mood = initial_mood
        self.current_task = TaskState.IDLE
        self.session_start = datetime.now()
        
        self.current_faces.clear()
        self.current_objects.clear()
        self.current_room = None
        self.current_room_id = None
        
        self.emotion_history.clear()
        self.recent_speech_buffer.clear()
        
        self.is_being_held = False
        self.is_on_charger = False
        self.last_touch_time = None
        
        logger.info(f"Working memory reset for new session (mood: {initial_mood})")
    
    def get_recent_faces(self, limit: int = 5) -> List['FaceObservation']:
        """
        Get recently observed faces.
        
        Phase 4: Added for ReasoningEngine context assembly.
        
        Args:
            limit: Max number of faces to return
        
        Returns:
            List of FaceObservation objects, most recent first
        """
        # Sort by last_seen_session timestamp (most recent first)
        sorted_faces = sorted(
            self.current_faces.values(),
            key=lambda f: f.last_seen_session,
            reverse=True
        )
        return sorted_faces[:limit]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.
        
        Returns:
            Dict with session statistics
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_start': self.session_start,
            'session_duration_seconds': session_duration,
            'current_mood': self.current_mood,
            'current_task': self.current_task.value,
            'faces_seen': len(self.current_faces),
            'objects_detected': len(self.current_objects),
            'current_room': self.current_room,
            'emotion_events': len(self.emotion_history),
            'mood_trend': self.get_mood_trend(),
            'speech_buffer_size': len(self.recent_speech_buffer),
            'unique_faces': len(self.current_faces),
            'unique_objects': len(set(obj.object_type for obj in self.current_objects.values()))
        }
