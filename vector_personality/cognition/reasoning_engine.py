"""
Reasoning Engine Module

Assembles context from all sensors for intelligent responses.
Combines faces, objects, room, mood, and personality into coherent context.

Features:
- Context assembly from WorkingMemory
- Memory retrieval for relevant past interactions
- Curiosity-based question generation
- Priority ranking of observations
- Context compression for API efficiency

Phase 4 - Cognition & OpenAI Integration
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Assemble context and generate curiosity-driven questions.
    
    Combines data from:
    - WorkingMemory (faces, objects, room, mood)
    - PersonalityModule (traits)
    - Database (past conversations, interactions)
    
    Attributes:
        working_memory: WorkingMemory instance
        db_connector: Optional database connector
    """
    
    QUESTION_TEMPLATES = {
        "unknown_object": [
            "What is that {object}?",
            "I've never seen a {object} before. What does it do?",
            "That {object} looks interesting. Can you tell me about it?",
            "What's the {object} for?"
        ],
        "new_face": [
            "Who is this? I don't think we've met!",
            "I see a new face! What's your name?",
            "Hello! I haven't seen you before. Who are you?"
        ],
        "new_room": [
            "Where are we? I don't recognize this place.",
            "What room is this?",
            "This looks different. What do you call this room?"
        ],
        "boredom": [
            "Want to tell me something interesting?",
            "I'm curious - what are you up to?",
            "Anything exciting happening?",
            "What should we do next?"
        ]
    }
    
    def __init__(
        self,
        working_memory,
        db_connector: Optional[Any] = None
    ):
        """
        Initialize reasoning engine.
        
        Args:
            working_memory: WorkingMemory instance
            db_connector: Optional database connector
        """
        self.working_memory = working_memory
        self.db_connector = db_connector
        self._last_question_time: Optional[datetime] = None
        self._asked_about: set = set()  # Track what we've asked about
        
        logger.info("ReasoningEngine initialized")
    
    async def assemble_context(
        self,
        include_personality: bool = True,
        include_history: bool = False,
        max_objects: int = 10,
        max_faces: int = 5
    ) -> Dict[str, Any]:
        """
        Assemble complete context for GPT-4.
        
        Args:
            include_personality: Include personality traits
            include_history: Include past conversations
            max_objects: Max objects to include
            max_faces: Max faces to include
        
        Returns:
            Context dict with all relevant information
        """
        context = {}
        
        # Current mood
        context['mood'] = self.working_memory.current_mood
        
        # Current room
        context['room'] = self.working_memory.current_room
        context['room_id'] = self.working_memory.current_room_id
        
        # Recent faces (prioritize recent, high confidence)
        faces = self.working_memory.get_recent_faces(limit=max_faces)
        context['faces'] = [
            {
                'name': face.name,
                'face_id': face.face_id,
                'last_seen': face.last_seen_session.isoformat()
            }
            for face in faces
        ]
        
        # Recent objects (prioritize stable, high confidence)
        stable_objects = self.working_memory.get_stable_objects(min_detections=2)
        all_objects = list(self.working_memory.current_objects.values())
        
        # Combine and sort by confidence
        combined_objects = stable_objects + [
            obj for obj in all_objects if obj not in stable_objects
        ]
        combined_objects.sort(key=lambda x: x.confidence, reverse=True)
        
        context['objects'] = [
            {
                'type': obj.object_type,
                'confidence': obj.confidence,
                'location': obj.location_description,
                'observation_count': obj.detection_count
            }
            for obj in combined_objects[:max_objects]
        ]
        
        # Personality traits
        if include_personality and hasattr(self.working_memory, 'personality'):
            personality = self.working_memory.personality
            if personality:
                traits = personality.effective_traits
                context['personality'] = {
                    'curiosity': traits.curiosity,
                    'touchiness': traits.touchiness,
                    'vitality': traits.vitality,
                    'friendliness': traits.friendliness,
                    'courage': traits.courage,
                    'sassiness': traits.sassiness
                }
        
        # Conversation history
        if include_history and self.db_connector:
            try:
                # Get recent conversations (last 24 hours)
                recent_convos = await self._get_recent_conversations(hours=24, limit=5)
                if recent_convos:
                    context['recent_conversations'] = recent_convos
            except Exception as e:
                logger.warning(f"Could not retrieve conversation history: {e}")
        
        # Session summary
        summary = self.working_memory.get_session_summary()
        context['session'] = {
            'duration_seconds': summary.get('duration_seconds', 0),
            'unique_faces': summary.get('unique_faces', 0),
            'unique_objects': summary.get('unique_objects', 0)
        }
        
        logger.debug(f"Assembled context: {len(context['objects'])} objects, {len(context['faces'])} faces")
        return context
    
    async def generate_curiosity_question(
        self,
        min_interval_seconds: int = 30
    ) -> Optional[str]:
        """
        Generate curiosity-driven question based on observations.
        
        Args:
            min_interval_seconds: Min time between questions
        
        Returns:
            Question text or None
        """
        # Check cooldown
        now = datetime.now()
        if self._last_question_time:
            elapsed = (now - self._last_question_time).total_seconds()
            if elapsed < min_interval_seconds:
                return None
        
        # Priority 1: Unknown objects (high confidence, not asked about)
        unknown_objects = [
            obj for obj in self.working_memory.current_objects.values()
            if obj.confidence > 0.7 and obj.object_type not in self._asked_about
        ]
        
        if unknown_objects:
            obj = random.choice(unknown_objects)
            template = random.choice(self.QUESTION_TEMPLATES['unknown_object'])
            question = template.format(object=obj.object_type)
            
            self._asked_about.add(obj.object_type)
            self._last_question_time = now
            logger.info(f"Generated curiosity question about object: {obj.object_type}")
            return question
        
        # Priority 2: New faces (not greeted)
        new_faces = [
            face for face in self.working_memory.get_recent_faces(limit=3)
            if face.user_id and face.user_id not in self._asked_about
        ]
        
        if new_faces:
            face = new_faces[0]
            question = random.choice(self.QUESTION_TEMPLATES['new_face'])
            
            self._asked_about.add(face.user_id)
            self._last_question_time = now
            logger.info("Generated curiosity question about new face")
            return question
        
        # Priority 3: Unknown room
        if self.working_memory.current_room and self.working_memory.current_room not in self._asked_about:
            question = random.choice(self.QUESTION_TEMPLATES['new_room'])
            
            self._asked_about.add(self.working_memory.current_room)
            self._last_question_time = now
            logger.info("Generated curiosity question about room")
            return question
        
        # Priority 4: Boredom (no recent observations)
        if len(self.working_memory.current_objects) == 0:
            question = random.choice(self.QUESTION_TEMPLATES['boredom'])
            self._last_question_time = now
            logger.info("Generated boredom question")
            return question
        
        # No questions to ask
        return None
    
    async def should_ask_question(
        self,
        idle_time_seconds: float,
        curiosity_threshold: float = 0.6,
        min_idle_seconds: int = 60
    ) -> bool:
        """
        Determine if Vector should proactively ask a question.
        
        Args:
            idle_time_seconds: Time since last interaction
            curiosity_threshold: Personality curiosity threshold
            min_idle_seconds: Min idle time before asking
        
        Returns:
            True if should ask question
        """
        # Check idle time
        if idle_time_seconds < min_idle_seconds:
            return False
        
        # Check curiosity trait
        if hasattr(self.working_memory, 'personality'):
            personality = self.working_memory.personality
            if personality:
                traits = personality.effective_traits
                if traits.curiosity < curiosity_threshold:
                    return False
        
        # Check if there's something new to ask about
        has_new_stimuli = (
            len(self.working_memory.current_objects) > 0 or
            len(self.working_memory.get_recent_faces(limit=1)) > 0
        )
        
        if not has_new_stimuli:
            # Random chance to ask boredom question
            return random.random() < 0.3  # 30% chance
        
        return True
    
    def compress_context_for_api(
        self,
        context: Dict[str, Any],
        max_objects: int = 5,
        max_faces: int = 3
    ) -> Dict[str, Any]:
        """
        Compress context to reduce API token usage.
        
        Args:
            context: Full context dict
            max_objects: Max objects to keep
            max_faces: Max faces to keep
        
        Returns:
            Compressed context dict
        """
        compressed = {
            'mood': context.get('mood'),
            'room': context.get('room')
        }
        
        # Keep top objects
        if 'objects' in context:
            compressed['objects'] = context['objects'][:max_objects]
        
        # Keep top faces
        if 'faces' in context:
            compressed['faces'] = context['faces'][:max_faces]
        
        # Simplify personality (just top 2 traits)
        if 'personality' in context:
            traits = context['personality']
            sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
            compressed['personality'] = dict(sorted_traits[:2])
        
        return compressed
    
    async def _get_recent_conversations(
        self,
        hours: int = 24,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent conversations from database"""
        if not self.db_connector:
            return []
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            result = await self.db_connector.query(
                """
                SELECT TOP (?) 
                    user_text,
                    bot_response,
                    timestamp
                FROM conversations
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (limit, cutoff_time)
            )
            
            return [
                {
                    'user': row['user_text'],
                    'bot': row['bot_response'],
                    'time': row['timestamp'].isoformat()
                }
                for row in (result or [])
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving conversations: {e}")
            return []
    
    def reset_asked_about(self):
        """Reset tracking of asked-about items (e.g., new session)"""
        self._asked_about.clear()
        self._last_question_time = None
        logger.info("Reset curiosity tracking")
    
    def get_context_summary(self, context: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of context.
        
        Args:
            context: Context dict
        
        Returns:
            Summary text
        """
        parts = []
        
        # Mood
        mood = context.get('mood', 50)
        if mood >= 70:
            parts.append("Feeling good")
        elif mood >= 40:
            parts.append("Feeling okay")
        else:
            parts.append("Feeling down")
        
        # Room
        if context.get('room'):
            parts.append(f"in {context['room']}")
        
        # Objects
        obj_count = len(context.get('objects', []))
        if obj_count > 0:
            parts.append(f"seeing {obj_count} object(s)")
        
        # Faces
        face_count = len(context.get('faces', []))
        if face_count > 0:
            parts.append(f"with {face_count} person(s)")
        
        return ", ".join(parts) if parts else "no context"


# Factory function
def create_reasoning_engine(
    working_memory,
    db_connector: Optional[Any] = None
) -> ReasoningEngine:
    """
    Create and initialize ReasoningEngine.
    
    Args:
        working_memory: WorkingMemory instance
        db_connector: Optional database connector
    
    Returns:
        Initialized ReasoningEngine instance
    """
    return ReasoningEngine(
        working_memory=working_memory,
        db_connector=db_connector
    )
