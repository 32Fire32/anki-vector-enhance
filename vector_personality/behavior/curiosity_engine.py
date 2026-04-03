"""
Curiosity Engine: GPT-4-driven question generation

Generates context-aware curiosity questions based on Vector's observations,
personality traits, and conversation history.

Features:
- GPT-4 question generation about unknown objects/faces
- Topic prioritization (faces > new objects > known objects)
- Repetition avoidance (don't ask same question twice)
- Curiosity decay after learning (satisfaction)
- Context-aware questions (room, mood, personality)

Dependencies:
- WorkingMemory (Phase 1): Context retrieval
- PersonalityModule (Phase 1): Curiosity trait influence
- ReasoningEngine (Phase 4): Context assembly
- OpenAIClient (Phase 4): GPT-4 API
- SQLServerConnector (Phase 1): Question history tracking
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta


class CuriosityEngine:
    """
    Generates curiosity-driven questions using GPT-4.
    
    Features:
    - Topic selection (prioritize unknown entities)
    - Question generation with personality context
    - Repetition avoidance
    - Learning satisfaction tracking
    """
    
    def __init__(
        self,
        working_memory,
        personality_module,
        reasoning_engine,
        openai_client,
        db_connector,
        min_interval_seconds: int = 600,  # 10 minutes between questions (reduced spam)
        curiosity_decay_hours: int = 24  # Curiosity decays after 24 hours
    ):
        """
        Initialize CuriosityEngine
        
        Args:
            working_memory: WorkingMemory instance
            personality_module: PersonalityModule instance
            reasoning_engine: ReasoningEngine instance
            openai_client: OpenAIClient instance
            db_connector: SQLServerConnector instance
            min_interval_seconds: Minimum time between questions
            curiosity_decay_hours: Hours until curiosity fully decays
        """
        self.working_memory = working_memory
        self.personality = personality_module
        self.reasoning_engine = reasoning_engine
        self.openai_client = openai_client
        self.db_connector = db_connector
        
        # Configuration
        self.min_interval_seconds = min_interval_seconds
        self.curiosity_decay_hours = curiosity_decay_hours
        
        # State tracking
        self.last_question_time: Optional[datetime] = None
        self.curiosity_levels: Dict[str, float] = {}  # topic -> curiosity level (0.0-1.0)
        self.learned_topics: Dict[str, datetime] = {}  # topic -> learning time
    
    async def generate_curiosity_question(
        self,
        min_interval_seconds: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate a curiosity question about current context
        
        Args:
            min_interval_seconds: Override default interval
            
        Returns:
            Question string or None if too soon / no topics
        """
        # Check interval
        interval = min_interval_seconds or self.min_interval_seconds
        if self.last_question_time:
            time_since_last = (datetime.now() - self.last_question_time).total_seconds()
            if time_since_last < interval:
                return None
        
        # Select topic
        topic = await self.select_next_topic()
        if not topic:
            return None
        
        # Check if should ask
        if not await self.should_ask_about(topic):
            return None
        
        # Assemble context
        context = await self.reasoning_engine.assemble_context(
            include_personality=True,
            include_history=False
        )
        
        # Build prompt
        prompt = self._build_question_prompt(topic, context)
        
        # Generate question with GPT-4
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are Vector, a curious robot. Generate a short, natural question in Italian (max 15 words) about the given topic. Respond ONLY with the question, nothing else."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            question = await self.openai_client.chat_completion(
                messages=messages,
                max_tokens=50,
                temperature=0.8
            )
            
            # Clean up question
            question = question.strip().strip('"')
            
            # Update timing FIRST to prevent duplicate questions
            self.last_question_time = datetime.now()
            
            # Record question to database
            await self._record_question(topic, question)
            
            # Mark topic as recently asked to prevent immediate repeat
            topic_key = topic.get('key')
            if topic_key:
                # Add to learned topics with current time (will decay after curiosity_decay_hours)
                self.learned_topics[topic_key] = datetime.now()
            
            return question
            
        except Exception as e:
            # Fallback to template questions
            return self._get_fallback_question(topic)
    
    async def select_next_topic(self) -> Optional[Dict]:
        """
        Select next topic to ask about
        
        Priority:
        1. Unknown faces (high)
        2. New objects with high confidence (medium)
        3. Room context if unknown (low)
        
        Returns:
            Topic dict or None
        """
        # Priority 1: Unknown faces
        for face in self.working_memory.current_faces.values():
            # Skip faces without names (None)
            if face.name and (face.name.startswith("Unknown") or face.name == "stranger"):
                topic_key = f"face_{face.face_id}"
                if topic_key not in self.learned_topics:
                    return {
                        'type': 'face',
                        'name': face.name,
                        'face_id': face.face_id,
                        'priority': 'high',
                        'key': topic_key
                    }
        
        # Priority 2: New objects (high confidence, not learned)
        objects = self.working_memory.get_stable_objects(min_detections=2)
        for obj in objects:
            # Skip generic 'person' detections - too noisy and not interesting
            if obj.object_type == 'person':
                continue
                
            topic_key = f"object_{obj.object_type}"
            # Require higher confidence (0.85) and not recently asked
            if topic_key not in self.learned_topics and obj.confidence >= 0.85:
                return {
                    'type': 'object',
                    'name': obj.object_type,
                    'confidence': obj.confidence,
                    'location': obj.location_description,
                    'priority': 'medium',
                    'key': topic_key
                }
        
        # Priority 3: Unknown room - DISABLED (too annoying during normal operation)
        # User can ask Vector directly if they want room info
        # topic_key = "room_context"
        # if topic_key not in self.learned_topics:
        #     if self.working_memory.current_room is None or self.working_memory.current_room == "unknown":
        #         return {
        #             'type': 'room',
        #             'name': 'unknown',
        #             'priority': 'low',
        #             'key': topic_key
        #         }
        
        return None
    
    async def should_ask_about(self, topic: Dict) -> bool:
        """
        Determine if should ask about topic (checks history and cooldown)
        
        Args:
            topic: Topic dict from select_next_topic()
            
        Returns:
            True if should ask, False if too recent
        """
        topic_key = topic['key']
        
        # Check if recently learned
        if topic_key in self.learned_topics:
            learned_time = self.learned_topics[topic_key]
            hours_since = (datetime.now() - learned_time).total_seconds() / 3600
            
            # Still in decay period
            if hours_since < self.curiosity_decay_hours:
                return False
        
        # Check recent question history from database
        try:
            recent_questions = await self.db_connector.query("""
                SELECT TOP 10 topic, timestamp
                FROM curiosity_questions
                WHERE topic = ?
                ORDER BY timestamp DESC
            """, (topic_key,))
            
            if recent_questions:
                last_asked = recent_questions[0]['timestamp']
                minutes_since = (datetime.now() - last_asked).total_seconds() / 60
                
                # Asked within last hour
                if minutes_since < 60:
                    return False
        except Exception:
            pass  # Database error, continue anyway
        
        return True
    
    async def record_learning(self, topic_key: str, answer: str):
        """
        Record that Vector learned about a topic
        
        Args:
            topic_key: Topic identifier
            answer: Answer received
        """
        self.learned_topics[topic_key] = datetime.now()
        
        # Decay curiosity
        if topic_key in self.curiosity_levels:
            self.curiosity_levels[topic_key] *= 0.5  # 50% decay
        
        # Save to database
        try:
            await self.db_connector.execute("""
                INSERT INTO learning_events (timestamp, topic, answer, satisfaction_level)
                VALUES (GETDATE(), ?, ?, ?)
            """, (topic_key, answer, 0.8))  # High satisfaction
        except Exception:
            pass
    
    def get_curiosity_level(self, topic_key: str) -> float:
        """
        Get current curiosity level for a topic
        
        Args:
            topic_key: Topic identifier
            
        Returns:
            Curiosity level 0.0-1.0
        """
        if topic_key not in self.curiosity_levels:
            # Initial curiosity based on personality
            traits = self.personality.effective_traits
            self.curiosity_levels[topic_key] = traits.curiosity
        
        return self.curiosity_levels[topic_key]
    
    def calculate_question_frequency(self) -> float:
        """
        Calculate question frequency multiplier based on personality
        
        Returns:
            Multiplier (1.0 = baseline, >1.0 = more frequent)
        """
        traits = self.personality.effective_traits
        curiosity = traits.curiosity
        vitality = traits.vitality
        
        # High curiosity + vitality = more questions
        return 0.5 + (curiosity * 0.8) + (vitality * 0.3)
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _build_question_prompt(self, topic: Dict, context: Dict) -> str:
        """
        Build GPT-4 prompt for question generation
        
        Args:
            topic: Topic dict
            context: Context from reasoning engine
            
        Returns:
            Prompt string
        """
        prompt_parts = []
        
        # Topic info
        if topic['type'] == 'face':
            prompt_parts.append(f"I see an unknown person. ")
        elif topic['type'] == 'object':
            prompt_parts.append(f"I see a {topic['name']} (confidence: {topic.get('confidence', 0):.0%}). ")
        elif topic['type'] == 'room':
            prompt_parts.append("I'm in an unknown room. ")
        
        # Context
        if context.get('room') and context['room'] != 'unknown':
            prompt_parts.append(f"I'm in the {context['room']}. ")
        
        # Mood influence
        mood = context.get('mood', 50)
        if mood >= 70:
            prompt_parts.append("I'm feeling happy and curious! ")
        elif mood <= 30:
            prompt_parts.append("I'm feeling a bit down. ")
        
        # Personality
        traits = self.personality.effective_traits
        if traits.curiosity > 0.7:
            prompt_parts.append("I'm very curious about new things. ")
        
        prompt_parts.append("\n\nGenerate a short, friendly question about this.")
        
        return "".join(prompt_parts)
    
    def _get_fallback_question(self, topic: Dict) -> str:
        """
        Get template question as fallback (Italian)
        
        Args:
            topic: Topic dict
            
        Returns:
            Question string
        """
        if topic['type'] == 'face':
            return "Chi sei?"
        elif topic['type'] == 'object':
            return f"Cos'è quel {topic['name']}?"
        elif topic['type'] == 'room':
            return "Dove mi trovo?"
        else:
            return "Cos'è quello?"
    
    async def _record_question(self, topic: Dict, question: str):
        """
        Record question to database
        
        Args:
            topic: Topic dict
            question: Generated question
        """
        try:
            await self.db_connector.execute("""
                INSERT INTO curiosity_questions (timestamp, topic, question, mood)
                VALUES (GETDATE(), ?, ?, ?)
            """, (topic['key'], question, self.working_memory.current_mood))
        except Exception:
            pass  # Non-critical


# ============================================================================
# Factory Function
# ============================================================================

def create_curiosity_engine(
    working_memory,
    personality_module,
    reasoning_engine,
    openai_client,
    db_connector,
    min_interval_seconds: int = 120
):
    """
    Factory function to create CuriosityEngine
    
    Args:
        working_memory: WorkingMemory instance
        personality_module: PersonalityModule instance
        reasoning_engine: ReasoningEngine instance
        openai_client: OpenAIClient instance
        db_connector: SQLServerConnector instance
        min_interval_seconds: Minimum seconds between questions (default: 120)
        
    Returns:
        Configured CuriosityEngine instance
    """
    return CuriosityEngine(
        working_memory=working_memory,
        personality_module=personality_module,
        reasoning_engine=reasoning_engine,
        openai_client=openai_client,
        db_connector=db_connector,
        min_interval_seconds=min_interval_seconds
    )
