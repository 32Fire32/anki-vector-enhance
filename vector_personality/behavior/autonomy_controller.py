"""
Autonomy Controller: Initiative system and exploration triggers

Manages Vector's autonomous decision-making, including when to initiate
interactions, explore the environment, and focus attention.

Features:
- Boredom detection → curiosity-driven question generation
- Exploration triggers (no face seen, unknown room)
- Attention management (prioritize faces over objects)
- Learning triggers (new objects → "what is this?")
- Personality-based initiative probability

Dependencies:
- WorkingMemory (Phase 1): Context awareness
- PersonalityModule (Phase 1): Curiosity/courage influence
- ReasoningEngine (Phase 4): Context assembly
- TaskManager (Phase 5): Task scheduling
"""

import asyncio
import random
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class AutonomyController:
    """
    Controls Vector's autonomous behavior and initiative.
    
    Manages when Vector should:
    - Ask questions (boredom-driven curiosity)
    - Explore environment (no faces seen, unknown room)
    - Focus attention (faces vs objects)
    - Initiate learning (new object detection)
    """
    
    def __init__(
        self,
        working_memory,
        personality_module,
        reasoning_engine,
        task_manager,
        robot,
        boredom_threshold_seconds: int = 20,  # 20 seconds
        exploration_threshold_seconds: int = 20,  # 5 minutes
        learning_confidence_threshold: float = 0.7
    ):
        """
        Initialize AutonomyController
        
        Args:
            working_memory: WorkingMemory instance
            personality_module: PersonalityModule instance
            reasoning_engine: ReasoningEngine instance
            task_manager: TaskManager instance
            robot: Vector robot instance for movement commands
            boredom_threshold_seconds: Idle time before boredom kicks in
            exploration_threshold_seconds: No-face-seen time before exploration
            learning_confidence_threshold: Minimum confidence to trigger learning
        """
        self.working_memory = working_memory
        self.personality = personality_module
        self.reasoning_engine = reasoning_engine
        self.task_manager = task_manager
        self.robot = robot
        
        # Configuration
        self.boredom_threshold_seconds = boredom_threshold_seconds
        self.exploration_threshold_seconds = exploration_threshold_seconds
        self.learning_confidence_threshold = learning_confidence_threshold
        
        # State tracking
        self.last_activity_time = datetime.now()
        self.last_face_seen_time = datetime.now()
        self.last_question_time: Optional[datetime] = None
        self.last_exploration_time: Optional[datetime] = None
        
        # Attention tracking
        self.current_attention_target: Optional[Dict] = None
        self.attention_duration_seconds = 0
        
        # Asked topics (prevent repetition)
        self.asked_about: List[str] = []
    
    async def should_initiate_interaction(self) -> bool:
        """
        Determine if Vector should proactively start interaction
        
        Uses personality traits (curiosity, vitality) and boredom level
        to decide if Vector should ask a question or start conversation.
        
        Returns:
            True if should initiate, False otherwise
        """
        # Calculate boredom level
        idle_duration = (datetime.now() - self.last_activity_time).total_seconds()
        
        # Not bored yet
        if idle_duration < self.boredom_threshold_seconds:
            return False
        
        # Get personality traits
        traits = self.personality.effective_traits
        curiosity = traits.curiosity
        vitality = traits.vitality
        
        # Calculate initiative probability
        # Base probability increases with idle time
        boredom_factor = min(idle_duration / self.boredom_threshold_seconds, 3.0)  # Cap at 3x
        
        # Personality modifiers
        curiosity_modifier = curiosity * 0.5  # High curiosity = more likely
        vitality_modifier = vitality * 0.3  # High vitality = more energetic
        
        # Combined probability (0.0 to 1.0)
        probability = min(
            (boredom_factor * 0.2) + curiosity_modifier + vitality_modifier,
            0.95  # Never 100% certain
        )
        
        # Randomize based on probability
        return random.random() < probability
    
    async def initiate_interaction(self):
        """
        Start autonomous interaction (add curiosity question task)
        """
        from vector_personality.behavior.task_manager import TaskPriority
        
        # Add curiosity question task
        await self.task_manager.add_task(
            name="autonomy_curiosity_question",
            priority=TaskPriority.MEDIUM,
            callback=self._curiosity_question_callback,
            task_type="curiosity_question"
        )
        
        # Update timing
        self.last_question_time = datetime.now()
        self.last_activity_time = datetime.now()
    
    async def should_explore_environment(self) -> bool:
        """
        Determine if Vector should explore room
        
        Triggers exploration when:
        - No faces seen recently (feels lonely)
        - Room unknown or under-explored
        - High courage + curiosity traits
        
        Returns:
            True if should explore
        """
        # Check if face seen recently
        time_since_face = (datetime.now() - self.last_face_seen_time).total_seconds()
        no_faces_seen = time_since_face > self.exploration_threshold_seconds
        
        # Check current faces in memory
        current_faces = len(self.working_memory.current_faces)
        
        # Check room context
        room_unknown = self.working_memory.current_room is None or self.working_memory.current_room == "unknown"
        
        # Get personality traits
        traits = self.personality.effective_traits
        courage = traits.courage
        curiosity = traits.curiosity
        
        # Exploration likelihood
        exploration_factors = []
        
        if no_faces_seen or current_faces == 0:
            exploration_factors.append(0.4)  # Strong factor
        
        if room_unknown:
            exploration_factors.append(0.3)
        
        # Personality influence
        exploration_factors.append(courage * 0.2)
        exploration_factors.append(curiosity * 0.15)
        
        # Combined probability
        probability = min(sum(exploration_factors), 0.9)
        
        return random.random() < probability
    
    async def start_exploration(self):
        """
        Initiate room exploration
        """
        from vector_personality.behavior.task_manager import TaskPriority, TaskState
        
        # Add exploration task
        await self.task_manager.add_task(
            name="autonomy_exploration",
            priority=TaskPriority.MEDIUM,
            callback=self._exploration_callback,
            task_type="exploration"
        )
        
        # Transition to EXPLORING
        await self.task_manager.transition_to(TaskState.EXPLORING)
        
        # Update timing
        self.last_exploration_time = datetime.now()
        self.last_activity_time = datetime.now()
    
    async def complete_exploration(self, room_name: str, discovered_objects: List[str]):
        """
        Complete exploration and update memory
        
        Args:
            room_name: Inferred room name
            discovered_objects: List of object types found
        """
        # Update room in working memory
        self.working_memory.set_room(room_name)
        
        # Add discovered objects
        for obj in discovered_objects:
            # Note: Actual detection happens elsewhere, this is just bookkeeping
            pass
        
        # Return to IDLE
        from vector_personality.behavior.task_manager import TaskState
        await self.task_manager.transition_to(TaskState.IDLE)
    
    async def get_attention_target(self) -> Optional[Dict]:
        """
        Determine what Vector should focus attention on
        
        Priority order:
        1. Faces (social interaction priority)
        2. New/unknown objects (learning)
        3. Known objects (context)
        
        Returns:
            Dict with target info or None
        """
        # Priority 1: Faces
        if self.working_memory.current_faces:
            # Get most recently seen face
            face = max(
                self.working_memory.current_faces.values(),
                key=lambda f: f.last_seen_session
            )
            self.last_face_seen_time = datetime.now()
            
            return {
                'type': 'face',
                'name': face.name,
                'face_id': face.face_id,
                'priority': 'high'
            }
        
        # Priority 2: New/unknown objects (high confidence)
        objects = self.working_memory.get_stable_objects(min_detections=1)
        for obj in objects:
            if obj.confidence >= self.learning_confidence_threshold:
                # Check if already asked about
                if obj.object_type not in self.asked_about:
                    return {
                        'type': 'object',
                        'name': obj.object_type,
                        'confidence': obj.confidence,
                        'location': obj.location_description,
                        'priority': 'medium'
                    }
        
        # Priority 3: Known objects (context awareness)
        if objects:
            obj = objects[0]
            return {
                'type': 'object',
                'name': obj.object_type,
                'confidence': obj.confidence,
                'priority': 'low'
            }
        
        return None
    
    async def should_ask_about_object(self, object_type: str) -> bool:
        """
        Determine if should ask about an object
        
        Args:
            object_type: Type of object (e.g., "laptop", "plant")
            
        Returns:
            True if should ask (learning trigger)
        """
        # Already asked about?
        if object_type in self.asked_about:
            return False
        
        # Get object from memory
        objects = self.working_memory.get_stable_objects(min_detections=1)
        target_obj = None
        for obj in objects:
            if obj.object_type == object_type:
                target_obj = obj
                break
        
        if not target_obj:
            return False
        
        # Check confidence threshold
        if target_obj.confidence < self.learning_confidence_threshold:
            return False
        
        # Personality influence (curiosity)
        traits = self.personality.effective_traits
        curiosity = traits.curiosity
        
        # Higher curiosity = more likely to ask
        return random.random() < (0.5 + curiosity * 0.4)
    
    async def initiate_learning(self, topic: str):
        """
        Start learning about a topic (add learning task)
        
        Args:
            topic: Object type or topic to learn about
        """
        from vector_personality.behavior.task_manager import TaskPriority, TaskState
        
        # Mark as asked
        self.asked_about.append(topic)
        
        # Add learning task
        await self.task_manager.add_task(
            name=f"learn_about_{topic}",
            priority=TaskPriority.HIGH,
            callback=lambda: self._learning_callback(topic),
            task_type="learning",
            metadata={'topic': topic}
        )
        
        # Transition to LEARNING
        await self.task_manager.transition_to(TaskState.LEARNING)
        
        # Update timing
        self.last_activity_time = datetime.now()
    
    async def update_activity(self):
        """Update last activity timestamp (called on any interaction)"""
        self.last_activity_time = datetime.now()
    
    async def reset_asked_topics(self):
        """Clear asked topics list (e.g., new session)"""
        self.asked_about = []
    
    # ========================================================================
    # Private Callbacks
    # ========================================================================
    
    async def _curiosity_question_callback(self):
        """Callback for curiosity question task"""
        # This will be called by CuriosityEngine
        return "curiosity_question_triggered"
    
    async def _exploration_callback(self):
        """
        Callback for exploration task
        
        Makes Vector actively explore the environment:
        - Drive forward to explore new areas
        - Turn to scan different angles
        - Look around with head movements
        - Look for faces and objects
        """
        try:
            from anki_vector.util import distance_mm, speed_mmps, degrees
            import logging
            
            logger = logging.getLogger(__name__)
            logger.info("Starting exploration behavior")
            
            # Exploration sequence: drive → turn → look → repeat
            for _ in range(3):  # 3 exploration moves
                # 1. Drive forward a moderate distance
                drive_distance = random.uniform(200, 400)  # 200-400mm
                logger.debug(f"Exploration: driving {drive_distance:.0f}mm")
                
                # Run blocking SDK call in thread
                await asyncio.to_thread(
                    self.robot.behavior.drive_straight,
                    distance_mm(drive_distance),
                    speed_mmps(100),
                    should_play_anim=True
                )
                # await asyncio.sleep(2.5)  # No need to sleep if we await the blocking call
                
                # 2. Turn to scan a different angle
                turn_angle = random.uniform(-120, 120)  # Wide scan range
                logger.debug(f"Exploration: turning {turn_angle:.0f}°")
                
                # Run blocking SDK call in thread
                await asyncio.to_thread(
                    self.robot.behavior.turn_in_place,
                    degrees(turn_angle)
                )
                # await asyncio.sleep(2.0)  # No need to sleep if we await the blocking call
                
                # 3. Look around with head
                logger.debug("Exploration: scanning with head")
                for head_angle in [-15, 0, 15, 0]:  # Sweep head
                    await asyncio.to_thread(
                        self.robot.behavior.set_head_angle,
                        degrees(head_angle)
                    )
                    await asyncio.sleep(0.8)
            
            logger.info("Exploration behavior completed")
            return "exploration_completed"
            
        except Exception as e:
            logger.error(f"Exploration failed: {e}", exc_info=True)
            return "exploration_failed"
    
    async def _learning_callback(self, topic: str):
        """
        Callback for learning task
        
        Args:
            topic: Topic to learn about
        """
        # This triggers question generation about the topic
        return f"learning_about_{topic}"


# ============================================================================
# Factory Function
# ============================================================================

def create_autonomy_controller(
    working_memory,
    personality_module,
    reasoning_engine,
    task_manager,
    robot,
    boredom_threshold_seconds: int = 300,
    exploration_threshold_seconds: int = 300
):
    """
    Factory function to create AutonomyController
    
    Args:
        working_memory: WorkingMemory instance
        personality_module: PersonalityModule instance
        reasoning_engine: ReasoningEngine instance
        task_manager: TaskManager instance
        robot: Vector robot instance for movement commands
        boredom_threshold_seconds: Idle time before boredom (default: 300s)
        exploration_threshold_seconds: No-face time before exploration (default: 300s)
        
    Returns:
        Configured AutonomyController instance
    """
    return AutonomyController(
        working_memory=working_memory,
        personality_module=personality_module,
        reasoning_engine=reasoning_engine,
        task_manager=task_manager,
        robot=robot,
        boredom_threshold_seconds=boredom_threshold_seconds,
        exploration_threshold_seconds=exploration_threshold_seconds
    )
