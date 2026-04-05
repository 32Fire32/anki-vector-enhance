"""
Task Manager: State machine and priority queue for autonomous behavior

Manages Vector's task execution, state transitions, and scheduling.

Features:
- State machine integration (IDLE, LISTENING, PROCESSING, EXPLORING, LEARNING, PAUSED)
- Priority queue for task scheduling (CRITICAL > HIGH > MEDIUM > LOW)
- Idle timeout triggers (automatic exploration)
- Manual override capability (user interaction interrupts tasks)
- Task cooldown periods (prevent repetitive actions)
- State persistence across pauses

Dependencies:
- StateMachine (Phase 3): Manages state transitions
- WorkingMemory (Phase 1): Task state tracking
- PersonalityModule (Phase 1): Influences task priorities
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import heapq

from vector_personality.memory.working_memory import TaskState

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels (higher number = higher priority)"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task representation with priority ordering"""
    priority: int
    name: str
    callback: Callable
    created_at: datetime = field(default_factory=datetime.now)
    task_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Higher priority comes first (max heap behavior)"""
        return self.priority > other.priority


class TaskManager:
    """
    Manages task execution and state transitions for autonomous behavior.
    
    Features:
    - Priority-based task queue
    - State machine integration
    - Idle timeout detection
    - Task cooldown management
    - Manual override handling
    """
    
    def __init__(
        self,
        working_memory,
        personality_module,
        state_machine,
        db_connector,
        idle_timeout_seconds: int = 60  # 1 minute default
    ):
        """
        Initialize TaskManager
        
        Args:
            working_memory: WorkingMemory instance
            personality_module: PersonalityModule instance
            state_machine: StateMachine instance
            db_connector: SQLServerConnector instance
            idle_timeout_seconds: Seconds of idle before auto-exploration
        """
        self.working_memory = working_memory
        self.personality = personality_module
        self.state_machine = state_machine
        self.db_connector = db_connector
        
        # Configuration
        self.idle_timeout_seconds = idle_timeout_seconds
        
        # State
        self.current_state = TaskState.IDLE
        self.task_queue: List[Task] = []
        self.current_task: Optional[Task] = None
        self.paused_state: Optional[TaskState] = None
        
        # Timing
        self.last_activity_time = datetime.now()
        self.last_state_transition = datetime.now()
        
        # Cooldown tracking (task_type -> last_execution_time)
        self.task_cooldowns: Dict[str, datetime] = {}
        
        # Default cooldown periods (seconds)
        self.cooldown_periods = {
            'curiosity_question': 120,  # 2 minutes
            'exploration': 90,  # 1.5 minutes
            'learning': 60,  # 1 minute
            'greeting': 30  # 30 seconds
        }
        
        # Valid state transitions
        self.valid_transitions = {
            TaskState.IDLE: [TaskState.LISTENING, TaskState.EXPLORING, TaskState.PAUSED],
            TaskState.LISTENING: [TaskState.PROCESSING, TaskState.IDLE, TaskState.PAUSED],
            TaskState.PROCESSING: [TaskState.IDLE, TaskState.LEARNING, TaskState.PAUSED],
            TaskState.EXPLORING: [TaskState.IDLE, TaskState.LEARNING, TaskState.LISTENING, TaskState.PROCESSING, TaskState.PAUSED],
            TaskState.LEARNING: [TaskState.IDLE, TaskState.PROCESSING, TaskState.PAUSED],
            TaskState.PAUSED: [TaskState.IDLE, TaskState.LISTENING, TaskState.EXPLORING, TaskState.LEARNING]
        }
    
    async def transition_to(self, new_state: TaskState) -> bool:
        """
        Attempt state transition (validates against allowed transitions)
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition successful, False if invalid
        """
        if new_state in self.valid_transitions[self.current_state]:
            old_state = self.current_state
            self.current_state = new_state
            self.last_state_transition = datetime.now()
            
            # Update state machine (Phase 3 integration)
            if hasattr(self.state_machine, 'current_state'):
                # Map TaskState to StateMachine states if needed
                self.state_machine.current_state = new_state.value
            
            logger.info(f"📋 State transition: {old_state.value} → {new_state.value}")
            
            return True
        else:
            logger.warning(f"📋 Invalid state transition: {self.current_state.value} → {new_state.value} (not allowed)")
        return False
    
    async def add_task(
        self,
        name: str,
        priority: TaskPriority,
        callback: Callable,
        task_type: str = "generic",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add task to priority queue
        
        Args:
            name: Task name
            priority: TaskPriority enum
            callback: Async function to execute
            task_type: Task category for cooldown tracking
            metadata: Additional task data
            
        Returns:
            True if added, False if blocked by cooldown
        """
        # Check cooldown
        if task_type in self.task_cooldowns:
            last_execution = self.task_cooldowns[task_type]
            cooldown_period = self.cooldown_periods.get(task_type, 60)
            if (datetime.now() - last_execution).total_seconds() < cooldown_period:
                return False
        
        # Create task
        task = Task(
            priority=priority.value,
            name=name,
            callback=callback,
            task_type=task_type,
            metadata=metadata or {}
        )
        
        # Add to priority queue (heapq maintains min heap, but Task.__lt__ inverts for max heap)
        heapq.heappush(self.task_queue, task)
        
        return True
    
    async def schedule_task(
        self,
        name: str,
        priority: TaskPriority,
        callback: Callable,
        task_type: str = "generic"
    ) -> bool:
        """
        Schedule task (alias for add_task with cooldown check)
        
        Returns:
            True if scheduled, False if blocked by cooldown
        """
        return await self.add_task(name, priority, callback, task_type)
    
    async def get_next_task(self) -> Optional[Dict]:
        """
        Get highest priority task from queue
        
        Returns:
            Task dict or None if queue empty
        """
        if not self.task_queue:
            return None
        
        task = heapq.heappop(self.task_queue)
        
        return {
            'name': task.name,
            'priority': TaskPriority(task.priority),
            'callback': task.callback,
            'task_type': task.task_type,
            'metadata': task.metadata,
            'created_at': task.created_at
        }
    
    async def execute_next_task(self) -> Optional[Any]:
        """
        Execute highest priority task
        
        Returns:
            Task result or None if queue empty
        """
        task_dict = await self.get_next_task()
        if not task_dict:
            return None
        
        self.current_task = task_dict
        self.last_activity_time = datetime.now()
        
        try:
            # Execute callback
            if asyncio.iscoroutinefunction(task_dict['callback']):
                result = await task_dict['callback']()
            else:
                result = task_dict['callback']()
            
            # Update cooldown
            self.task_cooldowns[task_dict['task_type']] = datetime.now()
            
            return result
            
        except Exception as e:
            # Log error but don't crash
            await self._log_error(f"Task execution failed: {task_dict['name']}", e)
            return None
        finally:
            self.current_task = None
    
    async def manual_override(self, reason: str) -> bool:
        """
        Interrupt current task due to manual user interaction
        
        Args:
            reason: Override reason (e.g., "user_interaction", "picked_up")
            
        Returns:
            True if interrupted, False if already idle
        """
        if self.current_state == TaskState.IDLE:
            return False
        
        # Save current state for potential resume
        self.paused_state = self.current_state
        
        # Transition to PAUSED
        await self.transition_to(TaskState.PAUSED)
        
        # Log override
        await self._log_manual_override(reason)
        
        return True
    
    async def check_idle_timeout(self) -> bool:
        """
        Check if idle timeout exceeded (triggers auto-exploration)
        
        Returns:
            True if timeout exceeded
        """
        if self.current_state != TaskState.IDLE:
            return False
        
        idle_duration = (datetime.now() - self.last_activity_time).total_seconds()
        return idle_duration >= self.idle_timeout_seconds
    
    async def trigger_exploration_if_idle(self):
        """
        Add exploration task if idle timeout exceeded
        """
        if await self.check_idle_timeout():
            # Add exploration task (medium priority)
            await self.add_task(
                name="auto_exploration",
                priority=TaskPriority.MEDIUM,
                callback=self._exploration_callback,
                task_type="exploration"
            )
    
    async def resume_from_pause(self) -> bool:
        """
        Resume from PAUSED state to previous state
        
        Returns:
            True if resumed, False if not paused
        """
        if self.current_state != TaskState.PAUSED:
            return False
        
        if self.paused_state:
            await self.transition_to(self.paused_state)
            self.paused_state = None
            return True
        else:
            # No saved state, go to IDLE
            await self.transition_to(TaskState.IDLE)
            return True
    
    async def clear_queue(self):
        """Clear all pending tasks"""
        self.task_queue = []
    
    async def get_queue_status(self) -> Dict:
        """
        Get current queue status
        
        Returns:
            Dict with queue stats
        """
        return {
            'current_state': self.current_state.value,
            'queue_length': len(self.task_queue),
            'current_task': self.current_task['name'] if self.current_task else None,
            'idle_seconds': (datetime.now() - self.last_activity_time).total_seconds(),
            'paused_state': self.paused_state.value if self.paused_state else None
        }
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    async def _exploration_callback(self):
        """Default exploration callback (auto-triggered after idle timeout)."""
        # Transition to EXPLORING state
        await self.transition_to(TaskState.EXPLORING)
        # Note: the actual driving/turning is handled by autonomy_controller's
        # _exploration_callback.  This task_manager callback is the one used by
        # trigger_exploration_if_idle() — it only sets the state.  When the
        # autonomy_controller finishes, it transitions back to IDLE.
        return "exploration_started"
    
    async def _log_state_transition(self, old_state: TaskState, new_state: TaskState):
        """Log state transition to database"""
        try:
            await self.db_connector.execute("""
                INSERT INTO state_transitions (timestamp, old_state, new_state, mood)
                VALUES (GETDATE(), ?, ?, ?)
            """, (old_state.value, new_state.value, self.working_memory.current_mood))
        except Exception:
            pass  # Non-critical, don't crash
    
    async def _log_manual_override(self, reason: str):
        """Log manual override event"""
        try:
            await self.db_connector.execute("""
                INSERT INTO manual_overrides (timestamp, reason, previous_state)
                VALUES (GETDATE(), ?, ?)
            """, (reason, self.paused_state.value if self.paused_state else None))
        except Exception:
            pass
    
    async def _log_error(self, message: str, error: Exception):
        """Log error to database"""
        try:
            await self.db_connector.execute("""
                INSERT INTO system_errors (timestamp, message, error_details)
                VALUES (GETDATE(), ?, ?)
            """, (message, str(error)))
        except Exception:
            pass


# ============================================================================
# Factory Function
# ============================================================================

def create_task_manager(
    working_memory,
    personality_module,
    state_machine,
    db_connector,
    idle_timeout_seconds: int = 60
) -> TaskManager:
    """
    Factory function to create TaskManager instance
    
    Args:
        working_memory: WorkingMemory instance
        personality_module: PersonalityModule instance
        state_machine: StateMachine instance (Phase 3)
        db_connector: SQLServerConnector instance
        idle_timeout_seconds: Idle timeout (default: 60s = 1 minute)
        
    Returns:
        Configured TaskManager instance
    """
    return TaskManager(
        working_memory=working_memory,
        personality_module=personality_module,
        state_machine=state_machine,
        db_connector=db_connector,
        idle_timeout_seconds=idle_timeout_seconds
    )
