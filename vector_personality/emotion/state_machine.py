"""
State Machine Module

Manages TaskState transitions with validation and triggers.
Ensures only valid state transitions occur and tracks state history.

States:
- IDLE: Waiting for input
- LISTENING: Actively listening for speech
- PROCESSING: Processing input/generating response
- EXPLORING: Looking around, detecting faces/objects
- LEARNING: Processing new information
- PAUSED: Manually paused (can be entered from any state)

Transitions:
- IDLE → LISTENING (speech detected)
- IDLE → EXPLORING (face/object detected)
- LISTENING → PROCESSING (speech ended)
- PROCESSING → IDLE (response complete)
- EXPLORING → IDLE (exploration complete)
- LEARNING → IDLE (learning complete)
- Any → PAUSED (manual pause)
- PAUSED → previous state (resume)
"""

import logging
from typing import Optional, List, Callable, Dict
from datetime import datetime, timedelta
from vector_personality.memory.working_memory import TaskState

logger = logging.getLogger(__name__)


class StateMachine:
    """
    State machine for TaskState transitions
    
    Validates transitions and maintains state history.
    Supports callbacks on state changes and timeout handling.
    """

    # Valid transitions: {from_state: [to_state1, to_state2, ...]}
    VALID_TRANSITIONS = {
        TaskState.IDLE: [
            TaskState.LISTENING,
            TaskState.EXPLORING,
            TaskState.LEARNING,
            TaskState.PAUSED
        ],
        TaskState.LISTENING: [
            TaskState.PROCESSING,
            TaskState.IDLE,
            TaskState.PAUSED
        ],
        TaskState.PROCESSING: [
            TaskState.IDLE,
            TaskState.LEARNING,
            TaskState.PAUSED
        ],
        TaskState.EXPLORING: [
            TaskState.IDLE,
            TaskState.LEARNING,
            TaskState.PAUSED
        ],
        TaskState.LEARNING: [
            TaskState.IDLE,
            TaskState.PAUSED
        ],
        TaskState.PAUSED: [
            TaskState.IDLE,
            TaskState.LISTENING,
            TaskState.PROCESSING,
            TaskState.EXPLORING,
            TaskState.LEARNING
        ]
    }

    # Event → State mapping
    EVENT_TRANSITIONS = {
        "speech_detected": TaskState.LISTENING,
        "face_detected": TaskState.EXPLORING,
        "object_detected": TaskState.EXPLORING,
        "processing_started": TaskState.PROCESSING,
        "learning_started": TaskState.LEARNING,
        "timeout": TaskState.IDLE,
        "complete": TaskState.IDLE,
        "pause": TaskState.PAUSED
    }

    def __init__(self, initial_state: TaskState = TaskState.IDLE):
        """
        Initialize StateMachine
        
        Args:
            initial_state: Starting state (default: IDLE)
        """
        self.current_state = initial_state
        self.previous_state: Optional[TaskState] = None
        self.state_before_pause: Optional[TaskState] = None
        
        # State history
        self.history: List[Dict] = []
        self._record_state_change(None, initial_state)
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # State durations
        self.state_start_time = datetime.now()
        self.state_durations: Dict[TaskState, float] = {}
        
        logger.info(f"StateMachine initialized in {initial_state.value} state")

    def transition(self, new_state: TaskState) -> bool:
        """
        Attempt to transition to new state
        
        Args:
            new_state: Desired state
        
        Returns:
            True if transition was allowed, False otherwise
        """
        # Check if transition is valid
        if not self._is_valid_transition(self.current_state, new_state):
            logger.warning(
                f"Invalid transition: {self.current_state.value} → {new_state.value}"
            )
            return False
        
        # Special handling for PAUSED state
        if new_state == TaskState.PAUSED:
            self.state_before_pause = self.current_state
        
        # Record duration of previous state
        self._record_state_duration()
        
        # Perform transition
        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state
        self.state_start_time = datetime.now()
        
        # Record in history
        self._record_state_change(old_state, new_state)
        
        # Fire callbacks
        self._fire_callbacks(old_state, new_state)
        
        logger.info(f"State transition: {old_state.value} → {new_state.value}")
        return True

    def trigger(self, event: str) -> bool:
        """
        Trigger state transition based on event
        
        Args:
            event: Event name (speech_detected, face_detected, etc.)
        
        Returns:
            True if event caused a transition
        """
        if event not in self.EVENT_TRANSITIONS:
            logger.debug(f"Unknown event: {event}")
            return False
        
        target_state = self.EVENT_TRANSITIONS[event]
        
        # Special handling for resume from PAUSED
        if event == "resume" and self.current_state == TaskState.PAUSED:
            if self.state_before_pause:
                return self.transition(self.state_before_pause)
        
        return self.transition(target_state)

    def _is_valid_transition(
        self,
        from_state: TaskState,
        to_state: TaskState
    ) -> bool:
        """
        Check if transition is valid
        
        Args:
            from_state: Current state
            to_state: Desired state
        
        Returns:
            True if transition is allowed
        """
        if from_state not in self.VALID_TRANSITIONS:
            return False
        
        return to_state in self.VALID_TRANSITIONS[from_state]

    def get_allowed_transitions(self) -> List[TaskState]:
        """
        Get list of allowed next states
        
        Returns:
            List of TaskState values that can be transitioned to
        """
        return self.VALID_TRANSITIONS.get(self.current_state, [])

    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for state transitions
        
        Callback signature: callback(from_state, to_state)
        
        Args:
            callback: Function to call on transitions
        """
        self.callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")

    def _fire_callbacks(
        self,
        from_state: TaskState,
        to_state: TaskState
    ) -> None:
        """Fire all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(from_state, to_state)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _record_state_change(
        self,
        from_state: Optional[TaskState],
        to_state: TaskState
    ) -> None:
        """Record state change in history"""
        self.history.append({
            "timestamp": datetime.now(),
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "duration": None  # Will be filled when state changes
        })

    def _record_state_duration(self) -> None:
        """Record duration of current state"""
        if not self.history:
            return
        
        duration = (datetime.now() - self.state_start_time).total_seconds()
        
        # Update last history entry
        self.history[-1]["duration"] = duration
        
        # Aggregate durations by state
        if self.current_state not in self.state_durations:
            self.state_durations[self.current_state] = 0.0
        
        self.state_durations[self.current_state] += duration

    def get_state_duration(self, state: TaskState) -> float:
        """
        Get total time spent in a state
        
        Args:
            state: TaskState to query
        
        Returns:
            Total seconds spent in that state
        """
        return self.state_durations.get(state, 0.0)

    def get_current_state_duration(self) -> float:
        """
        Get duration of current state
        
        Returns:
            Seconds in current state
        """
        return (datetime.now() - self.state_start_time).total_seconds()

    def get_transition_count(self) -> int:
        """
        Get total number of state transitions
        
        Returns:
            Transition count
        """
        return len(self.history) - 1  # Subtract initial state

    def get_recent_history(self, count: int = 10) -> List[Dict]:
        """
        Get recent state history
        
        Args:
            count: Number of recent entries
        
        Returns:
            List of history dictionaries
        """
        return self.history[-count:]

    def reset(self) -> None:
        """Reset to IDLE state and clear history"""
        self.current_state = TaskState.IDLE
        self.previous_state = None
        self.state_before_pause = None
        self.history = []
        self.state_durations = {}
        self.state_start_time = datetime.now()
        
        self._record_state_change(None, TaskState.IDLE)
        
        logger.info("StateMachine reset to IDLE")

    def __repr__(self) -> str:
        """String representation"""
        duration = self.get_current_state_duration()
        return (
            f"StateMachine(current={self.current_state.value}, "
            f"duration={duration:.1f}s, "
            f"transitions={self.get_transition_count()})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_state_machine(config: dict = None) -> StateMachine:
    """
    Factory function to create StateMachine with config
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured StateMachine instance
    """
    initial_state = TaskState.IDLE
    
    if config and "initial_state" in config:
        state_name = config["initial_state"]
        initial_state = TaskState[state_name.upper()]
    
    return StateMachine(initial_state)


def state_to_emoji(state: TaskState) -> str:
    """
    Map state to emoji for display
    
    Args:
        state: TaskState value
    
    Returns:
        Emoji string
    """
    emoji_map = {
        TaskState.IDLE: "😴",
        TaskState.LISTENING: "👂",
        TaskState.PROCESSING: "🤔",
        TaskState.EXPLORING: "👀",
        TaskState.LEARNING: "📚",
        TaskState.PAUSED: "⏸️"
    }
    return emoji_map.get(state, "❓")


def visualize_state_flow(history: List[Dict]) -> str:
    """
    Create ASCII visualization of state flow
    
    Args:
        history: State history list
    
    Returns:
        ASCII diagram string
    """
    if not history:
        return "No state history"
    
    lines = ["State Flow:", "=" * 50]
    
    for entry in history:
        from_state = entry["from_state"] or "START"
        to_state = entry["to_state"]
        duration = entry["duration"]
        
        if duration is not None:
            duration_str = f" ({duration:.1f}s)"
        else:
            duration_str = ""
        
        lines.append(f"{from_state} → {to_state}{duration_str}")
    
    return "\n".join(lines)


def get_state_statistics(sm: StateMachine) -> Dict:
    """
    Get statistics about state machine usage
    
    Args:
        sm: StateMachine instance
    
    Returns:
        Dictionary with statistics
    """
    total_time = sum(sm.state_durations.values())
    
    percentages = {}
    for state, duration in sm.state_durations.items():
        if total_time > 0:
            percentages[state.value] = (duration / total_time) * 100
        else:
            percentages[state.value] = 0.0
    
    return {
        "current_state": sm.current_state.value,
        "total_transitions": sm.get_transition_count(),
        "current_duration": sm.get_current_state_duration(),
        "state_durations": {k.value: v for k, v in sm.state_durations.items()},
        "state_percentages": percentages
    }
