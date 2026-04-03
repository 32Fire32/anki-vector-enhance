"""
Behavior Module: Autonomous task management and curiosity-driven exploration

This module provides:
- TaskManager: State machine and priority queue for task execution
- AutonomyController: Initiative system and exploration triggers
- CuriosityEngine: GPT-4-driven question generation

Phase 5: Autonomous Behavior
"""

from vector_personality.behavior.task_manager import (
    TaskManager,
    TaskState,
    TaskPriority,
    create_task_manager
)

from vector_personality.behavior.autonomy_controller import (
    AutonomyController,
    create_autonomy_controller
)

from vector_personality.behavior.curiosity_engine import (
    CuriosityEngine,
    create_curiosity_engine
)

from vector_personality.behavior.startup_controller import (
    StartupController
)

__all__ = [
    # Classes
    'TaskManager',
    'TaskState',
    'TaskPriority',
    'AutonomyController',
    'CuriosityEngine',
    'StartupController',
    
    # Factory functions
    'create_task_manager',
    'create_autonomy_controller',
    'create_curiosity_engine',
]
