"""
Vector Personality Package
Enhanced personality system for Anki Vector robot.

Modules:
- core: Vector agent orchestrator, personality traits, configuration
- memory: SQL Server integration, working/long-term memory
- perception: Audio, speech, face, object detection
- cognition: OpenAI API, response budget, reasoning
- behavior: Task state machine, autonomy, expression
"""

__version__ = "1.0.0"
__author__ = "Vector Personality Project"

from vector_personality.core import (
    VectorAgent,
    PersonalityTraits,
    PersonalityModule,
)

__all__ = [
    'VectorAgent',
    'PersonalityTraits',
    'PersonalityModule',
]
