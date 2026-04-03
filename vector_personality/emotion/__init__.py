"""
Emotion Module

Phase 3: Emotion Engine - Complete Implementation

Manages Vector's emotional state through:
- Mood calculation from 6 emotion drivers (MoodEngine)
- Visual feedback via eye colors (EyeColorMapper)
- State management and transitions (StateMachine)

Usage:
    from vector_personality.emotion import (
        MoodEngine,
        EyeColorMapper,
        StateMachine
    )
"""

# Core emotion modules
from .mood_engine import (
    MoodEngine,
    create_mood_engine,
    mood_to_descriptor,
    simulate_mood_cycle
)
from .eye_color_mapper import (
    EyeColorMapper,
    create_eye_color_mapper,
    rgb_to_hex,
    hex_to_rgb
)
from .state_machine import (
    StateMachine,
    create_state_machine,
    state_to_emoji,
    visualize_state_flow,
    get_state_statistics
)

__all__ = [
    # Mood Engine
    'MoodEngine',
    'create_mood_engine',
    'mood_to_descriptor',
    'simulate_mood_cycle',
    
    # Eye Color Mapper
    'EyeColorMapper',
    'create_eye_color_mapper',
    'rgb_to_hex',
    'hex_to_rgb',
    
    # State Machine
    'StateMachine',
    'create_state_machine',
    'state_to_emoji',
    'visualize_state_flow',
    'get_state_statistics',
]
