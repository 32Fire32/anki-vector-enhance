"""
Personality configuration defaults.
"""

# Default personality trait values
DEFAULT_PERSONALITY_TRAITS = {
    'curiosity': 0.7,      # Moderately curious
    'touchiness': 0.6,     # Medium sensitivity to touch
    'vitality': 0.8,       # High energy
    'friendliness': 0.7,   # Warm and engaging
    'courage': 0.5,        # Cautious but willing
    'sassiness': 0.3,      # Subtle sarcasm
}

# Room-specific behavior modifiers
ROOM_BEHAVIOR_ADJUSTMENTS = {
    'working_desk': {
        'quietness_multiplier': 0.9,        # Be quiet
        'curiosity_reduction': 0.3,         # Less autonomous questions
        'expressiveness_reduction': 0.4,    # Subdued animations
    },
    'kids_room': {
        'playfulness_boost': 0.9,           # More energetic
        'sassiness_boost': 0.2,             # More playful teasing
        'vocalization_increase': 0.3,       # More talkative
    },
    'bedroom': {
        'quietness_multiplier': 0.8,        # Quiet
        'vitality_reduction': 0.2,          # Lower energy suggestions
        'curiosity_reduction': 0.4,         # Fewer interruptions
    },
    'kitchen': {
        'friendliness_boost': 0.1,          # More chatty
        'curiosity_increase': 0.2,          # Interested in cooking
    },
    'living_room': {
        'friendliness_boost': 0.2,          # More social
        'sassiness_boost': 0.1,             # More playful
    },
}

# API cost tiers (in euros)
API_COST_TIERS = {
    'free': 0.0,              # SDK-only (greetings, simple status)
    'cheap': 0.002,           # Context-aware responses
    'moderate': 0.01,         # Complex reasoning, learning
    'expensive': 0.05,        # Novel problem-solving
}

# Emotion drivers (mood delta)
EMOTION_DRIVERS = {
    'face_recognized': 10,               # +10 mood when seeing familiar face
    'face_new': 5,                       # +5 mood for new face
    'petted': 10,                        # +10 to +30 (touchiness-dependent)
    'moved_roughly': -20,                # -20 to -50 (touchiness-dependent)
    'ignored_per_minute': -10,           # -10 per minute of silence/inattention
    'task_success': 15,                  # +15 when task completed
    'curiosity_satisfied': 20,           # +20 when question is answered
    'contradiction': -5,                 # -5 when user contradicts Vector
}

# Mood → eye color mapping
MOOD_TO_COLOR = {
    0: (255, 0, 0),           # Angry: pure red
    20: (255, 64, 0),         # Upset: red-orange
    40: (255, 128, 0),        # Grumpy: orange
    60: (255, 255, 0),        # Neutral: yellow
    80: (0, 255, 0),          # Content: green
    100: (0, 255, 255),       # Joyful: cyan
}

# Task state priority levels
TASK_PRIORITY = {
    'critical': 100,          # User direct command
    'high': 75,               # Learning user preference, emotional response
    'medium': 50,             # Exploration, curiosity
    'low': 25,                # Routine checks, status updates
}

# Default task state
DEFAULT_TASK_STATE = 'idle'

# Always-listening configuration
AUDIO_CONFIG = {
    'sample_rate': 16000,                # Hz
    'frame_duration_ms': 10,             # WebRTC VAD frame duration
    'silence_threshold': 0.02,           # VAD energy threshold
    'buffer_duration_sec': 10,           # Rolling window size
}

# Database configuration template
DATABASE_CONFIG_TEMPLATE = {
    'server': 'localhost',
    'database': 'vector_memory',
    'trusted_connection': True,
    'timeout': 30,
}
