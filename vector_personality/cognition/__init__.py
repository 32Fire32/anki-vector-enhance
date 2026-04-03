"""
Cognition Module

Phase 4: Cognition & LLM Integration

This module handles intelligent response generation with LLM clients
and context-aware reasoning.

Components:
- GroqClient: Llama 3.3 70B (primary LLM, privacy-focused)
- OpenAIClient: GPT-4 Turbo (fallback LLM), Whisper, TTS
- ResponseGenerator: Personality-aware prompt construction
- ReasoningEngine: Context assembly from all sensors

Usage:
    from vector_personality.cognition import (
        create_openai_client,
        create_response_generator,
        create_reasoning_engine
    )
    
    # Initialize components
    client = create_openai_client(api_key)
    generator = create_response_generator(client, personality)
    reasoning = create_reasoning_engine(working_memory, db_connector)
    
    # Generate response
    context = await reasoning.assemble_context()
    response = await generator.generate_response(
        user_input="Hello!",
        context=context,
        mood=70
    )
"""

try:
    from .openai_client import (
        OpenAIClient,
        create_openai_client
    )
except ImportError:
    OpenAIClient = None
    create_openai_client = None

try:
    from .groq_client import (
        GroqClient,
        LLMClient
    )
except ImportError:
    GroqClient = None
    LLMClient = None

from .response_generator import (
    ResponseGenerator,
    create_response_generator
)

from .reasoning_engine import (
    ReasoningEngine,
    create_reasoning_engine
)

__all__ = [
    # Classes
    'OpenAIClient',
    'GroqClient',
    'LLMClient',
    'ResponseGenerator',
    'ReasoningEngine',
    
    # Factory functions
    'create_openai_client',
    'create_response_generator',
    'create_reasoning_engine',
]
