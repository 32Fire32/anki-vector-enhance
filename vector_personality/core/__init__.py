"""
Vector Personality Core Module
Initialization and module exports.
"""

__version__ = "1.0.0"
__author__ = "Vector Personality Project"

from .personality import PersonalityTraits, PersonalityModule
from .vector_agent import VectorAgent

__all__ = [
    'PersonalityTraits',
    'PersonalityModule',
    'VectorAgent',
]
