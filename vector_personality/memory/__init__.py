"""
Memory module package for Vector Personality Project.
Implements dual-tier memory architecture (Principle II).

Tier 1: Working Memory (session-scoped, fast access)
Tier 2: Persistent Memory (ChromaDB, survives restarts)
"""

from .working_memory import (
    WorkingMemory,
    TaskState,
    FaceObservation,
    ObjectObservation,
    EmotionHistoryEntry
)

from .chromadb_connector import ChromaDBConnector, initialize_database

# Backward compatibility alias
SQLServerConnector = ChromaDBConnector

__all__ = [
    'WorkingMemory',
    'TaskState',
    'FaceObservation',
    'ObjectObservation',
    'EmotionHistoryEntry',
    'ChromaDBConnector',
    'SQLServerConnector',
    'initialize_database'
]
