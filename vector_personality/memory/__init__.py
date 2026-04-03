"""
Memory module package for Vector Personality Project.
Implements dual-tier memory architecture (Principle II).

Tier 1: Working Memory (session-scoped, fast access)
Tier 2: Persistent Memory (SQL Server, survives restarts)
"""

from .working_memory import (
    WorkingMemory,
    TaskState,
    FaceObservation,
    ObjectObservation,
    EmotionHistoryEntry
)

from .sql_server_connector import SQLServerConnector, initialize_database

__all__ = [
    'WorkingMemory',
    'TaskState',
    'FaceObservation',
    'ObjectObservation',
    'EmotionHistoryEntry',
    'SQLServerConnector',
    'initialize_database'
]
