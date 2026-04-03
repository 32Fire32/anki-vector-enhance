"""
Phase 1 unit tests for Memory Foundation.
Tests SQL Server connectivity, working memory, and face detection integration.

Run with: pytest tests/test_phase1_memory.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from vector_personality.memory import (
    WorkingMemory,
    TaskState,
    SQLServerConnector,
    initialize_database
)


class TestWorkingMemory:
    """Test suite for WorkingMemory class."""
    
    def test_initialization(self):
        """Test working memory initializes correctly."""
        wm = WorkingMemory(initial_mood=60)
        assert wm.current_mood == 60
        assert wm.current_task == TaskState.IDLE
        assert len(wm.current_faces) == 0
        assert len(wm.current_objects) == 0
    
    def test_mood_update(self):
        """Test mood update with clamping."""
        wm = WorkingMemory(initial_mood=50)
        
        # Positive mood change
        wm.update_mood(30, 'test_event')
        assert wm.current_mood == 80
        
        # Negative mood change
        wm.update_mood(-40, 'test_event')
        assert wm.current_mood == 40
        
        # Clamp to max (100)
        wm.update_mood(100, 'test_event')
        assert wm.current_mood == 100
        
        # Clamp to min (0)
        wm.update_mood(-200, 'test_event')
        assert wm.current_mood == 0
    
    def test_face_observation(self):
        """Test face observation tracking."""
        wm = WorkingMemory()
        
        # Observe face first time
        obs1 = wm.observe_face('face-uuid-1', name='Alice', mood_impact=10)
        assert obs1.name == 'Alice'
        assert obs1.interaction_count == 1
        assert len(wm.current_faces) == 1
        
        # Observe same face again
        obs2 = wm.observe_face('face-uuid-1', name='Alice', mood_impact=5)
        assert obs2.interaction_count == 2
        assert len(wm.current_faces) == 1  # Still just 1 unique face
    
    def test_object_observation(self):
        """Test object detection tracking."""
        wm = WorkingMemory()
        
        # Detect object first time
        obs1 = wm.observe_object('fridge', confidence=0.92)
        assert obs1.object_type == 'fridge'
        assert obs1.detection_count == 1
        
        # Detect same object again
        obs2 = wm.observe_object('fridge', confidence=0.95)
        assert obs2.detection_count == 2
        assert obs2.confidence == 0.95  # Should keep highest confidence
    
    def test_stable_objects(self):
        """Test stable object filtering."""
        wm = WorkingMemory()
        
        # Detect objects with different counts
        wm.observe_object('desk', confidence=0.9)
        wm.observe_object('desk', confidence=0.91)
        wm.observe_object('chair', confidence=0.85)
        wm.observe_object('chair', confidence=0.86)
        wm.observe_object('chair', confidence=0.87)
        
        # Only 'chair' has 3+ detections
        stable = wm.get_stable_objects(min_detections=3)
        assert len(stable) == 1
        assert stable[0].object_type == 'chair'
    
    def test_mood_trend(self):
        """Test mood trend calculation."""
        wm = WorkingMemory(initial_mood=50)
        
        # Create upward trend
        for i in range(5):
            wm.update_mood(5, 'positive_event')
        
        trend = wm.get_mood_trend(lookback_events=5)
        assert trend > 0  # Positive trend
        
        # Create downward trend
        for i in range(5):
            wm.update_mood(-5, 'negative_event')
        
        trend = wm.get_mood_trend(lookback_events=5)
        assert trend < 0  # Negative trend
    
    def test_task_state_transitions(self):
        """Test task state management."""
        wm = WorkingMemory()
        
        assert wm.get_task_state() == TaskState.IDLE
        assert wm.is_available_for_interaction() is True
        
        wm.set_task_state(TaskState.PROCESSING)
        assert wm.get_task_state() == TaskState.PROCESSING
        assert wm.is_available_for_interaction() is False
        
        wm.set_task_state(TaskState.LISTENING)
        assert wm.is_available_for_interaction() is True
    
    def test_session_reset(self):
        """Test session reset clears all data."""
        wm = WorkingMemory(initial_mood=70)
        
        # Add some data
        wm.observe_face('face-1', name='Bob')
        wm.observe_object('laptop', confidence=0.9)
        wm.update_mood(20, 'test')
        wm.add_speech("Hello Vector!")
        
        # Reset
        wm.reset_for_new_session(initial_mood=50)
        
        # Verify everything cleared
        assert wm.current_mood == 50
        assert len(wm.current_faces) == 0
        assert len(wm.current_objects) == 0
        assert len(wm.emotion_history) == 0
        assert len(wm.recent_speech_buffer) == 0


@pytest.mark.asyncio
class TestSQLServerConnector:
    """Test suite for SQLServerConnector (requires SQL Server running)."""
    
    @pytest.fixture
    async def db_connector(self):
        """Fixture providing database connector."""
        connector = SQLServerConnector(server='localhost', database='vector_memory_test')
        
        # Create test database if not exists
        # NOTE: This requires pre-existing 'master' database access
        # In production, database should be created manually
        
        yield connector
        
        await connector.close()
    
    @pytest.mark.skip(reason="Requires SQL Server setup")
    async def test_connection(self, db_connector):
        """Test database connectivity."""
        result = await db_connector.test_connection()
        assert result is True
    
    @pytest.mark.skip(reason="Requires SQL Server setup")
    async def test_face_crud(self, db_connector):
        """Test face CRUD operations."""
        # Create face
        face_id = await db_connector.create_face(name='TestUser')
        assert face_id is not None
        
        # Read face
        face = await db_connector.get_face_by_id(face_id)
        assert face['name'] == 'TestUser'
        
        # Update face interaction
        success = await db_connector.update_face_interaction(face_id, mood_change=10)
        assert success is True
        
        # Verify interaction count incremented
        face = await db_connector.get_face_by_id(face_id)
        assert face['total_interactions'] == 1
    
    @pytest.mark.skip(reason="Requires SQL Server setup")
    async def test_conversation_storage(self, db_connector):
        """Test conversation storage and retrieval."""
        # Create face first
        face_id = await db_connector.create_face(name='TestSpeaker')
        
        # Store conversation
        conv_id = await db_connector.store_conversation(
            speaker_id=face_id,
            text="Hello Vector!",
            emotional_context=60,
            response_text="Hi there!",
            response_type='sdk'
        )
        assert conv_id is not None
        
        # Retrieve history
        history = await db_connector.get_face_history(face_id, limit=10)
        assert len(history) > 0
        assert history[0]['text'] == "Hello Vector!"
    
    @pytest.mark.skip(reason="Requires SQL Server setup")
    async def test_budget_tracking(self, db_connector):
        """Test budget tracking operations."""
        # Get today's budget
        budget = await db_connector.get_today_budget(vitality_level=0.8)
        assert budget is not None
        assert budget['vitality_level'] == 0.8
        
        # Update spending
        success = await db_connector.update_budget_spending(euros_spent=0.01, api_calls=1)
        assert success is True
        
        # Get weekly usage
        total_euros, total_calls = await db_connector.get_weekly_budget_usage()
        assert total_euros >= 0.01
        assert total_calls >= 1
    
    @pytest.mark.skip(reason="Requires SQL Server setup")
    async def test_personality_learning(self, db_connector):
        """Test personality adjustment storage."""
        # Save adjustment
        success = await db_connector.save_personality_adjustment(
            curiosity_delta=0.05,
            friendliness_delta=-0.02,
            feedback_text="User said I'm too curious"
        )
        assert success is True
        
        # Get cumulative deltas
        deltas = await db_connector.get_cumulative_personality_deltas()
        assert deltas['curiosity'] >= 0.05
        assert deltas['friendliness'] <= -0.02


class TestIntegration:
    """Integration tests for Phase 1 components."""
    
    def test_working_memory_and_mood_system(self):
        """Test working memory integrates with mood tracking."""
        wm = WorkingMemory(initial_mood=50)
        
        # Simulate face recognition (positive mood)
        wm.observe_face('face-uuid-1', name='Friend', mood_impact=10)
        assert wm.current_mood == 60
        
        # Simulate being moved roughly (negative mood)
        wm.update_mood(-20, 'moved_roughly')
        assert wm.current_mood == 40
        
        # Check emotion history recorded both events
        recent_events = wm.get_recent_emotion_events(max_events=2)
        assert len(recent_events) == 2
        assert recent_events[0].event_type == 'face_recognized'
        assert recent_events[1].event_type == 'moved_roughly'


# ========== Manual Integration Test Script ==========

async def manual_test_full_stack():
    """
    Manual test of full Phase 1 stack.
    Requires SQL Server running with vector_memory database.
    
    Run with: python -m pytest tests/test_phase1_memory.py::manual_test_full_stack -v -s
    """
    print("\n=== Phase 1 Manual Integration Test ===\n")
    
    # 1. Initialize database connector
    print("1. Connecting to SQL Server...")
    db = SQLServerConnector(server='localhost', database='vector_memory')
    connection_ok = await db.test_connection()
    print(f"   Connection status: {'OK' if connection_ok else 'FAILED'}")
    
    if not connection_ok:
        print("   ERROR: Cannot connect to SQL Server. Aborting test.")
        return
    
    # 2. Initialize working memory
    print("\n2. Initializing working memory...")
    wm = WorkingMemory(initial_mood=50)
    print(f"   Initial mood: {wm.current_mood}")
    
    # 3. Simulate face detection
    print("\n3. Simulating face detection...")
    face_id = await db.create_face(name='TestUser')
    print(f"   Created face in database: {face_id}")
    
    wm.observe_face(face_id, name='TestUser', mood_impact=10)
    print(f"   Face observed in working memory, mood: {wm.current_mood}")
    
    # 4. Simulate conversation
    print("\n4. Simulating conversation...")
    conv_id = await db.store_conversation(
        speaker_id=face_id,
        text="Hello Vector!",
        emotional_context=wm.current_mood,
        response_text="Hi TestUser!",
        response_type='sdk'
    )
    print(f"   Conversation stored: {conv_id}")
    
    # 5. Test budget tracking
    print("\n5. Testing budget tracking...")
    budget = await db.get_today_budget(vitality_level=0.8)
    print(f"   Today's budget: €{budget['euros_spent']:.4f} / €{budget['budget_limit_euros']:.4f}")
    
    await db.update_budget_spending(euros_spent=0.01, api_calls=1)
    print("   Updated budget with €0.01 spend")
    
    # 6. Get session summary
    print("\n6. Session summary:")
    summary = wm.get_session_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # 7. Cleanup
    await db.close()
    print("\n=== Test Complete ===\n")


if __name__ == '__main__':
    # Run manual integration test
    asyncio.run(manual_test_full_stack())
