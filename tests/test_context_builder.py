import pytest
from unittest.mock import AsyncMock

from vector_personality.cognition.context_builder import MemoryRetriever, ContextBuilder


def test_detect_memory_request_keywords():
    r = MemoryRetriever()
    req = r.detect_memory_request("Vector, ti ricordi quando abbiamo parlato di Paw Patrol?")
    assert req.is_memory_request is True
    assert any("paw" in k.lower() for k in req.keywords)


def test_detect_memory_request_for_name_question():
    r = MemoryRetriever()
    req = r.detect_memory_request("Come si chiama mio figlio?")
    assert req.is_memory_request is True


@pytest.mark.asyncio
async def test_context_builder_injects_retrieved_memories():
    class DummyWM:
        current_mood = 55
        current_room = "salotto"

    mock_db = AsyncMock()
    mock_db.get_known_faces.return_value = [
        {"name": "Nicola", "total_interactions": 10, "last_seen": None},
    ]
    mock_db.get_recent_objects.return_value = [
        {"object_type": "palla", "confidence": 0.9, "last_detected": None},
    ]
    mock_db.get_recent_conversations.return_value = [
        {"timestamp": None, "text": "Ciao", "speaker_name": "Nicola"},
    ]
    mock_db.search_old_conversations.return_value = [
        {"timestamp": None, "text": "Parliamo di Paw Patrol", "person": "Nicola", "days_ago": 7},
    ]

    builder = ContextBuilder(db_connector=mock_db, working_memory=DummyWM())
    ctx = await builder.build_conversation_context(user_text="ti ricordi Paw Patrol?")

    assert "RICORDI RECUPERATI" in ctx
    assert "Paw Patrol" in ctx


@pytest.mark.asyncio
async def test_context_builder_can_retrieve_yesterdays_fact_even_if_not_in_recent_list():
    class DummyWM:
        current_mood = 55
        current_room = None

    mock_db = AsyncMock()
    mock_db.get_known_faces.return_value = []
    mock_db.get_recent_objects.return_value = []
    # Simulate a busy day: the base "recent" list doesn't include yesterday's relevant fact.
    mock_db.get_recent_conversations.return_value = [
        {"timestamp": None, "text": "Parliamo di altro", "speaker_name": "Unknown", "response_text": "Ok"},
    ]
    mock_db.search_old_conversations.return_value = [
        {"timestamp": None, "text": "Mio fratello si chiama Paolo", "person": "Unknown", "days_ago": 1},
    ]

    builder = ContextBuilder(db_connector=mock_db, working_memory=DummyWM())
    ctx = await builder.build_conversation_context(user_text="ti ricordi mio fratello Paolo?")

    assert "RICORDI RECUPERATI" in ctx
    assert "Paolo" in ctx
    # Must include recent conversations too (exclude_recent_hours=0)
    mock_db.search_old_conversations.assert_awaited()
    _, kwargs = mock_db.search_old_conversations.await_args
    assert kwargs.get("exclude_recent_hours") == 0
