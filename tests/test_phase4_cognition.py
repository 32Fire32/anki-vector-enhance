"""
Phase 4: Cognition & OpenAI Integration Tests

Test-Driven Development approach for cognition system:
1. BudgetEnforcer: €2/hour limit with cost tracking and fail-fast
2. OpenAIClient: GPT-4 Turbo wrapper with streaming and error handling
3. ResponseGenerator: Personality-aware prompts with conversation history
4. ReasoningEngine: Context assembly from all sensors for intelligent responses

Test Coverage:
- BudgetEnforcer: Cost calculation, hourly limits, fail-fast logic, database integration
- OpenAIClient: GPT-4/Whisper API calls, streaming, rate limits, error handling
- ResponseGenerator: System prompt construction, conversation history, personality influence
- ReasoningEngine: Context assembly, memory retrieval, curiosity-based questions
- Integration: Question → context → GPT-4 → response → TTS → budget update
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import uuid4
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Phase 4 modules (will fail until implemented)
try:
    from vector_personality.cognition.budget_enforcer import BudgetEnforcer
except ImportError:
    BudgetEnforcer = None

try:
    from vector_personality.cognition.openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from vector_personality.cognition.response_generator import ResponseGenerator
except ImportError:
    ResponseGenerator = None

try:
    from vector_personality.cognition.reasoning_engine import ReasoningEngine
except ImportError:
    ReasoningEngine = None

# Import dependencies
from vector_personality.memory.working_memory import WorkingMemory
from vector_personality.core.personality import PersonalityModule


# ============================================================================
# Test Class 1: BudgetEnforcer (Cost Tracking & Fail-Fast)
# ============================================================================

@pytest.mark.skipif(BudgetEnforcer is None, reason="BudgetEnforcer not yet implemented")
class TestBudgetEnforcer:
    """Test budget enforcement with €2/hour limit"""

    def setup_method(self):
        """Initialize BudgetEnforcer for each test"""
        # Mock database connector with async methods
        self.mock_db = MagicMock()
        self.mock_db.query = AsyncMock(return_value=[{'total': 0.0}])
        self.mock_db.execute = AsyncMock()
        self.enforcer = BudgetEnforcer(
            db_connector=self.mock_db,
            hourly_limit_euros=2.0
        )

    def test_initialization(self):
        """Test BudgetEnforcer initializes with correct limits"""
        assert self.enforcer.hourly_limit_euros == 2.0
        assert self.enforcer.db_connector == self.mock_db

    def test_gpt4_turbo_cost_calculation(self):
        """Test GPT-4 Turbo token cost calculation"""
        # GPT-4 Turbo: $0.01/1K input, $0.03/1K output (€0.009/€0.027 at 1.11 rate)
        input_tokens = 1000
        output_tokens = 500
        
        cost = self.enforcer.calculate_cost(
            model="gpt-4-turbo",
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Input: 1000 * 0.009 / 1000 = 0.009
        # Output: 500 * 0.027 / 1000 = 0.0135
        # Total: ~0.0225 euros
        assert 0.020 <= cost <= 0.025

    def test_whisper_cost_calculation(self):
        """Test Whisper API cost calculation"""
        # Whisper: $0.006/minute (€0.0054 at 1.11 rate)
        duration_seconds = 30.0
        
        cost = self.enforcer.calculate_cost(
            model="whisper-1",
            duration_seconds=duration_seconds
        )
        
        # 30 seconds = 0.5 minutes * 0.0054 = 0.0027 euros
        assert 0.002 <= cost <= 0.003

    def test_tts_cost_calculation(self):
        """Test TTS HD cost calculation"""
        # TTS HD: $0.030/1M chars (€0.027/1M at 1.11 rate)
        char_count = 1000
        
        cost = self.enforcer.calculate_cost(
            model="tts-1-hd",
            char_count=char_count
        )
        
        # 1000 chars * 0.027 / 1000000 = 0.000027 euros
        assert cost < 0.0001

    @pytest.mark.asyncio
    async def test_check_budget_allows_within_limit(self):
        """Test budget check allows requests within hourly limit"""
        # Mock database to return €1.50 used in past hour
        self.mock_db.query = AsyncMock(return_value=[{'total': 1.50}])
        
        # Request €0.40 (total would be €1.90, under €2.00 limit)
        allowed = await self.enforcer.check_budget(estimated_cost=0.40)
        
        assert allowed is True
        self.mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_budget_blocks_over_limit(self):
        """Test budget check blocks requests over hourly limit"""
        # Mock database to return €1.80 used in past hour
        self.mock_db.query = AsyncMock(return_value=[{'total': 1.80}])
        
        # Request €0.30 (total would be €2.10, over €2.00 limit)
        allowed = await self.enforcer.check_budget(estimated_cost=0.30)
        
        assert allowed is False

    @pytest.mark.asyncio
    async def test_record_usage_saves_to_database(self):
        """Test usage recording saves to database"""
        # Mock successful database insert
        self.mock_db.execute = AsyncMock()
        
        await self.enforcer.record_usage(
            model="gpt-4-turbo",
            input_tokens=500,
            output_tokens=200,
            cost_euros=0.015
        )
        
        # Verify database was called
        self.mock_db.execute.assert_called_once()
        # Check that execute was called with INSERT statement
        call_args = self.mock_db.execute.call_args[0]
        assert 'INSERT INTO budget_usage' in call_args[0]

    @pytest.mark.asyncio
    async def test_get_remaining_budget(self):
        """Test remaining budget calculation"""
        # Mock €1.30 used in past hour
        self.mock_db.query = AsyncMock(return_value=[{'total': 1.30}])
        
        remaining = await self.enforcer.get_remaining_budget()
        
        assert 0.69 <= remaining <= 0.71  # €2.00 - €1.30 = €0.70

    @pytest.mark.asyncio
    async def test_fail_fast_on_zero_budget(self):
        """Test fail-fast immediately returns False when budget exhausted"""
        # Mock €2.00 used (exactly at limit)
        self.mock_db.query = AsyncMock(return_value=[{'total': 2.00}])
        
        # Should immediately return False without expensive calculations
        allowed = await self.enforcer.check_budget(estimated_cost=0.01)
        
        assert allowed is False


# ============================================================================
# Test Class 2: OpenAIClient (GPT-4 & Whisper Wrapper)
# ============================================================================

@pytest.mark.skipif(OpenAIClient is None, reason="OpenAIClient not yet implemented")
class TestOpenAIClient:
    """Test OpenAI API client with error handling"""

    def setup_method(self):
        """Initialize OpenAIClient for each test"""
        self.mock_budget = MagicMock()
        self.client = OpenAIClient(
            api_key="test-key",
            budget_enforcer=self.mock_budget
        )

    @pytest.mark.asyncio
    async def test_chat_completion_success(self):
        """Test successful GPT-4 chat completion"""
        # Mock budget allows request
        self.mock_budget.check_budget = AsyncMock(return_value=True)
        self.mock_budget.record_usage = AsyncMock()
        
        # Mock OpenAI API response with proper SDK v2+ structure
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Hello! How can I help you?'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        
        self.client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        response = await self.client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4-turbo"
        )
        
        assert response == 'Hello! How can I help you?'
        self.mock_budget.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_budget_blocked(self):
        """Test GPT-4 request blocked by budget"""
        # Mock budget denies request
        self.mock_budget.check_budget = AsyncMock(return_value=False)
        
        response = await self.client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4-turbo"
        )
        
        # Should return fallback message
        assert "budget" in response.lower() or "limit" in response.lower()

    @pytest.mark.asyncio
    async def test_chat_completion_with_streaming(self):
        """Test GPT-4 streaming mode"""
        self.mock_budget.check_budget = AsyncMock(return_value=True)
        self.mock_budget.record_usage = AsyncMock()
        
        # Simulate stream chunks with proper SDK v2+ structure
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content='Hello'))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=' there'))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content='!'))]),
            ]
            for chunk in chunks:
                yield chunk
        
        self.client.client.chat.completions.create = AsyncMock(return_value=mock_stream())
        
        full_response = ""
        async for chunk in self.client.chat_completion_stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4-turbo"
        ):
            full_response += chunk
        
        assert full_response == "Hello there!"

    @pytest.mark.asyncio
    async def test_whisper_transcription(self):
        """Test Whisper audio transcription"""
        self.mock_budget.check_budget = AsyncMock(return_value=True)
        self.mock_budget.record_usage = AsyncMock()
        
        # Mock Whisper API with proper SDK v2+ structure
        mock_response = MagicMock()
        mock_response.text = 'Hello Vector'
        self.client.client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        
        # Create mock audio file
        mock_audio = MagicMock()
        text = await self.client.transcribe_audio(
            audio_file=mock_audio,
            duration_seconds=3.0
        )
        
        assert text == 'Hello Vector'
        self.mock_budget.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self):
        """Test automatic retry on rate limit error"""
        from openai import RateLimitError, APIError
        self.mock_budget.check_budget = AsyncMock(return_value=True)
        self.mock_budget.record_usage = AsyncMock()
        
        # Mock success response
        mock_success = MagicMock()
        mock_success.choices = [MagicMock()]
        mock_success.choices[0].message.content = 'Success after retry'
        mock_success.usage = MagicMock()
        mock_success.usage.prompt_tokens = 10
        mock_success.usage.completion_tokens = 5
        
        # Create mock request for APIError
        mock_request = MagicMock()
        
        # First call fails with API error, second succeeds
        self.client.client.chat.completions.create = AsyncMock(
            side_effect=[
                APIError("Rate limit exceeded", request=mock_request, body=None),
                mock_success
            ]
        )
        
        with patch('asyncio.sleep'):  # Speed up test
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4-turbo",
                max_retries=2
            )
        
        assert response == 'Success after retry'
        assert self.client.client.chat.completions.create.call_count == 2


# ============================================================================
# Test Class 3: ResponseGenerator (Personality-Aware Prompts)
# ============================================================================

@pytest.mark.skipif(ResponseGenerator is None, reason="ResponseGenerator not yet implemented")
class TestResponseGenerator:
    """Test response generation with personality influence"""

    def setup_method(self):
        """Initialize ResponseGenerator for each test"""
        mock_memory = MagicMock()
        self.personality = PersonalityModule(mock_memory)
        self.mock_client = MagicMock()
        self.generator = ResponseGenerator(
            openai_client=self.mock_client,
            personality_module=self.personality
        )

    def test_system_prompt_includes_personality_traits(self):
        """Test system prompt reflects personality traits"""
        # Set high curiosity, low sassiness
        self.personality.base_traits.curiosity = 0.9
        self.personality.base_traits.sassiness = 0.2
        
        prompt = self.generator.build_system_prompt()
        
        # Should mention curiosity (checking keywords is sufficient)
        assert "curious" in prompt.lower() or "inquisitive" in prompt.lower() or "wonder" in prompt.lower()
        # Should not be overly sarcastic with low sassiness
        assert len(prompt) > 50  # Has substantial content

    def test_system_prompt_changes_with_mood(self):
        """Test system prompt adapts to current mood"""
        # Happy mood
        prompt_happy = self.generator.build_system_prompt(mood=80)
        assert any(word in prompt_happy.lower() for word in ['cheerful', 'happy', 'positive', 'enthusiastic'])
        
        # Sad mood
        prompt_sad = self.generator.build_system_prompt(mood=20)
        assert any(word in prompt_sad.lower() for word in ['subdued', 'quiet', 'low', 'sad'])

    def test_system_prompt_includes_memory_context(self):
        """Test system prompt includes memory context when provided"""
        prompt = self.generator.build_system_prompt(
            context={
                "memory_context": "Ricordo: ieri hai detto che ti chiami Nicola."
            }
        )

        assert "📚" in prompt
        assert "memoria" in prompt.lower()
        assert "ti chiami nicola" in prompt.lower()

    def test_conversation_history_context(self):
        """Test conversation history is included in messages"""
        # Add conversation history
        history = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "I'm Vector!"},
            {"role": "user", "content": "Nice to meet you"}
        ]
        
        messages = self.generator.build_messages(
            user_input="How are you?",
            conversation_history=history,
            max_history=3
        )
        
        # Should include system prompt + history + new message
        assert len(messages) >= 4
        assert messages[0]['role'] == 'system'
        assert messages[-1]['content'] == "How are you?"

    def test_conversation_history_truncation(self):
        """Test conversation history is truncated to max_history"""
        # Add 10 conversation turns
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Reply {i}"})
        
        messages = self.generator.build_messages(
            user_input="New question",
            conversation_history=history,
            max_history=3  # Only keep last 3 turns
        )
        
        # System + 3 user messages + 3 assistant messages + new message = 8
        assert len(messages) <= 10  # System + up to 6 history + new

    @pytest.mark.asyncio
    async def test_generate_response_with_context(self):
        """Test response generation includes context"""
        # Mock OpenAI client
        self.mock_client.chat_completion = AsyncMock(
            return_value="I see you're in the bedroom!"
        )
        
        # Use proper context structure matching ReasoningEngine output
        response = await self.generator.generate_response(
            user_input="Where am I?",
            context={
                "room": "bedroom",
                "objects": [
                    {"type": "bed", "confidence": 0.9, "location": "center"},
                    {"type": "lamp", "confidence": 0.85, "location": "nightstand"}
                ]
            },
            mood=60
        )
        
        assert "bedroom" in response.lower()
        self.mock_client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_response_to_tts(self):
        """Test streaming response can be sent to TTS in real-time"""
        # Mock streaming client that returns async iterator
        async def mock_stream(*args, **kwargs):
            for word in ["Hello", " ", "there", "!"]:
                yield word
        
        self.mock_client.chat_completion_stream = mock_stream
        
        collected = []
        async for chunk in self.generator.generate_response_stream(
            user_input="Hi",
            mood=70
        ):
            collected.append(chunk)
        
        assert "".join(collected) == "Hello there!"

    def test_fallback_response_on_error(self):
        """Test fallback response when API fails"""
        # Simulate API error
        self.mock_client.chat_completion = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        response = asyncio.run(self.generator.generate_response(
            user_input="Hello",
            mood=50
        ))
        
        # Should return fallback, not raise exception
        assert response is not None
        assert len(response) > 0


# ============================================================================
# Test Class 4: ReasoningEngine (Context Assembly)
# ============================================================================

@pytest.mark.skipif(ReasoningEngine is None, reason="ReasoningEngine not yet implemented")
class TestReasoningEngine:
    """Test context assembly from all sensors"""

    def setup_method(self):
        """Initialize ReasoningEngine for each test"""
        self.working_memory = WorkingMemory()
        self.engine = ReasoningEngine(
            working_memory=self.working_memory
        )

    @pytest.mark.asyncio
    async def test_assemble_context_includes_all_sources(self):
        """Test context assembly includes faces, objects, room, mood"""
        # Populate working memory
        self.working_memory.observe_face("Alice", "user_123")
        self.working_memory.observe_object("laptop", 0.95, "on desk")
        self.working_memory.set_room("office")
        self.working_memory.update_mood(20, "face_recognized")
        
        context = await self.engine.assemble_context()
        
        assert 'faces' in context
        assert 'Alice' in str(context['faces'])
        assert 'objects' in context
        assert 'laptop' in str(context['objects'])
        assert 'room' in context
        assert context['room'] == 'office'
        assert 'mood' in context
        assert context['mood'] > 50  # Mood increased from face

    @pytest.mark.asyncio
    async def test_context_prioritizes_recent_observations(self):
        """Test context prioritizes recent observations over old ones"""
        # Add old observation
        self.working_memory.observe_object("old_item", 0.5, "seen 10 min ago")
        
        # Wait and add recent observation
        time.sleep(0.1)
        self.working_memory.observe_object("new_item", 0.9, "just seen")
        
        context = await self.engine.assemble_context(max_objects=1)
        
        # Should prioritize new_item
        assert 'new_item' in str(context['objects'])

    @pytest.mark.asyncio
    async def test_generate_curiosity_question(self):
        """Test curiosity-based question generation"""
        # Observe unknown object
        self.working_memory.observe_object("unknown_device", 0.85, "on table")
        
        question = await self.engine.generate_curiosity_question()
        
        # Should ask about the unknown object
        assert question is not None
        assert any(word in question.lower() for word in ['what', 'unknown', 'device'])

    @pytest.mark.asyncio
    async def test_no_question_when_no_new_stimuli(self):
        """Test no question generated when nothing new to explore"""
        # Empty working memory
        question = await self.engine.generate_curiosity_question()
        
        # Should return None or generic question
        assert question is None or len(question) < 50

    @pytest.mark.asyncio
    async def test_context_includes_personality_state(self):
        """Test context includes current personality trait values"""
        # Set up personality module on working_memory
        from vector_personality.core.personality import PersonalityModule
        mock_memory = MagicMock()
        self.working_memory.personality = PersonalityModule(mock_memory)
        
        context = await self.engine.assemble_context(include_personality=True)
        
        assert 'personality' in context
        assert 'curiosity' in str(context['personality'])

    @pytest.mark.asyncio
    async def test_memory_retrieval_for_context(self):
        """Test retrieval of relevant memories for context"""
        # Populate working memory with conversation
        self.working_memory.observe_face("Bob", "user_456")
        
        # Mock database with past conversations about Bob
        mock_db = MagicMock()
        mock_db.query = AsyncMock(
            return_value=[{"user_text": "Hi", "bot_response": "Hello!", "timestamp": datetime.now()}]
        )
        self.engine.db_connector = mock_db
        
        context = await self.engine.assemble_context(include_history=True)
        
        # Should include past conversations with correct key name
        assert 'recent_conversations' in context
        assert len(context['recent_conversations']) > 0


# ============================================================================
# Integration Test: Complete Cognition Pipeline
# ============================================================================

@pytest.mark.skipif(
    any(cls is None for cls in [BudgetEnforcer, OpenAIClient, ResponseGenerator, ReasoningEngine]),
    reason="Phase 4 modules not yet implemented"
)
class TestCognitionIntegration:
    """Test complete cognition pipeline"""

    def setup_method(self):
        """Initialize all cognition components"""
        # Mock database with proper async methods
        self.mock_db = MagicMock()
        self.mock_db.query = AsyncMock(return_value=[{'total': 0.50}])  # €0.50 used
        self.mock_db.execute = AsyncMock()
        
        # Initialize components
        self.budget = BudgetEnforcer(
            db_connector=self.mock_db,
            hourly_limit_euros=2.0
        )
        
        self.openai_client = OpenAIClient(
            api_key="test-key",
            budget_enforcer=self.budget
        )
        
        mock_memory = MagicMock()
        self.personality = PersonalityModule(mock_memory)
        
        self.response_gen = ResponseGenerator(
            openai_client=self.openai_client,
            personality_module=self.personality
        )
        
        self.working_memory = WorkingMemory()
        self.reasoning = ReasoningEngine(
            working_memory=self.working_memory
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_question_to_response(self):
        """Test: Question → Context → GPT-4 → Response → Budget Update"""
        # Setup context
        self.working_memory.observe_face("Charlie", "user_789")
        self.working_memory.set_room("living_room")
        self.working_memory.update_mood(15, "face_recognized")
        
        # Mock database for budget
        self.mock_db.query = AsyncMock(return_value=[{'total': 0.5}])
        self.mock_db.execute = AsyncMock()
        
        # Mock GPT-4 response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Hi Charlie! Great to see you in the living room!'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        
        self.openai_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Assemble context
        context = await self.reasoning.assemble_context()
        
        # Generate response
        response = await self.response_gen.generate_response(
            user_input="Hello Vector!",
            context=context,
            mood=self.working_memory.current_mood
        )
        
        # Verify response includes context
        assert "Charlie" in response or "living_room" in response.lower()
        
        # Verify budget was recorded
        self.mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_budget_enforcement_blocks_expensive_request(self):
        """Test budget enforcement prevents over-limit requests"""
        # Clear budget cache and set at limit (any request will exceed)
        self.budget._cached_usage = None
        self.budget._last_check_time = None
        self.mock_db.query = AsyncMock(return_value=[{'total': 2.00}])  # €2.00 used (at limit)
        
        # Try to make request (any cost will exceed limit)
        response = await self.response_gen.generate_response(
            user_input="Tell me a long story",
            mood=50
        )
        
        # Should return budget error message
        assert "budget" in response.lower() or "limit" in response.lower()

    @pytest.mark.asyncio
    async def test_personality_influences_response_style(self):
        """Test personality traits affect response generation"""
        # High sassiness personality
        self.personality.base_traits.sassiness = 0.9
        
        with patch('openai.ChatCompletion.acreate') as mock_api:
            mock_api.return_value = {
                'choices': [{
                    'message': {'content': 'Well, obviously the sky is blue! 🙄'}
                }],
                'usage': {'prompt_tokens': 50, 'completion_tokens': 15}
            }
            
            # Build system prompt
            system_prompt = self.response_gen.build_system_prompt()
            
            # Should reflect sassy personality
            assert any(word in system_prompt.lower() for word in ['sarcastic', 'witty', 'playful', 'sassy'])

    @pytest.mark.asyncio
    async def test_cognition_response_time_under_5_seconds(self):
        """Test complete pipeline responds within 5 seconds"""
        with patch('openai.ChatCompletion.acreate') as mock_api:
            mock_api.return_value = {
                'choices': [{
                    'message': {'content': 'Quick response'}
                }],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 5}
            }
            
            start = time.time()
            
            context = await self.reasoning.assemble_context()
            response = await self.response_gen.generate_response(
                user_input="Hi",
                context=context,
                mood=50
            )
            
            elapsed = time.time() - start
            
            assert elapsed < 5.0  # Should be fast with mocked API
            assert response is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
