import asyncio
import sys, os
import pytest
# Ensure repository root is on sys.path so tests can import package modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_personality.cognition.groq_client import GroqClient

class FakeOpenAI:
    def __init__(self, resp="OpenAI fallback response"):
        self._resp = resp
    async def chat_completion(self, messages, model=None, temperature=0.7, max_tokens=150, max_retries=None):
        await asyncio.sleep(0)  # ensure coroutine
        return self._resp

class DummyException(Exception):
    pass

def test_groq_fallback_to_openai(monkeypatch):
    # Create GroqClient with dummy key
    gc = GroqClient(api_key='dummy', default_model='llama-3.3-70b-versatile', max_retries=1)

    # Monkeypatch the underlying groq client to raise an error on call
    class DummyCompletions:
        async def create(self, **kwargs):
            raise Exception('simulated groq failure')
    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()
    class DummyClient:
        def __init__(self):
            self.chat = DummyChat()
    gc.client = DummyClient()

    # Attach fake OpenAI client
    fake_openai = FakeOpenAI(resp='FAKE OPENAI')
    gc.openai_client = fake_openai
    gc.openai_fallback_enabled = True

    # Call chat_completion - should return OpenAI response when Groq fails
    messages = [{'role':'user','content':'Hello'}]
    res = asyncio.get_event_loop().run_until_complete(gc.chat_completion(messages))
    assert res == 'FAKE OPENAI'