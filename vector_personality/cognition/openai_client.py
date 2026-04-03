"""
OpenAI Client Module

Wrapper for OpenAI API (GPT-4 Turbo, Whisper, TTS).

Features:
- GPT-4 Turbo chat completions with streaming
- Whisper audio transcription
- Rate limit handling with exponential backoff
- Comprehensive error handling

Phase 4 - Cognition & OpenAI Integration
"""

import asyncio
from openai import AsyncOpenAI, OpenAIError, RateLimitError, APIError
from typing import Optional, List, Dict, Any, AsyncIterator
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI API client with error handling.
    
    Attributes:
        api_key: OpenAI API key
        default_model: Default GPT model
        max_retries: Maximum retry attempts for rate limits
    """
    
    FALLBACK_RESPONSES = [
        "Ho problemi di connessione. Puoi riprovare?",
        "I miei circuiti sono un po' sovraccarichi. Dammi un attimo!",
        "Hmm, devo pensarci. Puoi chiedere di nuovo?",
        "Sto avendo difficoltà tecniche. Scusa!"
    ]
    
    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o",
        max_retries: int = 3
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            default_model: Default GPT model
            max_retries: Max retry attempts
        """
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
        self.max_retries = max_retries
        
        logger.info(f"OpenAIClient initialized: model={default_model}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: Optional[int] = None
    ) -> str:
        """
        Generate chat completion with GPT-4.
        
        Args:
            messages: List of message dicts (role, content)
            model: Model name (default: self.default_model)
            temperature: Randomness (0-2, default: 0.7)
            max_tokens: Max response tokens
            max_retries: Override default retry count
        
        Returns:
            Response text
        """
        model = model or self.default_model
        max_retries = max_retries if max_retries is not None else self.max_retries
        
        # Make API call with retries
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract response
                content = response.choices[0].message.content
                
                logger.info(f"Chat completion success: {len(content)} chars")
                return content
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    return self._get_fallback_response()
            
            except APIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0)
                else:
                    return self._get_fallback_response()
            
            except Exception as e:
                logger.error(f"Unexpected error in chat completion: {e}")
                return self._get_fallback_response()
        
        return self._get_fallback_response()
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """
        Generate streaming chat completion.
        
        Yields response chunks as they arrive for real-time TTS.
        
        Args:
            messages: List of message dicts
            model: Model name
            temperature: Randomness
            max_tokens: Max response tokens
        
        Yields:
            Response text chunks
        """
        model = model or self.default_model
        
        try:
            response_stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            logger.info(f"Streaming completion: {len(full_response)} chars")
            
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
            yield self._get_fallback_response()
    
    async def transcribe_audio(
        self,
        audio_file: Any,
        duration_seconds: float,
        language: str = "en"
    ) -> str:
        """
        Transcribe audio with Whisper API.
        
        Args:
            audio_file: Audio file object (file-like)
            duration_seconds: Audio duration for cost calculation
            language: Language code (default: "en")
        
        Returns:
            Transcribed text
        """
        try:
            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
            
            text = response.text
            
            logger.info(f"Whisper transcription: {len(text)} chars from {duration_seconds:.1f}s audio")
            return text
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
    
    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1-hd",
        output_path: Optional[Path] = None
    ) -> Optional[bytes]:
        """
        Generate speech from text with TTS API.
        
        Args:
            text: Text to speak
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model (tts-1 or tts-1-hd)
            output_path: Save audio to file (optional)
        
        Returns:
            Audio bytes (or None if saved to file)
        """
        try:
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Get audio bytes
            audio_bytes = response.content
            
            # Save to file if requested
            if output_path:
                output_path.write_bytes(audio_bytes)
                logger.info(f"TTS saved to {output_path}")
                audio_result = None
            else:
                audio_result = audio_bytes
            
            logger.info(f"TTS generated: {len(text)} chars")
            return audio_result
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def _get_fallback_response(self) -> str:
        """Get random fallback response"""
        return random.choice(self.FALLBACK_RESPONSES)


# Factory function
def create_openai_client(
    api_key: str,
    model: str = "gpt-4o"
) -> OpenAIClient:
    """
    Create and initialize OpenAIClient.
    
    Args:
        api_key: OpenAI API key
        model: Default GPT model
    
    Returns:
        Initialized OpenAIClient instance
    """
    return OpenAIClient(
        api_key=api_key,
        default_model=model
    )
