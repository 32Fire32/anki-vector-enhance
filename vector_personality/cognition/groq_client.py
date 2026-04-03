"""
Groq Client Module

Wrapper for Groq API (Llama 3.3 70B, Mixtral).
Privacy-focused: Groq doesn't train on API data.

Features:
- OpenAI-compatible API format (easy migration)
- Llama 3.1 70B Versatile (recommended for conversations)
- Extremely fast inference (300+ tokens/sec)
- Comprehensive error handling

Phase 10 - LLM Migration (T121)
"""

import asyncio
from groq import AsyncGroq, GroqError
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GroqClient:
    """
    Groq API client with budget enforcement and error handling.
    
    Privacy Benefits:
    - Groq doesn't train on API data
    - No data retention for inference
    - EU-compliant infrastructure
    - Open-source models (Llama 3.1, Mixtral)
    
    Attributes:
        api_key: Groq API key
        default_model: Default model (llama-3.1-70b-versatile)
        max_retries: Maximum retry attempts
    """
    
    FALLBACK_RESPONSES = [
        "Sto avendo problemi di connessione. Puoi riprovare?",
        "I miei circuiti sono un po' sovraccarichi. Dammi un momento!",
        "Hmm, devo pensarci su. Puoi chiedere di nuovo?",
        "Sto riscontrando alcune difficoltà tecniche. Mi dispiace!"
    ]
    
    # Available Groq models (updated December 2025)
    MODELS = {
        "llama-3.3-70b-versatile": {
            "context_window": 131072,
            "description": "Best for conversations, reasoning, and Italian (replaces 3.1)",
            "speed": "fast"
        },
        "llama-3.1-8b-instant": {
            "context_window": 131072,
            "description": "Fastest, good for simple tasks",
            "speed": "very_fast"
        },
        "mixtral-8x7b-32768": {
            "context_window": 32768,
            "description": "Good alternative, multilingual",
            "speed": "fast"
        },
        "gemma2-9b-it": {
            "context_window": 8192,
            "description": "Google's Gemma 2 9B",
            "speed": "fast"
        }
    }
    
    def __init__(
        self,
        api_key: str,
        default_model: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
        openai_client: Optional[Any] = None,
        openai_fallback_enabled: bool = False,
        openai_fallback_model: Optional[str] = None
    ):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            default_model: Default model (llama-3.3-70b-versatile recommended)
            max_retries: Max retry attempts
            openai_client: Optional OpenAIClient instance to use as fallback
            openai_fallback_enabled: Whether fallback to OpenAI is enabled
            openai_fallback_model: Optional model override for OpenAI fallback
        """
        self.api_key = api_key
        self.client = AsyncGroq(api_key=api_key)
        self.default_model = default_model
        self.max_retries = max_retries
        # OpenAI fallback members (may be set later by vector_agent)
        self.openai_client = openai_client
        self.openai_fallback_enabled = openai_fallback_enabled
        self.openai_fallback_model = openai_fallback_model

        logger.info(f"GroqClient initialized: model={default_model} (privacy-focused)")
        logger.info(f"✅ Model: {self.MODELS[default_model]['description']}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: Optional[int] = None
    ) -> str:
        """
        Generate chat completion with Llama 3.1.
        
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
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.debug(f"Groq chat_completion: attempt {attempt + 1}/{max_retries}")
                
                # Call Groq API (OpenAI-compatible interface)
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response_text = response.choices[0].message.content
                
                logger.info(f"✅ Groq response: {len(response_text)} chars, "
                           f"{response.usage.total_tokens} tokens")
                return response_text
            
            except GroqError as e:
                logger.error(f"Groq API error (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed - try OpenAI fallback if available
                    logger.error("Groq API failed after all retries")
                    if getattr(self, 'openai_fallback_enabled', False) and getattr(self, 'openai_client', None):
                        try:
                            logger.warning("⚠️ Groq unavailable - attempting OpenAI fallback")
                            # Use fallback model override if provided, else default
                            fb_model = self.openai_fallback_model
                            response_text = await self.openai_client.chat_completion(
                                messages=messages,
                                model=fb_model,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            logger.info("✅ OpenAI fallback response received")
                            return response_text
                        except Exception as e2:
                            logger.error(f"OpenAI fallback failed: {e2}")
                            return self._get_fallback_response()
                    else:
                        return self._get_fallback_response()
            
            except Exception as e:
                logger.error(f"Unexpected error in chat_completion: {e}")
                # Try OpenAI fallback if available
                if getattr(self, 'openai_fallback_enabled', False) and getattr(self, 'openai_client', None):
                    try:
                        logger.warning("⚠️ Groq unexpected error - attempting OpenAI fallback")
                        fb_model = self.openai_fallback_model
                        response_text = await self.openai_client.chat_completion(
                            messages=messages,
                            model=fb_model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        logger.info("✅ OpenAI fallback response received")
                        return response_text
                    except Exception as e2:
                        logger.error(f"OpenAI fallback failed: {e2}")
                        return self._get_fallback_response()
                return self._get_fallback_response()
        
        return self._get_fallback_response()
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> Optional[Any]:
        """
        Stream chat completion (for TTS integration).
        
        Args:
            messages: List of message dicts
            model: Model name
            temperature: Randomness
            max_tokens: Max tokens
        
        Returns:
            Stream iterator or None on error
        """
        model = model or self.default_model
        
        try:
            logger.debug(f"Groq streaming: model={model}")
            
            # Create streaming response
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            logger.info("✅ Groq streaming started")
            return stream
        
        except GroqError as e:
            logger.error(f"Groq streaming error: {e}")
            # Try OpenAI streaming fallback
            if getattr(self, 'openai_fallback_enabled', False) and getattr(self, 'openai_client', None):
                try:
                    fb_model = self.openai_fallback_model
                    logger.warning("⚠️ Groq streaming failed - attempting OpenAI streaming fallback")
                    return await self.openai_client.chat_completion_stream(messages=messages, model=fb_model, temperature=temperature, max_tokens=max_tokens)
                except Exception as e2:
                    logger.error(f"OpenAI streaming fallback failed: {e2}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
            return None
    
    def _get_fallback_response(self) -> str:
        """Return random Italian fallback response."""
        import random
        return random.choice(self.FALLBACK_RESPONSES)
    
    async def close(self):
        """Close client connection."""
        # Groq client doesn't need explicit closing
        logger.info("GroqClient closed")


# Backward compatibility alias (for easier migration)
LLMClient = GroqClient
