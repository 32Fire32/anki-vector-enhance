"""
Speech Recognition Module

Integrates Groq Whisper API for audio transcription.
Handles API calls, confidence scoring, retry logic, and error handling.

Key Features:
- Groq Whisper Large V3 for audio → text conversion
- Confidence scoring based on transcription quality
- Retry logic for API failures
- Language detection support

API Cost: FREE (Groq provides free API access)
Performance Target: <5s for 30s audio clip
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from openai import AsyncOpenAI
from datetime import datetime

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Speech recognition using Groq Whisper API
    
    Converts audio files to text with confidence scoring and
    automatic retry logic for API failures.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
        language: Optional[str] = None
    ):
        """
        Initialize SpeechRecognizer
        
        Args:
            api_key: Groq API key
            model: Whisper model name (default: "whisper-large-v3")
            language: Optional language code (e.g., "en", "it")
                     If None, language is auto-detected
        """
        self.api_key = api_key
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
        self.language = language
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_audio_seconds = 0.0
        
        logger.info(
            f"SpeechRecognizer initialized: model={model}, "
            f"language={language or 'auto-detect'}"
        )

    async def transcribe(
        self,
        audio_path: str,
        max_retries: int = 3,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            max_retries: Maximum retry attempts on failure
            prompt: Optional prompt to guide transcription style
        
        Returns:
            Dictionary with keys:
                - text: Transcribed text
                - language: Detected language code
                - confidence: Confidence score (0.0-1.0)
                - duration: Audio duration in seconds
                - timestamp: Transcription timestamp
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: After all retries exhausted
        """
        # Validate file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file size for duration estimate (rough)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        estimated_duration = file_size_mb * 60  # Very rough estimate
        
        # Retry loop
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Transcribing {audio_path} (attempt {attempt + 1}/{max_retries})")
                
                # Open audio file
                with open(audio_path, 'rb') as audio_file_obj:
                    # Call Whisper API (AsyncOpenAI)
                    response = await self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file_obj,
                        response_format="verbose_json",
                        language=self.language,
                        prompt=prompt
                    )
                
                # Extract result
                text = response.text.strip() if response.text else ""
                detected_language = getattr(response, 'language', 'unknown')
                duration = getattr(response, 'duration', estimated_duration)
                
                # Clean up common Whisper hallucinations (subtitle artifacts)
                text = self._remove_whisper_hallucinations(text)
                
                # Calculate confidence
                confidence = self.calculate_confidence(text, duration)
                
                # Update statistics
                self.total_calls += 1
                self.total_audio_seconds += duration
                
                result = {
                    "text": text,
                    "language": detected_language,
                    "confidence": confidence,
                    "duration": duration,
                    "timestamp": datetime.now()
                }
                
                logger.info(
                    f"Transcription successful: {len(text)} chars, "
                    f"language={detected_language}, confidence={confidence:.2f}"
                )
                
                return result
            
            except Exception as e:
                last_error = e
                self.total_failures += 1
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        logger.error(f"Transcription failed after {max_retries} attempts")
        raise Exception(f"Transcription failed: {last_error}")

    def calculate_confidence(
        self,
        text: str,
        duration: float
    ) -> float:
        """
        Calculate confidence score for transcription
        
        Heuristic based on:
        - Text length vs audio duration
        - Presence of filler words
        - Character diversity
        
        Args:
            text: Transcribed text
            duration: Audio duration in seconds
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not text or duration <= 0:
            return 0.0
        
        # Baseline confidence
        confidence = 0.7
        
        # Check text length appropriateness
        # Average speaking rate: ~150 words/minute = 2.5 words/second
        expected_words = duration * 2.5
        actual_words = len(text.split())
        
        if actual_words > 0:
            word_ratio = actual_words / max(expected_words, 1)
            # Ideal ratio: 0.5 - 1.5 (accounting for pauses)
            if 0.5 <= word_ratio <= 1.5:
                confidence += 0.15
            elif word_ratio < 0.3:
                confidence -= 0.2  # Too few words = poor detection
        
        # Check for empty or very short text
        if len(text) < 3:
            confidence = 0.1
        elif len(text) < 10:
            confidence -= 0.2
        else:
            confidence += 0.1
        
        # Check character diversity (more variety = higher confidence)
        unique_chars = len(set(text.lower()))
        if unique_chars > 10:
            confidence += 0.05
        
        # Clamp to 0-1
        return max(0.0, min(1.0, confidence))

    async def transcribe_batch(
        self,
        audio_paths: list,
        max_concurrent: int = 3
    ) -> list:
        """
        Transcribe multiple audio files concurrently
        
        Args:
            audio_paths: List of audio file paths
            max_concurrent: Maximum concurrent API calls
        
        Returns:
            List of transcription results (same order as input)
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_with_limit(path):
            async with semaphore:
                return await self.transcribe(path)
        
        # Execute all transcriptions
        tasks = [transcribe_with_limit(path) for path in audio_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Batch transcription complete: {len(audio_paths)} files")
        return results
    
    def _remove_whisper_hallucinations(self, text: str) -> str:
        """
        Remove common Whisper hallucinations (subtitle artifacts, credits, etc.)
        
        Whisper was trained on subtitle data and sometimes adds phrases like:
        - "Sottotitoli creati dalla comunità Amara.org"
        - "Sottotitoli e revisione a cura di QTSS"
        - "Grazie per aver guardato!"
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned text with hallucinations removed
        """
        if not text:
            return text
        
        # Common Italian subtitle phrases (case-insensitive)
        hallucination_patterns = [
            r'sottotitol[io].*',  # "Sottotitoli..." anything after
            r'.*amara\.org.*',     # Anything with amara.org
            r'.*qtss.*',           # QTSS subtitle service
            r'grazie per aver guardato.*',  # "Thanks for watching"
            r'iscriviti al canale.*',       # "Subscribe to the channel"
            r'metti mi piace.*',            # "Like this video"
            r'.*revisione a cura di.*',     # Revision credits
            r'.*comunità.*amara.*',         # Community credits
        ]
        
        import re
        cleaned = text
        
        for pattern in hallucination_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def is_likely_hallucination(
        self, 
        text: str, 
        confidence: float,
        has_wake_word: bool = False
    ) -> tuple[bool, str]:
        """
        Intelligently detect if transcription is likely a Whisper hallucination.
        
        Uses multi-level analysis:
        1. Lexical analysis: repetitions, simplicity, common patterns
        2. Confidence score validation
        3. Context awareness (wake word presence)
        
        Args:
            text: Transcribed text to analyze
            confidence: Whisper confidence score (0.0-1.0)
            has_wake_word: Whether text contains wake word (reduces false positives)
            
        Returns:
            Tuple of (is_hallucination: bool, reason: str)
        """
        if not text or len(text.strip()) == 0:
            return True, "empty_text"
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        words = text_clean.split()
        word_count = len(words)
        
        # Level 1: Obvious hallucination patterns (video artifacts)
        obvious_patterns = [
            r'al prossimo episodio',
            r'^\d{1,2}:\d{2}$',  # Timestamps like "00:00"
            r'^www\.',           # URLs
            r'sottotitol',
            r'comunità amara',
        ]
        
        import re
        for pattern in obvious_patterns:
            if re.search(pattern, text_lower):
                return True, f"obvious_pattern: {pattern}"
        
        # Level 2: Repetition detection (e.g., "si si si", "no no no")
        if word_count >= 2:
            # Check if all words are identical
            unique_words = set(w.lower() for w in words)
            if len(unique_words) == 1:
                # Even with wake word, repetitions are suspicious
                # Only allow if "vector" itself is repeated (e.g., "vector vector")
                # "ciao ciao", "ehi ehi" are still hallucinations
                is_vector_repetition = words[0].lower() == 'vector'
                if not is_vector_repetition:
                    return True, f"repetition: '{words[0]}' x{word_count}"
            
            # Check for alternating repetitions (e.g., "si no si no")
            if word_count >= 3 and len(unique_words) == 2:
                # Check if pattern repeats
                if all(words[i] == words[i % 2] for i in range(word_count)):
                    return True, f"alternating_repetition: {unique_words}"
        
        # Level 3: Single common word/phrase analysis
        # These are often hallucinations when spoken alone
        common_hallucinations = {
            'ciao': 'greeting_only',
            'ciao!': 'greeting_only',
            'buongiorno': 'greeting_only', 
            'buonasera': 'greeting_only',
            'buonanotte': 'greeting_only',
            'grazie': 'politeness_only',
            'prego': 'politeness_only',
            'scusa': 'politeness_only',
            'sì': 'affirmation_only',
            'no': 'negation_only',
            'ok': 'acknowledgment_only',
            'va bene': 'acknowledgment_only',
            "grazie per l'attenzione": 'video_artifact',
        }
        
        # Single-word hallucinations should ALWAYS be filtered
        # Even with wake word, single words like "Ciao" or "Sì" are suspicious
        # Only exception: if part of longer phrase (handled by word_count check later)
        if text_lower in common_hallucinations:
            # With wake word: allow only if 2+ words AND not in common list
            # This allows "sì grazie" but blocks "sì" alone
            if has_wake_word and word_count >= 2:
                # Check if it's a multi-word phrase containing common word
                pass  # Allow through to next checks
            else:
                return True, common_hallucinations[text_lower]
        
        # Level 4: Extremely low confidence
        # Whisper is conservative, so <0.3 is usually garbage
        if confidence < 0.3:
            return True, f"low_confidence: {confidence:.2f}"
        
        # Level 5: Very short + low confidence combination
        # Even with wake word, single word + low confidence = suspicious
        if word_count == 1 and confidence < 0.6:
            return True, f"short_and_low_confidence: {word_count} words, {confidence:.2f}"
        
        # Level 6: Lexical complexity analysis
        # Real speech has varied vocabulary; hallucinations often repeat
        if word_count >= 5:
            unique_ratio = len(unique_words) / word_count
            if unique_ratio < 0.4:  # Less than 40% unique words
                return True, f"low_lexical_diversity: {unique_ratio:.2f}"
        
        # Passed all checks - likely legitimate
        return False, "legitimate"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get transcription statistics
        
        Returns:
            Dictionary with usage statistics
        """
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = (self.total_calls - self.total_failures) / self.total_calls
        
        return {
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "success_rate": success_rate,
            "total_audio_seconds": self.total_audio_seconds,
            "total_audio_minutes": self.total_audio_seconds / 60,
            "estimated_cost_euros": (self.total_audio_seconds / 60) * 0.006
        }

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"SpeechRecognizer("
            f"model={self.model}, "
            f"language={self.language or 'auto'}, "
            f"calls={self.total_calls}, "
            f"failures={self.total_failures})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_speech_recognizer(
    api_key: str,
    budget_enforcer: Optional[Any] = None,
    config: dict = None
) -> SpeechRecognizer:
    """
    Factory function to create SpeechRecognizer with config
    
    Args:
        api_key: OpenAI API key
        budget_enforcer: Optional BudgetEnforcer for cost tracking
        config: Optional configuration dictionary
    
    Returns:
        Configured SpeechRecognizer instance
    """
    if config is None:
        config = {}
    
    return SpeechRecognizer(
        api_key=api_key,
        budget_enforcer=budget_enforcer,
        model=config.get('model', 'whisper-1'),
        language=config.get('language', None)
    )


async def transcribe_with_fallback(
    recognizer: SpeechRecognizer,
    audio_path: str,
    fallback_text: str = "[transcription failed]"
) -> str:
    """
    Transcribe with fallback on failure
    
    Args:
        recognizer: SpeechRecognizer instance
        audio_path: Path to audio file
        fallback_text: Text to return on failure
    
    Returns:
        Transcribed text or fallback text
    """
    try:
        result = await recognizer.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Transcription failed, using fallback: {e}")
        return fallback_text


def estimate_whisper_cost(audio_duration_seconds: float) -> float:
    """
    Estimate Whisper API cost
    
    Args:
        audio_duration_seconds: Audio duration in seconds
    
    Returns:
        Estimated cost in euros
    """
    minutes = audio_duration_seconds / 60
    cost_per_minute = 0.006  # €0.006 per minute
    return minutes * cost_per_minute
