"""
Local Whisper Speech Recognition using faster-whisper.

GPU-accelerated (CUDA) when available on the RTX 5070 Ti.
Falls back to CPU if CUDA is not found.

Same interface as SpeechRecognizer so the agent can use either
without code changes.

Performance (RTX 5070 Ti):
  - tiny  model: ~50ms  per utterance
  - small model: ~150ms per utterance   ← default
  - medium model: ~400ms per utterance
  - large-v3: ~900ms per utterance
"""

import asyncio
import concurrent.futures
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Common Whisper hallucination strings (Italian + generic)
_HALLUCINATION_FRAGMENTS = [
    "sottotitoli",
    "amara.org",
    "sottotitolato",
    "grazie per aver",
    "transcript",
    "transcribed by",
    "music playing",
    "[musica]",
    "[silenzio]",
]


class LocalWhisperRecognizer:
    """
    Local speech recognition using faster-whisper (CTranslate2).

    Compatible interface with SpeechRecognizer (Groq) so the agent
    can swap between them transparently.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: Optional[str] = "it",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        :param model_size: "tiny", "base", "small", "medium", "large-v3"
        :param language: ISO language code ("it", "en", …) or None for auto-detect
        :param device: "cuda", "cpu", or "auto" (picks CUDA if available)
        :param compute_type: "float16", "int8", or "auto"
        """
        self.model_size = model_size
        self.language = language
        self._model = None  # lazy-loaded on first call

        # Resolve device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Resolve compute type
        if compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        # Stats
        self.total_calls = 0
        self.total_failures = 0
        self.total_audio_seconds = 0.0

        # Dedicated executor for blocking Whisper inference.
        # Isolated from the asyncio loop's default executor to avoid
        # 'cannot schedule new futures after shutdown' errors.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper"
        )

        logger.info(
            f"LocalWhisperRecognizer ready: model={model_size} "
            f"device={self.device} compute={self.compute_type} "
            f"language={language or 'auto'}"
        )

    def _load_model(self):
        """Lazy-load the Whisper model (first call only)."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        logger.info(f"[Whisper] Loading {self.model_size} model on {self.device}...")
        t0 = time.time()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(f"[Whisper] Model loaded in {time.time()-t0:.1f}s ✅")

    async def transcribe(
        self,
        audio_path: str,
        max_retries: int = 2,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.

        :param audio_path: Path to WAV/MP3/etc.
        :param max_retries: Retry attempts on error.
        :param prompt: Optional initial prompt for Whisper.
        :returns: dict with keys: text, confidence, language, duration_seconds
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._transcribe_sync, audio_path, prompt
        )

    def _transcribe_sync(self, audio_path: str, prompt: Optional[str]) -> Dict[str, Any]:
        """Blocking transcription — run via executor to keep async loop free."""
        self.total_calls += 1
        t0 = time.time()

        try:
            self._load_model()

            path = Path(audio_path)
            file_size = path.stat().st_size if path.exists() else 0

            segments, info = self._model.transcribe(
                str(audio_path),
                language=self.language,
                initial_prompt=prompt,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 200,
                    "speech_pad_ms": 100,
                    "threshold": 0.5,           # Silero speech probability threshold (0-1)
                },
            )

            # Consume generator
            text_parts = []
            for seg in segments:
                text_parts.append(seg.text.strip())

            text = " ".join(text_parts).strip()
            elapsed = time.time() - t0
            audio_dur = info.duration if hasattr(info, 'duration') else 0.0
            self.total_audio_seconds += audio_dur

            # Filter hallucinations
            text_lower = text.lower()
            for fragment in _HALLUCINATION_FRAGMENTS:
                if fragment in text_lower:
                    logger.info(f"[Whisper] Hallucination filtered: {text!r}")
                    return {"text": "", "confidence": 0.0, "language": info.language, "duration_seconds": audio_dur}

            # Confidence proxy: ratio of non-whitespace chars to duration
            # (longer output per second = higher confidence)
            if audio_dur > 0 and text:
                char_rate = len(text) / audio_dur
                confidence = min(1.0, char_rate / 15.0)  # 15 chars/s ≈ confident speech
            else:
                confidence = 0.0 if not text else 0.5

            logger.info(
                f"[Whisper] ✅ {elapsed:.2f}s | {audio_dur:.1f}s audio | "
                f"lang={info.language} conf={confidence:.2f} | {text!r}"
            )
            return {
                "text": text,
                "confidence": confidence,
                "language": info.language,
                "duration_seconds": audio_dur,
            }

        except Exception as exc:
            self.total_failures += 1
            logger.error(f"[Whisper] Transcription error: {exc}", exc_info=True)
            return {"text": "", "confidence": 0.0, "language": self.language, "duration_seconds": 0.0}

    def is_likely_hallucination(self, text: str, confidence: float, conversation_active: bool = False) -> tuple:
        """Compatibility shim matching SpeechRecognizer interface."""
        text_lower = text.lower()
        for frag in _HALLUCINATION_FRAGMENTS:
            if frag in text_lower:
                return True, f"hallucination fragment: {frag!r}"
        if confidence < 0.05 and not conversation_active:
            return True, "very low confidence"
        return False, ""
