"""
Audio Processing Module

Implements Voice Activity Detection (VAD) using webrtcvad.
Manages circular buffer for audio streaming and silence detection.

Key Features:
- Real-time VAD with 30ms frame processing
- Circular buffer with configurable size
- Silence detection with 300ms threshold
- WAV export for speech recognition

Performance Target: <30ms per frame (real-time capable)
"""

import webrtcvad
import numpy as np
import wave
import collections
from typing import Optional, List
from datetime import datetime, timedelta
import logging
import sounddevice as sd
import threading

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor with Voice Activity Detection
    
    Uses WebRTC VAD for speech detection and maintains a circular
    buffer for audio capture. Designed for real-time processing
    with Vector's microphone input.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_mode: int = 0,          # 0=permissive … 3=aggressive. Use 0 for Vector's mic (noisy motors).
        max_buffer_seconds: int = 30
    ):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Audio sample rate in Hz (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
            vad_mode: VAD aggressiveness (0-3, higher = more aggressive)
            max_buffer_seconds: Maximum buffer size in seconds
        
        Raises:
            ValueError: If parameters are invalid for webrtcvad
        """
        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Invalid sample_rate {sample_rate}. Must be 8000, 16000, 32000, or 48000")
        
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Invalid frame_duration_ms {frame_duration_ms}. Must be 10, 20, or 30")
        
        if not 0 <= vad_mode <= 3:
            raise ValueError(f"Invalid vad_mode {vad_mode}. Must be 0-3")
        
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_mode = vad_mode
        
        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_mode)
        
        # Circular buffer (stores audio frames)
        max_frames = int(max_buffer_seconds * 1000 / frame_duration_ms)
        self.max_buffer_size = max_frames
        self.buffer = collections.deque(maxlen=max_frames)
        
        # Silence detection
        self.silence_threshold_ms = 1200  # 1.2s — bridges gaps between words on BT/USB mics
        self.silence_frames = int(self.silence_threshold_ms / frame_duration_ms)
        self.consecutive_silence_count = 0

        # Pre-roll: include audio before VAD onset so we don't clip the speech start.
        # 600ms covers the quiet onset consonants (h, s, f) that appear before the voiced vowel.
        self.pre_roll_ms = 600
        
        # Speech detection threshold (require multiple consecutive speech frames)
        self.min_speech_frames = 2  # Lower = more sensitive. Increase if too many false positives.
        self.consecutive_speech_count = 0
        
        # Recording state
        self.is_recording = False
        self.recording_start_time: Optional[datetime] = None
        self.stream = None
        self.speech_detected = False
        self.speech_buffer = []  # Buffer for current utterance
        self.completed_utterances = collections.deque()  # Queue for completed utterances
        self.max_energy = 0.0  # Track max energy for diagnostics
        
        # TTS muting (prevents feedback loop from Vector's own speech)
        self.is_muted = False  # When True, ignores all audio input
        
        logger.info(
            f"AudioProcessor initialized: {sample_rate}Hz, {frame_duration_ms}ms frames, "
            f"VAD mode {vad_mode}, buffer {max_buffer_seconds}s"
        )

    def start_listening(self, device=None):
        """
        Start microphone stream in background.
        
        Args:
            device: Device index or None for default. Use list_devices() to see options.
        """
        if self.stream:
            return
            
        try:
            # Determine how many channels the device actually supports.
            # Many BT headsets only expose stereo (2-ch) input even though we want mono.
            dev_idx = device if device is not None else sd.default.device[0]
            dev_info = sd.query_devices(dev_idx, 'input')
            max_ch = int(dev_info.get('max_input_channels', 1))
            channels = 1 if max_ch >= 1 else max_ch  # prefer mono; BT fallback handled below
            try:
                self.stream = sd.InputStream(
                    device=device,
                    channels=channels,
                    samplerate=self.sample_rate,
                    dtype='int16',
                    blocksize=self.frame_size,
                    callback=self._audio_callback
                )
            except Exception:
                # If mono fails (e.g. BT headset requires stereo), open with native channel count
                channels = max_ch or 2
                logger.warning(f"Mono open failed for device {dev_idx}, retrying with {channels} channels")
                self.stream = sd.InputStream(
                    device=device,
                    channels=channels,
                    samplerate=self.sample_rate,
                    dtype='int16',
                    blocksize=self.frame_size,
                    callback=self._audio_callback
                )
            self._open_channels = channels  # remember for downmix in callback
            self.stream.start()
            self.is_recording = True
            logger.info(f"🎤 Microphone listening started: {dev_info['name']} ({channels}ch)")
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self.is_recording = False  # ensure flag is clear so retries don't re-enter
    
    @staticmethod
    def list_devices():
        """List all available audio input devices."""
        print("\n" + "="*60)
        print("AVAILABLE MICROPHONES:")
        print("="*60)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default = " [DEFAULT]" if i == sd.default.device[0] else ""
                print(f"  {i}: {dev['name']}{default}")
        print("="*60 + "\n")

    def stop_listening(self):
        """Stop microphone stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_recording = False
            logger.info("🎤 Microphone listening stopped")

    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice InputStream."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Downmix to mono if device was opened with >1 channels
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1, keepdims=True).astype(np.int16)
        else:
            mono = indata
        
        # Calculate energy for diagnostics
        energy = np.sqrt(np.mean(mono.astype(np.float32) ** 2))
        if energy > self.max_energy:
            self.max_energy = energy
            
        # indata is numpy array (frames, channels)
        # Convert to bytes for webrtcvad
        audio_bytes = mono.tobytes()
        
        # Process frame
        self.add_frame(audio_bytes)

    def is_speech(self, frame: bytes) -> bool:
        """
        Check if audio frame contains speech
        
        Args:
            frame: Raw audio frame (16-bit PCM, little-endian)
        
        Returns:
            True if frame contains speech, False otherwise
        """
        try:
            result = self.vad.is_speech(frame, self.sample_rate)
            # Log first 20 raw VAD results so we can see what webrtcvad thinks
            if not hasattr(self, '_vad_calls'):
                self._vad_calls = 0
            self._vad_calls += 1
            if self._vad_calls <= 20:
                logger.info(f'[VAD] call #{self._vad_calls}: is_speech={result} frame_len={len(frame)}B')
            return result
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False

    def add_frame(self, frame: bytes) -> bool:
        """
        Add audio frame to buffer and manage speech segments.
        """
        # Skip speech detection when muted (e.g., during TTS playback)
        if self.is_muted:
            return False

        # Track total frames received for diagnostics
        if not hasattr(self, '_total_frames'):
            self._total_frames = 0
        self._total_frames += 1

        # Calculate audio energy to filter out weak/distant audio
        audio_array = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        energy = np.sqrt(np.mean(audio_array ** 2))

        # Track max energy for the monitoring window
        if energy > self.max_energy:
            self.max_energy = energy

        # Log energy every 100 frames (~3s) to help diagnose mic issues
        if self._total_frames % 100 == 0:
            logger.debug(
                f'[VAD] frame #{self._total_frames}: '
                f'energy={energy:.1f} max={self.max_energy:.1f} '
                f'speech={self.speech_detected} sil={self.consecutive_silence_count}'
            )
        elif self._total_frames <= 10:
            # Log first 10 frames at INFO to confirm data is flowing
            logger.info(f'[VAD] First frames flowing: #{self._total_frames} energy={energy:.1f}')

        # Filter out very weak audio (likely TV/distant conversations)
        # Threshold: 200 = very weak background hiss. Adjust based on mic noise floor.
        # Lower if Vector's mic noise floor is high (e.g. 707).
        MIN_ENERGY_THRESHOLD = 50.0

        if energy < MIN_ENERGY_THRESHOLD:
            logger.debug(f'[VAD] Frame below energy threshold: {energy:.1f} < {MIN_ENERGY_THRESHOLD}')
            # Too weak, treat as silence
            # If we're in a speech segment, this counts as silence frame
            if self.speech_detected:
                self.consecutive_silence_count += 1
                self.speech_buffer.append(frame)  # Keep frame for continuity
                
                # Check if silence timeout reached (end of utterance)
                if self.consecutive_silence_count >= self.silence_frames:
                    logger.info("🛑 Speech ended (low energy)")
                    self.speech_detected = False
                    
                    # Move complete utterance to queue if it has enough speech
                    if self.speech_buffer and len(self.speech_buffer) > self.silence_frames:
                        # Remove trailing silence frames
                        trimmed = self.speech_buffer[:-self.silence_frames]
                        complete_audio = b''.join(trimmed)
                        self.completed_utterances.append(complete_audio)
                        logger.info(f"📦 Utterance queued ({len(complete_audio)} bytes)")
                    
                    self.speech_buffer = []
                    self.consecutive_silence_count = 0
                    self.consecutive_speech_count = 0
            else:
                # Not in speech, just reset counters
                self.consecutive_speech_count = 0
            
            # Add to circular buffer for context
            self.buffer.append(frame)
            return False
        
        # Check for speech
        has_speech = self.is_speech(frame)
        
        # Add to circular buffer (always)
        self.buffer.append(frame)
        
        # Logic for speech segment capture
        if has_speech:
            self.consecutive_speech_count += 1
            
            # Only activate speech detection after minimum consecutive frames
            if not self.speech_detected and self.consecutive_speech_count >= self.min_speech_frames:
                logger.info(f"🗣️ Speech started (confirmed after {self.min_speech_frames * self.frame_duration_ms}ms)")
                self.speech_detected = True
                # Capture a small pre-roll from the circular buffer so we don't lose the start of speech
                try:
                    pre_roll_frames = max(0, int(self.pre_roll_ms / self.frame_duration_ms) - 1)
                    buffered = list(self.buffer)
                    if pre_roll_frames > 0 and len(buffered) > 0:
                        # Take the last N frames (excluding current frame which will be appended below)
                        self.speech_buffer = buffered[-pre_roll_frames:]
                    else:
                        self.speech_buffer = []
                except Exception:
                    self.speech_buffer = []
            
            self.consecutive_silence_count = 0
            if self.speech_detected:
                self.speech_buffer.append(frame)
            
        else:
            # Silence detected - reset speech counter
            self.consecutive_speech_count = 0
            
            if self.speech_detected:
                # We are in a speech segment but this frame is silence
                self.consecutive_silence_count += 1
                self.speech_buffer.append(frame)
                
                # Check if silence timeout reached (end of utterance)
                if self.consecutive_silence_count >= self.silence_frames:
                    logger.info("🛑 Speech ended (silence timeout)")
                    self.speech_detected = False
                    
                    # Move complete utterance to queue
                    if self.speech_buffer and len(self.speech_buffer) > self.silence_frames:
                        # Remove trailing silence
                        utterance = b''.join(self.speech_buffer[:-self.silence_frames])
                        self.completed_utterances.append(utterance)
                        logger.info(f"📦 Utterance queued ({len(utterance)} bytes)")
                    
                    self.speech_buffer = []
                    self.consecutive_silence_count = 0
        
        return has_speech

    def get_last_utterance(self) -> Optional[bytes]:
        """
        Retrieve and clear the last complete utterance if available.
        Returns None if no complete utterance is ready.
        """
        if self.completed_utterances:
            return self.completed_utterances.popleft()
        return None
    
    def mute(self) -> None:
        """
        Mute audio input (e.g., during TTS playback to prevent feedback loop).
        When muted, speech detection is disabled but stream continues running.
        """
        self.is_muted = True
    
    def unmute(self) -> None:
        """
        Unmute audio input (after TTS playback complete).
        Resumes normal speech detection.
        """
        self.is_muted = False
        # Reset detection state to avoid partial utterances
        self.speech_detected = False
        self.speech_buffer = []
        self.consecutive_speech_count = 0
        self.consecutive_silence_count = 0

    def check_silence_timeout(self) -> bool:
        """
        Check if silence threshold has been exceeded
        
        Returns:
            True if silence duration >= 300ms
        """
        return self.consecutive_silence_count >= self.silence_frames

    def start_recording(self) -> None:
        """Start audio recording session"""
        self.is_recording = True
        self.recording_start_time = datetime.now()
        self.consecutive_silence_count = 0
        logger.info("Audio recording started")

    def stop_recording(self) -> None:
        """Stop audio recording session"""
        self.is_recording = False
        duration = None
        if self.recording_start_time:
            duration = (datetime.now() - self.recording_start_time).total_seconds()
        logger.info(f"Audio recording stopped (duration: {duration:.2f}s)")

    def clear_buffer(self) -> None:
        """Clear audio buffer"""
        self.buffer.clear()
        self.consecutive_silence_count = 0

    def discard_pending_utterances(self) -> None:
        """
        Discard any utterances queued for transcription and reset in-progress
        speech detection state — WITHOUT wiping the pre-roll ring buffer.

        Call this just before TTS playback so stale audio captured during LLM
        processing is dropped, but the ring buffer stays intact so pre-roll
        works correctly the moment the user starts speaking again after TTS.
        """
        self.completed_utterances.clear()
        # Reset in-progress speech capture so we start fresh after TTS
        self.speech_detected = False
        self.speech_buffer = []
        self.consecutive_speech_count = 0
        self.consecutive_silence_count = 0

    def export_wav(self, output_path: str) -> int:
        """
        Export buffered audio to WAV file
        
        Args:
            output_path: Path to output WAV file
        
        Returns:
            Number of frames written
        
        Raises:
            ValueError: If buffer is empty
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot export empty buffer")
        
        # Concatenate all frames
        audio_data = b''.join(self.buffer)
        
        # Write WAV file
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        
        frame_count = len(self.buffer)
        duration = frame_count * self.frame_duration_ms / 1000
        logger.info(f"Exported {frame_count} frames ({duration:.2f}s) to {output_path}")
        
        return frame_count

    def get_buffer_duration(self) -> float:
        """
        Get current buffer duration in seconds
        
        Returns:
            Buffer duration in seconds
        """
        return len(self.buffer) * self.frame_duration_ms / 1000

    def get_buffer_frames(self) -> List[bytes]:
        """
        Get list of all buffered frames
        
        Returns:
            List of audio frames
        """
        return list(self.buffer)

    def detect_speech_segments(self) -> List[tuple]:
        """
        Detect speech segments in buffered audio
        
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, frame in enumerate(self.buffer):
            has_speech = self.is_speech(frame)
            timestamp = i * self.frame_duration_ms / 1000
            
            if has_speech and not in_speech:
                # Speech segment started
                speech_start = timestamp
                in_speech = True
            elif not has_speech and in_speech:
                # Speech segment ended
                segments.append((speech_start, timestamp))
                in_speech = False
        
        # Handle case where speech continues to end of buffer
        if in_speech:
            end_time = len(self.buffer) * self.frame_duration_ms / 1000
            segments.append((speech_start, end_time))
        
        logger.debug(f"Detected {len(segments)} speech segments")
        return segments

    def get_energy_level(self, frame: bytes) -> float:
        """
        Calculate energy level of audio frame
        
        Args:
            frame: Raw audio frame
        
        Returns:
            RMS energy level (0.0 - 1.0)
        """
        # Convert bytes to numpy array
        audio_data = np.frombuffer(frame, dtype=np.int16)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        # Normalize to 0-1 (int16 max = 32768)
        normalized = rms / 32768.0
        
        return normalized

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"AudioProcessor("
            f"sample_rate={self.sample_rate}, "
            f"frame_duration_ms={self.frame_duration_ms}, "
            f"vad_mode={self.vad_mode}, "
            f"buffer_frames={len(self.buffer)}, "
            f"is_recording={self.is_recording})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_audio_processor(config: dict = None) -> AudioProcessor:
    """
    Factory function to create AudioProcessor with config
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured AudioProcessor instance
    """
    if config is None:
        config = {}
    
    return AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        frame_duration_ms=config.get('frame_duration_ms', 30),
        vad_mode=config.get('vad_mode', 3),
        max_buffer_seconds=config.get('max_buffer_seconds', 30)
    )


def pcm_to_numpy(frame: bytes, dtype=np.int16) -> np.ndarray:
    """
    Convert PCM bytes to numpy array
    
    Args:
        frame: Raw PCM audio frame
        dtype: NumPy data type (default: int16)
    
    Returns:
        NumPy array
    """
    return np.frombuffer(frame, dtype=dtype)


def numpy_to_pcm(array: np.ndarray) -> bytes:
    """
    Convert numpy array to PCM bytes
    
    Args:
        array: NumPy array (int16)
    
    Returns:
        Raw PCM bytes
    """
    return array.astype(np.int16).tobytes()
