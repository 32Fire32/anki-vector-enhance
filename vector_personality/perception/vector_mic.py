"""
Vector Microphone Audio Processor

Streams audio from Vector's built-in 4-microphone array via the SDK's
AudioFeed gRPC endpoint and feeds it into the existing AudioProcessor
VAD pipeline.  This replaces the PC microphone (sounddevice) input so
that speech recognition works without any external mic.

Advantages over a PC mic:
- Audio captured right next to the user interacting with Vector
- Source-direction metadata (12 directions) available per chunk
- No extra hardware needed

Vector's mic delivers ~16 kHz 16-bit mono PCM.  The AudioProcessor
expects the same format, so the data is passed through directly.
"""

import asyncio
import collections
import logging
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Vector delivers 16-bit signed PCM at this rate
VECTOR_MIC_SAMPLE_RATE = 16000

# Ring buffer keeps this many seconds of raw audio for sliding-window STT
_RING_BUFFER_SECONDS = 8
_RING_BUFFER_MAX_BYTES = int(VECTOR_MIC_SAMPLE_RATE * 2 * _RING_BUFFER_SECONDS)


class VectorMicProcessor:
    """Streams audio from Vector's microphones into an AudioProcessor.

    Usage::

        from vector_personality.perception.audio_processor import AudioProcessor
        from vector_personality.perception.vector_mic import VectorMicProcessor

        audio_proc = AudioProcessor(sample_rate=16000)
        mic = VectorMicProcessor(robot, audio_proc)
        mic.start()      # begins streaming in the SDK event loop
        ...
        mic.stop()

    :param robot: A connected ``anki_vector.Robot`` instance.
    :param audio_processor: The AudioProcessor that will receive frames.
    """

    def __init__(self, robot, audio_processor):
        self.robot = robot
        self.audio_processor = audio_processor
        self._running = False
        self._frame_buffer = bytearray()

        # Ring buffer for sliding-window STT (no webrtcvad needed)
        # Stores raw 16-bit PCM chunks for the last _RING_BUFFER_SECONDS of audio.
        self._ring_chunks: collections.deque = collections.deque()  # each element = bytes
        self._ring_bytes: int = 0
        self._ring_lock = threading.Lock()

        # Direction tracking (latest values from Vector)
        self.last_source_direction: int = 12     # 12 = invalid
        self.last_source_confidence: int = 0
        self.last_noise_floor: int = 0

        # Stats
        self.total_chunks_received: int = 0
        self.total_bytes_received: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the microphone audio feed from Vector."""
        if self._running:
            return

        self._running = True
        self._frame_buffer.clear()
        self.total_chunks_received = 0
        self.total_bytes_received = 0

        # Register our callback and start the SDK audio feed
        self.robot.audio.register_audio_callback(self._on_audio_chunk)
        self.robot.audio.init_audio_feed()
        logger.info("🎤 Vector microphone feed started (source-direction enabled)")

    def stop(self) -> None:
        """Stop the microphone audio feed."""
        if not self._running:
            return

        self._running = False
        try:
            self.robot.audio.close_audio_feed()
        except Exception:
            logger.debug("Audio feed close error (expected during shutdown)")
        self.robot.audio.unregister_audio_callback(self._on_audio_chunk)
        logger.info(
            "🎤 Vector microphone feed stopped  "
            f"(chunks={self.total_chunks_received}, bytes={self.total_bytes_received})"
        )

    @property
    def is_active(self) -> bool:
        return self._running and self.robot.audio.is_audio_feed_active

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_audio_chunk(self, chunk) -> None:
        """Called by the SDK AudioFeed for every incoming chunk.

        The chunk.data size varies per gRPC message.  We accumulate bytes
        and slice them into exact 30 ms frames (960 bytes at 16 kHz / 16-bit)
        that webrtcvad expects.
        """
        if not self._running:
            return

        self.total_chunks_received += 1
        self.total_bytes_received += len(chunk.data)

        if self.total_chunks_received == 1:
            logger.info(
                f'[VectorMic] ✅ First chunk: {len(chunk.data)}B '
                f'dir={chunk.source_direction} conf={chunk.source_confidence}'
            )
        elif self.total_chunks_received % 200 == 0:
            logger.debug(
                f'[VectorMic] chunk #{self.total_chunks_received} '
                f'{len(chunk.data)}B buf={len(self._frame_buffer)}B '
                f'dir={chunk.source_direction}'
            )

        # Update direction metadata
        self.last_source_direction = chunk.source_direction
        self.last_source_confidence = chunk.source_confidence
        self.last_noise_floor = chunk.noise_floor_power

        raw = chunk.data

        # ── Ring buffer for sliding-window STT ──────────────────────────
        with self._ring_lock:
            self._ring_chunks.append(raw)
            self._ring_bytes += len(raw)
            # Evict oldest chunks when the buffer exceeds the limit
            while self._ring_bytes > _RING_BUFFER_MAX_BYTES and self._ring_chunks:
                evicted = self._ring_chunks.popleft()
                self._ring_bytes -= len(evicted)

        # ── Legacy webrtcvad path (used for PC mic, kept for compatibility) ──
        self._frame_buffer.extend(raw)

        frame_bytes = self.audio_processor.frame_size * 2  # 16-bit = 2 bytes/sample
        frames_emitted = 0

        while len(self._frame_buffer) >= frame_bytes:
            frame = bytes(self._frame_buffer[:frame_bytes])
            del self._frame_buffer[:frame_bytes]
            self.audio_processor.add_frame(frame)
            frames_emitted += 1

        if self.total_chunks_received <= 3 and frames_emitted:
            logger.debug(f'[VectorMic] Emitted {frames_emitted} VAD frames from chunk #{self.total_chunks_received}')

    def get_audio_window(self, seconds: float = 4.0) -> bytes:
        """Return the most recent *seconds* of raw 16-bit PCM audio.

        Thread-safe.  Returns whatever is buffered if it's less than
        *seconds* worth (i.e. robot just started up).
        """
        n_bytes = int(VECTOR_MIC_SAMPLE_RATE * 2 * seconds)
        with self._ring_lock:
            data = b''.join(self._ring_chunks)
        return data[-n_bytes:] if len(data) >= n_bytes else data

    @property
    def buffered_seconds(self) -> float:
        """How many seconds of audio are currently in the ring buffer."""
        return self._ring_bytes / (VECTOR_MIC_SAMPLE_RATE * 2)

    def direction_label(self) -> str:
        """Return a human-readable label for the last detected sound direction.

        Vector has 12 microphone directions (30° sectors, 0 = front).
        """
        if self.last_source_direction >= 12:
            return "unknown"
        labels = [
            "front",            # 0
            "front-right",      # 1
            "right-front",      # 2
            "right",            # 3
            "right-rear",       # 4
            "rear-right",       # 5
            "rear",             # 6
            "rear-left",        # 7
            "left-rear",        # 8
            "left",             # 9
            "left-front",       # 10
            "front-left",       # 11
        ]
        return labels[self.last_source_direction]
