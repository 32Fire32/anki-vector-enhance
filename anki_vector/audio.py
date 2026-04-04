# Copyright (c) 2019 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support for accessing Vector's audio.

Vector's speakers can be used for playing user-provided audio.
Vector's microphones can be streamed via the AudioFeed gRPC endpoint,
providing mono 16-bit PCM audio at ~16 kHz together with source-direction
metadata (12 directions, 0-11).

The :class:`AudioComponent` class defined in this module is made available as
:attr:`anki_vector.robot.Robot.audio` and can be used to play audio data on the robot
and to stream microphone audio from the robot.
"""

# __all__ should order by constants, event classes, other classes, functions.
__all__ = ['AudioComponent', 'EvtAudioChunk']

import asyncio
import collections
from concurrent import futures
from enum import Enum
import struct
import time
import wave
from google.protobuf.text_format import MessageToString
from . import util
from .connection import on_connection_thread
from .events import Events
from .exceptions import VectorExternalAudioPlaybackException
from .messaging import protocol


# Vector's processed microphone sample rate
MICROPHONE_SAMPLE_RATE = 16000


class EvtAudioChunk:
    """An event containing a chunk of audio samples from Vector's microphones.

    :param data: Raw 16-bit signed LE PCM audio bytes.
    :param source_direction: Detected sound direction (0-11, 12 = invalid).
    :param source_confidence: Confidence of direction detection.
    :param noise_floor_power: Background noise power level.
    :param robot_time_stamp: Robot-side timestamp.
    :param group_id: Streaming group index.
    """

    def __init__(self, data: bytes, source_direction: int, source_confidence: int,
                 noise_floor_power: int, robot_time_stamp: int = 0, group_id: int = 0):
        self.data = data
        self.source_direction = source_direction
        self.source_confidence = source_confidence
        self.noise_floor_power = noise_floor_power
        self.robot_time_stamp = robot_time_stamp
        self.group_id = group_id

    @property
    def sample_count(self) -> int:
        """Number of 16-bit samples in this chunk."""
        return len(self.data) // 2


MAX_ROBOT_AUDIO_CHUNK_SIZE = 1024  # 1024 is maximum, larger sizes will fail
DEFAULT_FRAME_SIZE = MAX_ROBOT_AUDIO_CHUNK_SIZE // 2


class RobotVolumeLevel(Enum):
    """Use these values for setting the master audio volume.  See :meth:`set_master_volume`

    Note that muting the robot is not supported from the SDK.
    """
    LOW = 0
    MEDIUM_LOW = 1
    MEDIUM = 2
    MEDIUM_HIGH = 3
    HIGH = 4


class AudioComponent(util.Component):
    """Handles audio on Vector.

    The AudioComponent object plays audio data to Vector's speaker.

    The :class:`anki_vector.robot.Robot` or :class:`anki_vector.robot.AsyncRobot` instance
    owns this audio component.

    .. testcode::

        import anki_vector

        with anki_vector.Robot() as robot:
            robot.audio.stream_wav_file('../examples/sounds/vector_alert.wav')
    """

    def __init__(self, robot):
        super().__init__(robot)
        self._is_shutdown = False
        # don't create asyncio.Events here, they are not thread-safe
        self._is_active_event = None
        self._done_event = None

        # Microphone feed state
        self._audio_feed_task: asyncio.Task = None
        self._audio_feed_enabled = False
        self._latest_audio_chunk: EvtAudioChunk = None
        self._audio_callbacks = []

    @on_connection_thread(requires_control=False)
    async def set_master_volume(self, volume: RobotVolumeLevel) -> protocol.MasterVolumeResponse:
        """Sets Vector's master volume level.

        Note that muting the robot is not supported from the SDK.

        .. testcode::

            import anki_vector
            from anki_vector import audio

            with anki_vector.Robot(behavior_control_level=None) as robot:
                robot.audio.set_master_volume(audio.RobotVolumeLevel.MEDIUM_HIGH)

        :param volume: the robot's desired volume
        """

        volume_request = protocol.MasterVolumeRequest(volume_level=volume.value)
        return await self.conn.grpc_interface.SetMasterVolume(volume_request)

    def _open_file(self, filename):
        _reader = wave.open(filename, 'rb')
        _params = _reader.getparams()
        self.logger.info("Playing audio file %s", filename)

        if _params.sampwidth != 2 or _params.nchannels != 1 or _params.framerate > 16025 or _params.framerate < 8000:
            raise VectorExternalAudioPlaybackException(
                f"Audio format must be 8000-16025 hz, 16 bits, 1 channel.  "
                f"Found {_params.framerate} hz/{_params.sampwidth*8} bits/{_params.nchannels} channels")

        return _reader, _params

    async def _request_handler(self, reader, params, volume):
        """Handles generating request messages for the AudioPlaybackStream."""
        frames = params.nframes  # 16 bit samples, not bytes

        # send preparation message
        msg = protocol.ExternalAudioStreamPrepare(audio_frame_rate=params.framerate, audio_volume=volume)
        msg = protocol.ExternalAudioStreamRequest(audio_stream_prepare=msg)

        yield msg
        await asyncio.sleep(0)  # give event loop a chance to process messages

        # count of full and partial chunks
        total_chunks = (frames + DEFAULT_FRAME_SIZE - 1) // DEFAULT_FRAME_SIZE
        curr_chunk = 0
        start_time = time.time()
        self.logger.debug("Starting stream time %f", start_time)

        while frames > 0 and not self._done_event.is_set():
            read_count = min(frames, DEFAULT_FRAME_SIZE)
            audio_data = reader.readframes(read_count)
            msg = protocol.ExternalAudioStreamChunk(audio_chunk_size_bytes=len(audio_data), audio_chunk_samples=audio_data)
            msg = protocol.ExternalAudioStreamRequest(audio_stream_chunk=msg)
            yield msg
            await asyncio.sleep(0)

            # check if streaming is way ahead of audio playback time
            elapsed = time.time() - start_time
            expected_data_count = elapsed * params.framerate
            time_ahead = (curr_chunk * DEFAULT_FRAME_SIZE - expected_data_count) / params.framerate
            if time_ahead > 1.0:
                self.logger.debug("waiting %f to catchup chunk %f", time_ahead - 0.5, curr_chunk)
                await asyncio.sleep(time_ahead - 0.5)
            frames = frames - read_count
            curr_chunk += 1
            if curr_chunk == total_chunks:
                # last chunk:  time to stop stream
                msg = protocol.ExternalAudioStreamComplete()
                msg = protocol.ExternalAudioStreamRequest(audio_stream_complete=msg)

                yield msg
                await asyncio.sleep(0)

        reader.close()

        # Need the done message from the robot
        await self._done_event.wait()
        self._done_event.clear()

    @on_connection_thread(requires_control=True)
    async def stream_wav_file(self, filename, volume=50):
        """ Plays audio using Vector's speakers.

        .. testcode::

            import anki_vector

            with anki_vector.Robot() as robot:
                robot.audio.stream_wav_file('../examples/sounds/vector_alert.wav')

        :param filename: the filename/path to the .wav audio file
        :param volume: the audio playback level (0-100)
        """

        # TODO make this support multiple simultaneous sound playback
        if self._is_active_event is None:
            self._is_active_event = asyncio.Event()

        if self._is_active_event.is_set():
            raise VectorExternalAudioPlaybackException("Cannot start audio when another sound is playing")

        if volume < 0 or volume > 100:
            raise VectorExternalAudioPlaybackException("Volume must be between 0 and 100")
        _file_reader, _file_params = self._open_file(filename)
        playback_error = None
        self._is_active_event.set()

        if self._done_event is None:
            self._done_event = asyncio.Event()

        try:
            async for response in self.grpc_interface.ExternalAudioStreamPlayback(self._request_handler(_file_reader, _file_params, volume)):
                self.logger.info("ExternalAudioStream %s", MessageToString(response, as_one_line=True))
                response_type = response.WhichOneof("audio_response_type")
                if response_type == 'audio_stream_playback_complete':
                    playback_error = None
                elif response_type == 'audio_stream_buffer_overrun':
                    playback_error = response_type
                elif response_type == 'audio_stream_playback_failyer':
                    playback_error = response_type
                self._done_event.set()
        except asyncio.CancelledError:
            self.logger.debug('Audio Stream future was cancelled.')
        except futures.CancelledError:
            self.logger.debug('Audio Stream handler task was cancelled.')
        finally:
            self._is_active_event = None
            self._done_event = None

        if playback_error is not None:
            raise VectorExternalAudioPlaybackException(f"Error reported during audio playback {playback_error}")

    # ------------------------------------------------------------------
    # Microphone feed (streaming audio FROM Vector's built-in microphones)
    # ------------------------------------------------------------------

    def init_audio_feed(self) -> None:
        """Begin streaming audio from Vector's microphones.

        Audio chunks are delivered as :class:`EvtAudioChunk` events via any
        registered callbacks (see :meth:`register_audio_callback`).

        Each chunk contains raw 16-bit signed LE PCM data at ~16 kHz together
        with source-direction metadata.

        .. code-block:: python

            import anki_vector

            def on_audio(chunk):
                print(f"Got {len(chunk.data)} bytes from direction {chunk.source_direction}")

            with anki_vector.Robot(enable_audio_feed=True) as robot:
                robot.audio.register_audio_callback(on_audio)
                robot.audio.init_audio_feed()
                import time; time.sleep(10)
                robot.audio.close_audio_feed()
        """
        if not self._audio_feed_task or self._audio_feed_task.done():
            self._audio_feed_enabled = True
            self._audio_feed_task = self.conn.loop.create_task(self._request_and_handle_audio())

    def close_audio_feed(self) -> None:
        """Stop streaming audio from Vector's microphones."""
        if self._audio_feed_task:
            self._audio_feed_enabled = False
            self._audio_feed_task.cancel()
            future = self.conn.run_coroutine(self._audio_feed_task)
            try:
                future.result()
            except (futures.CancelledError, asyncio.CancelledError):
                self.logger.debug('Audio feed task was cancelled. This is expected during disconnection.')
            self._audio_feed_task = None

    def register_audio_callback(self, callback) -> None:
        """Register a callable to receive :class:`EvtAudioChunk` objects.

        :param callback: A callable that accepts one positional argument (an EvtAudioChunk).
        """
        self._audio_callbacks.append(callback)

    def unregister_audio_callback(self, callback) -> None:
        """Remove a previously registered audio callback."""
        try:
            self._audio_callbacks.remove(callback)
        except ValueError:
            pass

    @property
    def latest_audio_chunk(self) -> EvtAudioChunk:
        """The most recently received audio chunk, or None."""
        return self._latest_audio_chunk

    @property
    def is_audio_feed_active(self) -> bool:
        """True if the microphone audio feed is currently streaming."""
        return self._audio_feed_enabled and self._audio_feed_task is not None and not self._audio_feed_task.done()

    def _unpack_audio(self, msg) -> None:
        """Convert a gRPC AudioFeedResponse into an EvtAudioChunk and dispatch."""
        chunk = EvtAudioChunk(
            data=msg.signal_power,
            source_direction=msg.source_direction,
            source_confidence=msg.source_confidence,
            noise_floor_power=msg.noise_floor_power,
            robot_time_stamp=msg.robot_time_stamp,
            group_id=msg.group_id,
        )

        # On first chunk: dump raw bytes so we can verify the data format
        if not hasattr(self, '_audio_first_chunk_logged'):
            self._audio_first_chunk_logged = True
            raw = msg.signal_power
            import struct
            import math
            samples_i16 = struct.unpack_from(f'<{len(raw)//2}h', raw)
            unique = len(set(samples_i16))
            mn, mx = min(samples_i16), max(samples_i16)
            rms = math.sqrt(sum(s*s for s in samples_i16) / len(samples_i16)) if samples_i16 else 0
            self.logger.info(
                f'[AudioFeed] DATA DIAGNOSIS: {len(raw)}B → {len(samples_i16)} int16 samples '
                f'min={mn} max={mx} rms={rms:.1f} unique_values={unique} '
                f'noise_floor={msg.noise_floor_power} '
                f'first8_hex={raw[:16].hex()}'
            )
            if unique < 5:
                self.logger.warning(
                    '[AudioFeed] ⚠️  Only %d unique sample values — data may NOT be raw PCM! '
                    'signal_power might be amplitude/power envelope, not audio samples.', unique
                )

        self._latest_audio_chunk = chunk
        for cb in self._audio_callbacks:
            try:
                cb(chunk)
            except Exception:
                self.logger.exception("Error in audio callback")

    async def _request_and_handle_audio(self) -> None:
        """Query and listen for audio feed events from the robot."""
        try:
            req = protocol.AudioFeedRequest()
            self.logger.info('[AudioFeed] Sending AudioFeedRequest to Vector...')
            chunk_count = 0
            async for evt in self.grpc_interface.AudioFeed(req):
                if not self._audio_feed_enabled:
                    self.logger.info('[AudioFeed] Feed disabled, stopping stream.')
                    return
                chunk_count += 1
                if chunk_count == 1:
                    self.logger.info(
                        '[AudioFeed] ✅ First audio chunk received! '
                        f'data={len(evt.signal_power)}B dir={evt.source_direction} '
                        f'conf={evt.source_confidence}'
                    )
                elif chunk_count % 500 == 0:
                    self.logger.debug(
                        f'[AudioFeed] Chunk #{chunk_count}: '
                        f'{len(evt.signal_power)}B dir={evt.source_direction}'
                    )
                self._unpack_audio(evt)
            self.logger.warning(
                f'[AudioFeed] Stream ended after {chunk_count} chunks. '
                'Wire-pod may not support AudioFeed — check if AUDIO_SOURCE=pc fallback is needed.'
            )
        except asyncio.CancelledError:
            self.logger.debug('[AudioFeed] Task cancelled (expected during disconnection).')
        except Exception as exc:
            self.logger.error(
                f'[AudioFeed] Stream error: {exc!r}. '
                'Wire-pod may not implement this gRPC endpoint. '
                'Set AUDIO_SOURCE=pc to use PC microphone instead.',
                exc_info=True
            )
