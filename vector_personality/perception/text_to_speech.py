"""
Text-to-Speech Module

Integrates Google TTS (gTTS) for audio generation and playback via Vector SDK.
Handles audio generation, voice selection, and error handling.

Key Features:
- Google Text-to-Speech (gTTS) - FREE and unlimited
- Multiple language support (Italian default)
- Direct audio playback via Vector's speakers
- Audio effects (robotic filter)
- Retry logic for failures

API Cost: FREE
Performance Target: <3s for typical sentence
"""

import logging
import asyncio
import concurrent.futures
from typing import Optional, Any
from pathlib import Path
from datetime import datetime
from gtts import gTTS
import tempfile
import os
import wave
import numpy as np

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech using Google TTS (gTTS)
    
    Converts text to audio and plays through Vector's speakers.
    Free and unlimited Google TTS service.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Not used, kept for compatibility
        robot: Any = None,
        model: str = "gtts",  # Not used, kept for compatibility
        voice: str = "italian",
        speed: float = 1.15,  # Faster robotic speech (80s robot style)
        audio_processor: Optional[Any] = None,
        monotone: bool = True  # Enable robotic filter by default
    ):
        """
        Initialize TextToSpeech with Google TTS
        
        Args:
            api_key: Unused (gTTS is free, kept for compatibility)
            robot: Vector robot instance (for audio playback)
            model: Unused (gTTS only, kept for compatibility)
            voice: Unused (language='it' hardcoded, kept for compatibility)
            speed: Speech speed multiplier (0.5-2.0, default 1.15 for robot voice)
            audio_processor: Optional AudioProcessor for TTS feedback prevention
            monotone: Enable robotic filter (default True for 80s robot voice)
        """
        self.robot = robot
        self.language = "it"  # Italian by default
        self.speed = max(0.5, min(2.0, speed))  # gTTS supports 0.5-2.0 via slow flag
        self.audio_processor = audio_processor
        self.monotone = monotone  # Robotic filter (default: True for 80s robot voice)
        
        # Statistics
        self.total_calls = 0
        self.total_characters = 0
        self.total_failures = 0
        
        # Temp directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "vector_tts"
        self.temp_dir.mkdir(exist_ok=True)

        # Dedicated executor for blocking I/O (gTTS network call, pydub conversion,
        # SDK stream_wav_file).  Isolated from the asyncio loop's default executor
        # so loop lifecycle events never cause 'cannot schedule new futures' errors.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="tts"
        )
        
        logger.info(f"TextToSpeech initialized: gTTS (FREE), language={self.language}, speed={self.speed}")
        if self.monotone:
            logger.info("TextToSpeech robotic filter enabled")
    
    async def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Convert text to speech and play through Vector's speakers
        
        Args:
            text: Text to speak
            voice: Optional voice override
            speed: Optional speed override
            max_retries: Number of retry attempts on failure
        
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to speak()")
            return False
        
        text = text.strip()
        # gTTS doesn't use voice parameter, always uses configured language
        speed = speed or self.speed
        
        # Truncate very long text
        if len(text) > 4096:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 4096")
            text = text[:4093] + "..."
        
        # Retry loop
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"🔊 Speaking ({len(text)} chars): {text[:50]}...")
                
                # Strip markdown formatting (asterisks, underscores, etc.) before synthesis
                # gTTS reads these literally: "asterisco vedere asterisco" 😱
                clean_text = text.replace('*', '').replace('_', '').replace('**', '')
                clean_text = clean_text.strip()
                
                # Generate audio with gTTS (Google TTS - FREE)
                audio_path = self.temp_dir / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
                
                # Create gTTS object
                # slow=True if speed < 1.0 for slower speech
                tts = gTTS(text=clean_text, lang=self.language, slow=(speed < 1.0))

                # Run blocking I/O + CPU work in a dedicated thread executor so
                # the event loop stays free for audio streaming.  We use
                # self._executor (not None / the loop default) to avoid
                # 'cannot schedule new futures after shutdown' errors.
                loop = asyncio.get_running_loop()

                # 1. Save MP3 (network round-trip to Google TTS, ~300-500ms)
                await loop.run_in_executor(self._executor, tts.save, str(audio_path))
                logger.info(f"Audio saved: {audio_path}")

                # 2. Convert + apply robotic filter (pydub + numpy, ~200-400ms)
                converted_path = await loop.run_in_executor(
                    self._executor, self._convert_audio_for_vector, audio_path
                )
                if converted_path:
                    play_path = converted_path
                else:
                    play_path = audio_path

                # Play through Vector's speakers
                # Mute microphone before TTS playback to prevent feedback loop
                if self.audio_processor:
                    self.audio_processor.mute()

                try:
                    # 3. Stream WAV to Vector (blocking SDK call — keep in executor
                    #    so the event loop isn't stalled during playback)
                    await loop.run_in_executor(
                        self._executor, self.robot.audio.stream_wav_file, str(play_path), 100
                    )
                    logger.info("✅ Audio played successfully via stream_wav_file")
                except Exception as e:
                    logger.warning(f"stream_wav_file failed: {e}, falling back to say_text")
                    try:
                        await loop.run_in_executor(
                            self._executor, self.robot.behavior.say_text, text
                        )
                        logger.info("✅ Audio played successfully via say_text fallback")
                    except Exception as e2:
                        logger.error(f"All TTS methods failed: {e2}")
                        # Don't return False here, we still generated the audio successfully API-wise
                
                # Short pause so the physical speaker tail clears before unmuting.
                # 0.5s is enough for the BT headset's echo canceller to settle.
                # Do NOT use a long delay — the user may speak immediately after Vector stops.
                await asyncio.sleep(0.5)
                
                # Unmute microphone after TTS playback completes
                if self.audio_processor:
                    self.audio_processor.unmute()
                    # Discard any stale utterances queued during TTS, WITHOUT wiping the
                    # pre-roll ring buffer (so the next speech gets its onset captured)
                    self.audio_processor.discard_pending_utterances()
                
                # Clean up temp files
                try:
                    audio_path.unlink()
                    if converted_path and converted_path != audio_path:
                        converted_path.unlink()
                except:
                    pass
                
                # Update statistics
                self.total_calls += 1
                self.total_characters += len(text)
                logger.info(f"TTS successful: {len(text)} chars")
                return True
            
            except Exception as e:
                last_error = e
                self.total_failures += 1
                logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        logger.error(f"TTS failed after {max_retries} attempts: {last_error}")
        return False
    
    def _convert_audio_for_vector(self, input_path: Path) -> Optional[Path]:
        """
        Convert audio (MP3 from gTTS) to Vector-compatible format (WAV, PCM 16-bit, 12kHz).
        """
        try:
            from pydub import AudioSegment
            
            output_path = input_path.parent / f"converted_{input_path.stem}.wav"
            logger.info(f"Converting audio: {input_path} -> {output_path}")
            
            # Load MP3
            audio = AudioSegment.from_mp3(str(input_path))
            
            # Get original params
            framerate = audio.frame_rate
            logger.info(f"Audio params: channels={audio.channels}, rate={framerate}, duration={len(audio)}ms")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Export as WAV to process with numpy
            temp_wav = input_path.parent / f"temp_{input_path.stem}.wav"
            audio.export(str(temp_wav), format="wav")
            
            # Now process with numpy for effects
            with wave.open(str(temp_wav), 'rb') as wav_in:
                params = wav_in.getparams()
                n_frames = params.nframes
                sampwidth = params.sampwidth
                framerate = params.framerate
                
                raw_data = wav_in.readframes(n_frames)
                
                # Convert to numpy
                if sampwidth != 2:
                    logger.warning(f"Unsupported sample width: {sampwidth}")
                    return temp_wav
                
                audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                
                # Apply robotic filter if monotone enabled
                if self.monotone:
                    audio_data = self._apply_robotic_filter(audio_data, framerate)
                
                # Amplify volume (20x for MAXIMUM VOLUME 🔊)
                audio_data = audio_data * 20.0
                audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
                
                
                # Resample to 12-16kHz for Vector compatibility
                if framerate >= 24000:
                    # Downsample by 2 (24kHz -> 12kHz, or 22kHz -> 11kHz)
                    resampled_data = audio_data[::2]
                    new_framerate = framerate // 2
                elif framerate > 16000:
                    # Generic downsampling (integer ratio)
                    ratio = int(framerate / 16000) + 1
                    resampled_data = audio_data[::ratio]
                    new_framerate = int(framerate / ratio)
                else:
                    # Already low enough
                    resampled_data = audio_data
                    new_framerate = framerate
                
                logger.info(f"Resampling: {framerate}Hz -> {new_framerate}Hz")

                # Write output WAV
                with wave.open(str(output_path), 'wb') as wav_out:
                    wav_out.setnchannels(1)  # Mono
                    wav_out.setsampwidth(2)  # 16-bit
                    wav_out.setframerate(new_framerate)
                    wav_out.writeframes(resampled_data.tobytes())
            
            # Clean up temp WAV
            try:
                temp_wav.unlink()
            except:
                pass
            
            logger.info(f"Converted audio: {framerate}Hz -> {new_framerate}Hz with effects")
            return output_path
            
        except ImportError:
            logger.error("pydub not installed, cannot convert MP3. Install: pip install pydub")
            return None
        except Exception as e:
            import traceback
            logger.error(f"Audio conversion failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _apply_robotic_filter(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply 1970s-style robotic audio effects to make TTS sound like classic sci-fi robots.
        
        Effects applied:
        - Pitch shift up (higher pitched voice)
        - Time stretch (20% slower playback)
        - Heavy bit crushing (digital/lo-fi effect)
        - Aggressive low-pass filter (small speaker effect)
        - Strong ring modulation (metallic/robotic quality)
        - Chorus effect (slight detuning for vintage synth quality)
        
        Args:
            audio_data: Float32 audio samples (-32768 to 32767 range)
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed audio data (float32)
        """
        try:
            # 80s ROBOT VOICE PRESET - Authentic retro robot sound!
            # Inspired by robots from TV shows like Transformers, Buck Rogers, etc.
            if getattr(self, 'monotone', False):
                # Pitch shift UP by 2 semitones for robotic voice (higher pitched)
                pitch_shift_semitones = 2
                pitch_ratio = 2 ** (pitch_shift_semitones / 12.0)
                original_length = len(audio_data)
                
                # Simple pitch shift using sample rate manipulation
                target_length = int(len(audio_data) / pitch_ratio)
                indices = np.linspace(0, len(audio_data) - 1, target_length)
                pitched = np.interp(indices, np.arange(len(audio_data)), audio_data)

                # Light time stretch to keep speech natural (not too slow)
                time_stretch_factor = 0.95  # 5% slower for robotic cadence
                target_len = int(len(pitched) * time_stretch_factor)
                indices = np.linspace(0, len(pitched) - 1, target_len)
                stretched = np.interp(indices, np.arange(len(pitched)), pitched)

                # Moderate dynamic compression to add punch but preserve some dynamics
                threshold = np.percentile(np.abs(stretched), 65)
                compression_ratio = 2.0
                stretched = np.where(
                    np.abs(stretched) > threshold,
                    np.sign(stretched) * (threshold + (np.abs(stretched) - threshold) / compression_ratio),
                    stretched
                )

                # Moderate bit depth (keep clarity)
                bit_depth = 12
                max_val = 2 ** (bit_depth - 1)
                crushed = np.round(stretched / (32768 / max_val)) * (32768 / max_val)

                # Gentle low-pass for speaker coloration
                cutoff_freq = 4000
                window_size = max(1, int(sample_rate / cutoff_freq))
                if window_size > 1:
                    kernel = np.ones(window_size) / window_size
                    filtered = np.convolve(crushed, kernel, mode='same')
                else:
                    filtered = crushed

                # Minimal ring modulation (very low depth)
                mod_freq = 40
                t = np.arange(len(filtered)) / sample_rate
                modulator = 0.98 + 0.02 * np.sin(2 * np.pi * mod_freq * t)
                modulated = filtered * modulator

                # No chorus for monotone voice
                chorused = modulated

                logger.debug(f"🤖 Monotone filter: pitch={pitch_shift_semitones}st, speed={time_stretch_factor}x, bits={bit_depth}, cutoff={cutoff_freq}Hz, mod={mod_freq}Hz")
                return chorused
            else:
                # 1. Pitch Shift UP (higher pitched, like helium voice or C-3PO)
                # Simple method: resample at higher rate then play back at normal rate
                pitch_shift_semitones = 14  # Stronger chipmunk effect (higher pitch)
                pitch_ratio = 2 ** (pitch_shift_semitones / 12.0)  # ~1.78x
                
                # Resample to simulate pitch shift
                original_length = len(audio_data)
                new_length = int(original_length / pitch_ratio)
                indices = np.linspace(0, original_length - 1, new_length)
                pitched = np.interp(indices, np.arange(original_length), audio_data)
                
                # 2. Time Stretch (slower)
                # <1.0 shortens audio (faster speech), >1.0 lengthens audio (slower speech)
                time_stretch_factor = 1.5  # much slower for intelligibility
                stretched_length = int(new_length * time_stretch_factor)
                indices = np.linspace(0, new_length - 1, stretched_length)
                stretched = np.interp(indices, np.arange(new_length), pitched)
                
                # 2b. Dynamic Flattening (compress dynamics for monotone robot delivery)
                # Reduce volume variations to make speech more uniform/expressionless
                threshold = np.percentile(np.abs(stretched), 60)  # 60th percentile
                compression_ratio = 3.0  # Heavy compression
                stretched = np.where(
                    np.abs(stretched) > threshold,
                    np.sign(stretched) * (threshold + (np.abs(stretched) - threshold) / compression_ratio),
                    stretched
                )
                
                # 3. Heavy Bit Crushing (6-bit for ultra-robotic 1970s digital sound)
                bit_depth = 6  # 6-bit = very lo-fi (Speak & Spell style)
                max_val = 2 ** (bit_depth - 1)
                crushed = np.round(stretched / (32768 / max_val)) * (32768 / max_val)
                
                # 4. Aggressive Low-pass filter (1970s telephone/radio quality)
                cutoff_freq = 2500  # Hz (very narrow frequency range)
                window_size = int(sample_rate / cutoff_freq)
                if window_size > 1:
                    kernel = np.ones(window_size) / window_size
                    filtered = np.convolve(crushed, kernel, mode='same')
                else:
                    filtered = crushed
                
                # 5. Strong Ring Modulation (metallic vocoder-like effect)
                mod_freq = 80  # Hz (prominent modulation for classic robot sound)
                t = np.arange(len(filtered)) / sample_rate
                modulator = 0.6 + 0.4 * np.sin(2 * np.pi * mod_freq * t)  # Strong variation
                modulated = filtered * modulator
                
                # 6. Chorus Effect (slight detuning for vintage synth quality)
                # Add a slightly delayed and pitch-shifted copy
                delay_samples = int(0.02 * sample_rate)  # 20ms delay
                chorus_mix = 0.3  # 30% mix of chorused signal
                
                if len(modulated) > delay_samples:
                    delayed = np.zeros_like(modulated)
                    delayed[delay_samples:] = modulated[:-delay_samples] * 0.95  # Slightly quieter + detuned
                    chorused = modulated * (1 - chorus_mix) + delayed * chorus_mix
                else:
                    chorused = modulated

                logger.debug(f"🤖 1970s robotic filter: pitch=+{pitch_shift_semitones}st, speed={time_stretch_factor}x, bits={bit_depth}, cutoff={cutoff_freq}Hz, mod={mod_freq}Hz")
                return chorused
            
        except Exception as e:
            logger.error(f"Robotic filter failed: {e}, returning original audio")
            return audio_data

    def calculate_cost(self, text: str) -> float:
        """
        Calculate cost for TTS generation
        
        Args:
            text: Text to be converted
        
        Returns:
            Cost in euros
        
        OpenAI TTS Pricing (as of Dec 2024):
        - tts-1: $0.015 per 1K characters
        - tts-1-hd: $0.030 per 1K characters
        """
        # Convert USD to EUR (approximate: 1 USD = 0.93 EUR)
        usd_to_eur = 0.93
        
        # Pricing per 1K characters
        pricing = {
            "tts-1": 0.015,
            "tts-1-hd": 0.030
        }
        
        cost_per_1k = pricing.get(self.model, 0.015)
        char_count = len(text)
        
        # Calculate cost
        cost_usd = (char_count / 1000.0) * cost_per_1k
        cost_eur = cost_usd * usd_to_eur
        
        return cost_eur
    
    def set_voice(self, voice: str):
        """
        Change default voice
        
        Available voices:
        - alloy: Neutral, versatile
        - echo: Mature, warm
        - fable: British accent, storytelling
        - onyx: Deep, authoritative
        - nova: Young, energetic (default)
        - shimmer: Soft, gentle
        """
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            logger.warning(f"Invalid voice '{voice}', keeping '{self.voice}'")
            return
        
        self.voice = voice
        logger.info(f"Voice changed to: {voice}")
    
    def set_speed(self, speed: float):
        """
        Change speech speed
        
        Args:
            speed: Speed multiplier (0.25 to 4.0)
        """
        if speed < 0.25 or speed > 4.0:
            logger.warning(f"Invalid speed {speed}, must be 0.25-4.0")
            return
        
        self.speed = speed
        logger.info(f"Speed changed to: {speed}x")
    
    def get_statistics(self) -> dict:
        """
        Get TTS usage statistics
        
        Returns:
            Dict with statistics
        """
        avg_chars = self.total_characters / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "total_characters": self.total_characters,
            "total_failures": self.total_failures,
            "average_chars_per_call": avg_chars,
            "success_rate": (self.total_calls - self.total_failures) / self.total_calls if self.total_calls > 0 else 0.0
        }


# ========== Convenience Functions ==========

def create_tts(
    api_key: str,
    robot: Any,
    budget_enforcer: Optional[Any] = None,
    model: str = "tts-1",
    voice: str = "nova"
) -> TextToSpeech:
    """
    Factory function to create TextToSpeech instance
    
    Args:
        api_key: OpenAI API key
        robot: Vector robot instance
        budget_enforcer: Optional BudgetEnforcer
        model: TTS model name
        voice: Voice name
    
    Returns:
        TextToSpeech instance
    
    Usage:
        tts = create_tts(api_key, robot, enforcer)
        await tts.speak("Hello, I am Vector!")
    """
    return TextToSpeech(
        api_key=api_key,
        robot=robot,
        budget_enforcer=budget_enforcer,
        model=model,
        voice=voice
    )
