"""
Standalone Vector microphone capture test.

Connects to Vector, streams 10 seconds of audio from its built-in
microphone array, then saves the raw bytes to a WAV file for inspection.

Usage:
    python test_vector_mic.py

Output:
    vector_mic_capture.wav  (inspect in Audacity or any audio player)
    Prints per-chunk statistics to console.

This is a diagnostic tool — it bypasses the full agent to isolate
mic streaming issues.
"""
import asyncio
import math
import struct
import time
import wave

import anki_vector
from anki_vector.audio import EvtAudioChunk

CAPTURE_SECONDS = 10
SAMPLE_RATE = 16000          # assumed rate we tell webrtcvad
OUTPUT_WAV = "vector_mic_capture.wav"


def analyse_chunk(data: bytes, idx: int) -> None:
    """Print stats about a raw audio chunk."""
    n = len(data) // 2
    if n == 0:
        print(f"  chunk #{idx}: EMPTY")
        return
    samples = struct.unpack_from(f'<{n}h', data)
    mn, mx = min(samples), max(samples)
    rms = math.sqrt(sum(s * s for s in samples) / n)
    unique = len(set(samples))
    if idx <= 5 or idx % 50 == 0:
        print(
            f"  chunk #{idx:4d}: {len(data):5d}B  "
            f"min={mn:7d} max={mx:7d} rms={rms:7.1f}  "
            f"unique={unique:5d}  "
            f"first8={data[:8].hex()}"
        )


async def main():
    print(f"Connecting to Vector...")
    robot = anki_vector.Robot(enable_audio_feed=True)
    robot.connect(timeout=30)
    print("✅ Connected")

    all_data = bytearray()
    chunk_count = 0
    start = time.time()

    def on_chunk(chunk: EvtAudioChunk):
        nonlocal chunk_count
        chunk_count += 1
        all_data.extend(chunk.data)
        analyse_chunk(chunk.data, chunk_count)

    robot.audio.register_audio_callback(on_chunk)
    robot.audio.init_audio_feed()

    print(f"\n🎤 Recording for {CAPTURE_SECONDS}s — make some noise!\n")
    await asyncio.sleep(CAPTURE_SECONDS)

    robot.audio.close_audio_feed()
    robot.audio.unregister_audio_callback(on_chunk)

    print(f"\n--- CAPTURE SUMMARY ---")
    print(f"Chunks received : {chunk_count}")
    print(f"Total bytes     : {len(all_data)}")
    print(f"Duration        : {time.time() - start:.1f}s")

    if len(all_data) >= 2:
        samples = struct.unpack_from(f'<{len(all_data)//2}h', bytes(all_data))
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        mn, mx = min(samples), max(samples)
        unique = len(set(samples))
        print(f"Overall RMS     : {rms:.1f}")
        print(f"Sample range    : {mn} … {mx}")
        print(f"Unique values   : {unique}")

        if unique < 10:
            print("\n⚠️  Very few unique values — signal_power may NOT be raw PCM audio.")
            print("   It might be amplitude/power envelope bytes rather than audio samples.")
        elif rms < 50:
            print("\n⚠️  Very low RMS — audio is near-silent. Check mic orientation.")
        else:
            print(f"\n✅ Signal looks like real audio (rms={rms:.0f}, unique={unique})")

        # Save WAV regardless so it can be opened in Audacity
        with wave.open(OUTPUT_WAV, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(all_data))
        print(f"\n💾 Saved: {OUTPUT_WAV}")
        print("   Open in Audacity or VLC to listen / inspect the waveform.")
    else:
        print("\n❌ No audio data captured!")

    robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
