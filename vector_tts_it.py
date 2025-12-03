import io
import os
from gtts import gTTS
from pydub import AudioSegment
from anki_vector import Robot
from anki_vector.connection import ControlPriorityLevel
from googletrans import Translator  # <-- traduzione automatica

def speak_translated_text(text_en: str, robot_ip: str = None):
    """
    Traduci un testo inglese in italiano, genera la voce e la riproduce su Vector.

    Args:
        text_en (str): Il testo in inglese da tradurre e pronunciare.
        robot_ip (str, optional): L'indirizzo IP di Vector (se non viene rilevato automaticamente).
    """
    mp3_path = "vector_voice_it.mp3"
    wav_path = "vector_voice_it.wav"

    try:
        # 1. Traduzione inglese → italiano
        print("🌐 Translating from English to Italian...")
        translator = Translator()
        translation = translator.translate(text_en, src='en', dest='it')
        text_it = translation.text
        print(f"✅ Translated text: {text_it}")

        # 2. Genera MP3 con Google TTS (in italiano)
        print("🔊 Generating Italian voice...")
        tts = gTTS(text=text_it, lang='it')
        tts.save(mp3_path)

        # 3. Conversione MP3 → WAV per Vector
        print("🎵 Converting MP3 to WAV...")
        AudioSegment.from_mp3(mp3_path).set_frame_rate(16000).export(wav_path, format="wav")

        # 4. Connessione e riproduzione su Vector
        print("🤖 Connecting to Vector...")
        with Robot(robot_ip, default_logging=False, behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS) as robot:
            print("✅ Connected! Playing audio...")
            with open(wav_path, "rb") as f:
                robot.audio.stream_wav_file(io.BytesIO(f.read()))
            print("🇮🇹 Vector has spoken in Italian!")

    except Exception as e:
        print(f"⚠️ Error during execution: {e}")

    finally:
        # 5. Pulizia dei file temporanei
        for f in [mp3_path, wav_path]:
            if os.path.exists(f):
                os.remove(f)
