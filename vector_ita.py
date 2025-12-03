# vector_ita.py
import asyncio
import os
from anki_vector.behavior import BehaviorComponent
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
import tempfile

class ItalianBehavior(BehaviorComponent):
    """
    Override del BehaviorComponent per far parlare Vector in italiano.
    Genera WAV compatibile con Vector (PCM 16-bit).
    """

    async def say_text(self, text, **kwargs):
        # Traduzione in italiano
        translator = Translator()
        tradotto = translator.translate(text, src='en', dest='it').text
        print(f"[Vector Italiano] Traduzione: {tradotto}")

        # Crea file temporanei
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
            mp3_file = mp3_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
            wav_file = wav_tmp.name

        try:
            # Genera MP3 con gTTS
            tts = gTTS(text=tradotto, lang='it')
            tts.save(mp3_file)

            # Converte MP3 in WAV PCM 16-bit compatibile Vector
            audio = AudioSegment.from_mp3(mp3_file)
            audio = audio.set_channels(1)        # Mono
            audio = audio.set_frame_rate(16000)  # Frame rate consigliato da Vector SDK
            audio.export(wav_file, format="wav")
            
            # Riproduce il WAV su Vector
            # Note: the SDK's audio.stream_wav_file is a synchronous call that
            # starts playback and returns None, so do not await it.
            try:
                self.robot.audio.stream_wav_file(wav_file)
            except Exception as e:
                # Log and continue to cleanup files
                print(f"[Vector Italiano] Errore durante la riproduzione audio: {e}")

        finally:
            # Pulizia file temporanei
            if os.path.exists(mp3_file):
                os.remove(mp3_file)
            if os.path.exists(wav_file):
                os.remove(wav_file)


class RobotItaliano:
    """
    Wrapper del Robot originale. Mantiene tutti i metodi originali,
    sostituendo solo behavior con quello italiano.
    """

    def __init__(self, robot):
        self.robot = robot
        self.behavior = ItalianBehavior(robot)
        # Copia altri attributi del robot
        self.__dict__.update(robot.__dict__)

    def __getattr__(self, name):
        # Tutti gli altri metodi passano al robot originale
        return getattr(self.robot, name)
