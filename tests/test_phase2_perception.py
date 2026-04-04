"""
Phase 2: Perception Enhancement Tests

Test-Driven Development approach:
1. Write failing tests FIRST
2. Implement minimal code to pass
3. Refactor while keeping tests green

Test Coverage:
- AudioProcessor: VAD detection, buffer management, silence detection
- SpeechRecognizer: Whisper API integration, confidence scoring, error handling
- ObjectDetector: YOLOv5 inference, confidence filtering, COCO class mapping
- RoomInference: Pattern matching, room type detection, database integration
- Integration: End-to-end audio → speech → objects → room → database
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Phase 2 modules (will fail until implemented)
try:
    from vector_personality.perception.audio_processor import AudioProcessor
except ImportError:
    AudioProcessor = None

try:
    from vector_personality.perception.speech_recognition import SpeechRecognizer
except ImportError:
    SpeechRecognizer = None

try:
    from vector_personality.perception.object_detector import ObjectDetector
except ImportError:
    ObjectDetector = None

try:
    from vector_personality.perception.room_inference import RoomInference
except ImportError:
    RoomInference = None

# Import Phase 1 dependencies
from vector_personality.memory.chromadb_connector import ChromaDBConnector as SQLServerConnector
from vector_personality.memory.working_memory import WorkingMemory


# ============================================================================
# Test Class 1: AudioProcessor (VAD, Buffer Management, Silence Detection)
# ============================================================================

@pytest.mark.skipif(AudioProcessor is None, reason="AudioProcessor not yet implemented")
class TestAudioProcessor:
    """Test audio processing with Voice Activity Detection"""

    def setup_method(self):
        """Initialize AudioProcessor for each test"""
        self.processor = AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_mode=3  # Aggressive VAD
        )

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'processor') and self.processor:
            self.processor.stop()

    def test_initialization(self):
        """Test AudioProcessor initializes with correct parameters"""
        assert self.processor.sample_rate == 16000
        assert self.processor.frame_duration_ms == 30
        assert self.processor.vad_mode == 3
        assert not self.processor.is_recording
        assert len(self.processor.buffer) == 0

    def test_vad_detects_speech(self):
        """Test VAD correctly identifies speech frames"""
        # Create synthetic speech-like audio (random noise with high energy)
        speech_frame = np.random.randint(-32768, 32767, 480, dtype=np.int16)
        
        is_speech = self.processor.is_speech(speech_frame.tobytes())
        
        # VAD should detect high-energy signal as speech
        assert isinstance(is_speech, bool)

    def test_vad_detects_silence(self):
        """Test VAD correctly identifies silence frames"""
        # Create silent audio (all zeros)
        silence_frame = np.zeros(480, dtype=np.int16)
        
        is_speech = self.processor.is_speech(silence_frame.tobytes())
        
        # VAD should detect zeros as silence
        assert is_speech is False

    def test_buffer_management(self):
        """Test circular buffer stores audio correctly"""
        # Add multiple frames
        for i in range(10):
            frame = np.random.randint(-100, 100, 480, dtype=np.int16)
            self.processor.add_frame(frame.tobytes())
        
        # Buffer should contain frames
        assert len(self.processor.buffer) > 0
        assert len(self.processor.buffer) <= self.processor.max_buffer_size

    def test_silence_detection_threshold(self):
        """Test silence detection after 300ms threshold"""
        # Add silence frames (300ms = 10 frames at 30ms each)
        silence_frame = np.zeros(480, dtype=np.int16)
        
        for i in range(12):  # More than threshold
            self.processor.add_frame(silence_frame.tobytes())
            is_silent = self.processor.check_silence_timeout()
            
            if i < 10:
                assert not is_silent  # Not enough silence yet
            else:
                assert is_silent  # Threshold exceeded

    def test_start_stop_recording(self):
        """Test starting and stopping audio recording"""
        self.processor.start_recording()
        assert self.processor.is_recording is True
        
        self.processor.stop_recording()
        assert self.processor.is_recording is False

    def test_export_audio_to_wav(self):
        """Test exporting buffered audio to WAV file"""
        # Add some audio frames
        for i in range(50):
            frame = np.random.randint(-1000, 1000, 480, dtype=np.int16)
            self.processor.add_frame(frame.tobytes())
        
        # Export to WAV
        output_path = "test_audio_output.wav"
        self.processor.export_wav(output_path)
        
        # Verify file exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 44  # WAV header is 44 bytes
        
        # Cleanup
        os.remove(output_path)

    def test_clear_buffer(self):
        """Test clearing audio buffer"""
        # Add frames
        for i in range(10):
            frame = np.random.randint(-100, 100, 480, dtype=np.int16)
            self.processor.add_frame(frame.tobytes())
        
        assert len(self.processor.buffer) > 0
        
        self.processor.clear_buffer()
        assert len(self.processor.buffer) == 0


# ============================================================================
# Test Class 2: SpeechRecognizer (Groq Whisper API Integration)
# ============================================================================

@pytest.mark.skipif(SpeechRecognizer is None, reason="SpeechRecognizer not yet implemented")
class TestSpeechRecognition:
    """Test speech recognition with Groq Whisper API"""

    def setup_method(self):
        """Initialize SpeechRecognizer for each test"""
        self.recognizer = SpeechRecognizer(
            api_key="test_key_12345",  # Mock API key
            model="whisper-large-v3"
        )

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test SpeechRecognizer initializes correctly"""
        assert self.recognizer.model == "whisper-large-v3"
        assert self.recognizer.api_key == "test_key_12345"

    @pytest.mark.asyncio
    @patch('openai.Audio.atranscribe')
    async def test_transcribe_audio_success(self, mock_transcribe):
        """Test successful audio transcription"""
        # Mock Whisper API response
        mock_transcribe.return_value = {
            "text": "Hello, Vector!",
            "language": "en"
        }
        
        # Create mock audio file
        audio_path = "test_audio.wav"
        with open(audio_path, 'wb') as f:
            f.write(b'\x00' * 1000)  # Dummy WAV data
        
        try:
            result = await self.recognizer.transcribe(audio_path)
            
            assert result["text"] == "Hello, Vector!"
            assert result["language"] == "en"
            assert result["confidence"] >= 0.0
            assert result["confidence"] <= 1.0
        finally:
            os.remove(audio_path)

    @pytest.mark.asyncio
    @patch('openai.Audio.atranscribe')
    async def test_transcribe_empty_audio(self, mock_transcribe):
        """Test transcription of silent/empty audio"""
        mock_transcribe.return_value = {
            "text": "",
            "language": "en"
        }
        
        audio_path = "test_silence.wav"
        with open(audio_path, 'wb') as f:
            f.write(b'\x00' * 500)
        
        try:
            result = await self.recognizer.transcribe(audio_path)
            assert result["text"] == ""
        finally:
            os.remove(audio_path)

    @pytest.mark.asyncio
    @patch('openai.Audio.atranscribe')
    async def test_transcribe_with_retry(self, mock_transcribe):
        """Test retry logic on API failure"""
        # First call fails, second succeeds
        mock_transcribe.side_effect = [
            Exception("API timeout"),
            {"text": "Retry successful", "language": "en"}
        ]
        
        audio_path = "test_retry.wav"
        with open(audio_path, 'wb') as f:
            f.write(b'\x00' * 1000)
        
        try:
            result = await self.recognizer.transcribe(audio_path, max_retries=2)
            assert result["text"] == "Retry successful"
        finally:
            os.remove(audio_path)

    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # Test with different transcription lengths
        short_text = "Hi"
        long_text = "This is a longer transcription with more words"
        
        short_confidence = self.recognizer.calculate_confidence(short_text, duration=1.0)
        long_confidence = self.recognizer.calculate_confidence(long_text, duration=3.0)
        
        # Longer, well-formed text should have higher confidence
        assert 0.0 <= short_confidence <= 1.0
        assert 0.0 <= long_confidence <= 1.0
        assert long_confidence >= short_confidence

    @pytest.mark.asyncio
    @patch('openai.Audio.atranscribe')
    async def test_language_detection(self, mock_transcribe):
        """Test automatic language detection"""
        mock_transcribe.return_value = {
            "text": "Bonjour Vector",
            "language": "fr"
        }
        
        audio_path = "test_french.wav"
        with open(audio_path, 'wb') as f:
            f.write(b'\x00' * 1000)
        
        try:
            result = await self.recognizer.transcribe(audio_path)
            assert result["language"] == "fr"
        finally:
            os.remove(audio_path)


# ============================================================================
# Test Class 3: ObjectDetector (YOLOv5 Integration)
# ============================================================================

@pytest.mark.skipif(ObjectDetector is None, reason="ObjectDetector not yet implemented")
class TestObjectDetector:
    """Test object detection with YOLOv5"""

    def setup_method(self):
        """Initialize ObjectDetector for each test"""
        self.detector = ObjectDetector(
            model_path="yolov5n.pt",  # Nano model for speed
            confidence_threshold=0.5,
            device="cpu"
        )

    def test_initialization(self):
        """Test ObjectDetector initializes correctly"""
        assert self.detector.confidence_threshold == 0.5
        assert self.detector.device == "cpu"
        assert self.detector.model is not None

    def test_detect_objects_in_image(self):
        """Test object detection on synthetic image"""
        # Create dummy image (640x480 RGB)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = self.detector.detect(image)
        
        # Verify detection format
        assert isinstance(detections, list)
        for detection in detections:
            assert "class" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            assert detection["confidence"] >= 0.5  # Threshold applied

    def test_confidence_filtering(self):
        """Test detections below threshold are filtered"""
        # Create image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Set high threshold
        self.detector.confidence_threshold = 0.9
        detections = self.detector.detect(image)
        
        # All detections should meet threshold
        for detection in detections:
            assert detection["confidence"] >= 0.9

    def test_coco_class_mapping(self):
        """Test COCO class IDs map to readable names"""
        # Test known COCO classes
        assert self.detector.get_class_name(0) == "person"
        assert self.detector.get_class_name(56) == "chair"
        assert self.detector.get_class_name(62) == "laptop"
        assert self.detector.get_class_name(67) == "cell phone"

    def test_detect_from_vector_camera(self):
        """Test detection from Vector camera frame"""
        # Mock Vector camera image (numpy array)
        mock_camera_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        detections = self.detector.detect(mock_camera_frame)
        
        assert isinstance(detections, list)

    def test_batch_detection(self):
        """Test detecting objects in multiple frames"""
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        all_detections = self.detector.detect_batch(frames)
        
        assert len(all_detections) == 5
        assert all(isinstance(d, list) for d in all_detections)

    def test_fps_performance(self):
        """Test detection speed meets >5 FPS requirement"""
        import time
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Measure detection time
        start = time.time()
        for _ in range(10):
            self.detector.detect(image)
        elapsed = time.time() - start
        
        fps = 10 / elapsed
        assert fps >= 5.0, f"FPS {fps:.2f} below requirement (5 FPS)"


# ============================================================================
# Test Class 4: RoomInference (Pattern Matching)
# ============================================================================

@pytest.mark.skipif(RoomInference is None, reason="RoomInference not yet implemented")
class TestRoomInference:
    """Test room type inference from object patterns"""

    @pytest.mark.asyncio
    async def setup_method(self):
        """Initialize RoomInference and database connector"""
        self.connector = SQLServerConnector()
        self.inference = RoomInference(self.connector)

    @pytest.mark.asyncio
    async def teardown_method(self):
        """Cleanup after each test"""
        await self.connector.close()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test RoomInference initializes with patterns"""
        assert self.inference.connector is not None
        assert len(self.inference.room_patterns) > 0

    @pytest.mark.asyncio
    async def test_infer_kitchen(self):
        """Test kitchen detection from objects"""
        objects = [
            {"class": "refrigerator", "confidence": 0.9},
            {"class": "oven", "confidence": 0.85},
            {"class": "microwave", "confidence": 0.8},
            {"class": "sink", "confidence": 0.75}
        ]
        
        room_type = self.inference.infer_room_type(objects)
        assert room_type == "kitchen"

    @pytest.mark.asyncio
    async def test_infer_bedroom(self):
        """Test bedroom detection from objects"""
        objects = [
            {"class": "bed", "confidence": 0.95},
            {"class": "pillow", "confidence": 0.88},
            {"class": "clock", "confidence": 0.7}
        ]
        
        room_type = self.inference.infer_room_type(objects)
        assert room_type == "bedroom"

    @pytest.mark.asyncio
    async def test_infer_living_room(self):
        """Test living room detection from objects"""
        objects = [
            {"class": "couch", "confidence": 0.92},
            {"class": "tv", "confidence": 0.89},
            {"class": "remote", "confidence": 0.75}
        ]
        
        room_type = self.inference.infer_room_type(objects)
        assert room_type == "living_room"

    @pytest.mark.asyncio
    async def test_infer_unknown_room(self):
        """Test unknown room when patterns don't match"""
        objects = [
            {"class": "car", "confidence": 0.9},
            {"class": "traffic light", "confidence": 0.85}
        ]
        
        room_type = self.inference.infer_room_type(objects)
        assert room_type == "unknown"

    @pytest.mark.asyncio
    async def test_pattern_confidence_scoring(self):
        """Test confidence scoring for room patterns"""
        objects = [
            {"class": "refrigerator", "confidence": 0.95},
            {"class": "oven", "confidence": 0.90}
        ]
        
        scores = self.inference.calculate_pattern_scores(objects)
        
        # Kitchen should have highest score
        assert "kitchen" in scores
        assert scores["kitchen"] > 0.5

    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test GetOrCreateRoom database integration"""
        # Infer room
        objects = [
            {"class": "bed", "confidence": 0.95},
            {"class": "pillow", "confidence": 0.88}
        ]
        room_type = self.inference.infer_room_type(objects)
        
        # Store in database
        room_id = await self.inference.get_or_create_room(room_type)
        
        assert room_id is not None
        assert isinstance(room_id, type(uuid4()))

    @pytest.mark.asyncio
    async def test_room_transition_detection(self):
        """Test detecting room transitions"""
        # First room
        objects1 = [{"class": "bed", "confidence": 0.95}]
        room_id1 = await self.inference.process_objects(objects1)
        
        # Second room (different)
        objects2 = [{"class": "refrigerator", "confidence": 0.95}]
        room_id2 = await self.inference.process_objects(objects2)
        
        # Should detect transition
        assert room_id1 != room_id2
        assert self.inference.last_room_id == room_id2

    @pytest.mark.asyncio
    async def test_80_percent_accuracy_target(self):
        """Test room inference meets 80% accuracy target on test scenes"""
        # Test cases: (objects, expected_room)
        test_cases = [
            ([{"class": "bed", "confidence": 0.9}], "bedroom"),
            ([{"class": "refrigerator", "confidence": 0.9}], "kitchen"),
            ([{"class": "couch", "confidence": 0.9}], "living_room"),
            ([{"class": "toilet", "confidence": 0.9}], "bathroom"),
            ([{"class": "oven", "confidence": 0.9}], "kitchen"),
            ([{"class": "tv", "confidence": 0.9}], "living_room"),
            ([{"class": "dining table", "confidence": 0.9}], "dining_room"),
            ([{"class": "desk", "confidence": 0.9}], "office"),
            ([{"class": "sink", "confidence": 0.8}], "bathroom"),  # Could be kitchen
            ([{"class": "chair", "confidence": 0.9}], "unknown")  # Ambiguous
        ]
        
        correct = 0
        for objects, expected in test_cases:
            inferred = self.inference.infer_room_type(objects)
            if inferred == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.80, f"Accuracy {accuracy:.2%} below target (80%)"


# ============================================================================
# Integration Test: End-to-End Perception Pipeline
# ============================================================================

@pytest.mark.skipif(
    AudioProcessor is None or SpeechRecognizer is None or 
    ObjectDetector is None or RoomInference is None,
    reason="Phase 2 modules not yet implemented"
)
class TestPerceptionIntegration:
    """Test complete perception pipeline: audio → speech → objects → room → database"""

    @pytest.mark.asyncio
    async def setup_method(self):
        """Initialize all components"""
        self.audio = AudioProcessor(sample_rate=16000, frame_duration_ms=30)
        self.speech = SpeechRecognizer(api_key="test_key")
        self.detector = ObjectDetector(model_path="yolov5n.pt")
        self.connector = SQLServerConnector()
        self.room_inference = RoomInference(self.connector)
        self.working_memory = WorkingMemory()

    @pytest.mark.asyncio
    async def teardown_method(self):
        """Cleanup all components"""
        self.audio.stop()
        await self.connector.close()

    @pytest.mark.asyncio
    @patch('openai.Audio.atranscribe')
    async def test_end_to_end_pipeline(self, mock_transcribe):
        """Test complete perception flow"""
        # Mock Whisper response
        mock_transcribe.return_value = {
            "text": "Where is my laptop?",
            "language": "en"
        }
        
        # Step 1: Audio capture (simulated)
        for i in range(50):
            frame = np.random.randint(-1000, 1000, 480, dtype=np.int16)
            self.audio.add_frame(frame.tobytes())
        
        audio_path = "test_integration.wav"
        self.audio.export_wav(audio_path)
        
        try:
            # Step 2: Speech recognition
            speech_result = await self.speech.transcribe(audio_path)
            assert speech_result["text"] == "Where is my laptop?"
            
            # Step 3: Object detection (simulate laptop detection)
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = self.detector.detect(image)
            
            # Simulate laptop detection for test
            mock_detections = [
                {"class": "laptop", "confidence": 0.92, "bbox": [100, 100, 200, 200]},
                {"class": "desk", "confidence": 0.85, "bbox": [0, 200, 640, 480]}
            ]
            
            # Step 4: Room inference
            room_id = await self.room_inference.process_objects(mock_detections)
            assert room_id is not None
            
            # Step 5: Update working memory
            self.working_memory.current_room_id = room_id
            for detection in mock_detections:
                self.working_memory.observe_object(
                    detection["class"],
                    detection["confidence"],
                    "detected in frame"
                )
            
            # Step 6: Store conversation in database
            conversation_id = await self.connector.create_conversation(
                speaker_id=None,  # Unknown speaker
                text=speech_result["text"],
                room_id=room_id,
                emotional_context="curious",
                response_text=None,
                response_type=None
            )
            
            assert conversation_id is not None
            
            # Verify database state
            room = await self.connector.get_room(room_id)
            assert room is not None
            
            recent_convos = await self.connector.get_recent_conversations(limit=1)
            assert len(recent_convos) == 1
            assert recent_convos[0]["text"] == "Where is my laptop?"
            
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    @pytest.mark.asyncio
    async def test_mood_impact_from_perception(self):
        """Test perception events affect mood correctly"""
        initial_mood = self.working_memory.current_mood
        
        # New object detection (+3 mood)
        self.working_memory.observe_object("laptop", 0.9, "on desk")
        self.working_memory.update_mood(3.0, "new_object_detected")
        
        assert self.working_memory.current_mood == initial_mood + 3.0
        
        # Room transition (+5 mood)
        room_id = uuid4()
        self.working_memory.current_room_id = room_id
        self.working_memory.update_mood(5.0, "room_transition")
        
        assert self.working_memory.current_mood == initial_mood + 8.0

    @pytest.mark.asyncio
    async def test_perception_performance_targets(self):
        """Test all perception modules meet performance targets"""
        import time
        
        # Audio processing: Should handle real-time (30ms frames)
        frame = np.random.randint(-1000, 1000, 480, dtype=np.int16)
        start = time.time()
        for _ in range(100):
            self.audio.add_frame(frame.tobytes())
        audio_time = (time.time() - start) / 100
        assert audio_time < 0.030, f"Audio processing too slow: {audio_time*1000:.1f}ms"
        
        # Object detection: >5 FPS
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        start = time.time()
        for _ in range(10):
            self.detector.detect(image)
        detection_time = (time.time() - start) / 10
        fps = 1 / detection_time
        assert fps >= 5.0, f"Object detection too slow: {fps:.1f} FPS"


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


if __name__ == "__main__":
    # Run tests with: pytest tests/test_phase2_perception.py -v
    pytest.main([__file__, "-v", "--tb=short"])
