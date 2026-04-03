"""
Test Suite for Intelligent Hallucination Detection (T146)

Tests the multi-level hallucination detection system that prevents
Vector from responding to Whisper hallucinations.

Test Coverage:
- Obvious patterns (timestamps, video artifacts)
- Repetitions (e.g., "si si si")
- Common single-word hallucinations
- Low confidence filtering
- Lexical diversity analysis
- Wake word exceptions
"""

import pytest
from vector_personality.perception.speech_recognition import SpeechRecognizer


class TestHallucinationDetection:
    """Test intelligent hallucination detection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create recognizer with dummy API key (not used in these tests)
        self.recognizer = SpeechRecognizer(api_key="dummy", language="it")
    
    # ==================== Obvious Pattern Tests ====================
    
    def test_detects_obvious_video_artifacts(self):
        """Should detect common video subtitle artifacts"""
        test_cases = [
            ("Al prossimo episodio!", "obvious_pattern"),
            ("00:00", "obvious_pattern"),
            ("www.example.com", "obvious_pattern"),
            ("Sottotitoli creati dalla comunità", "obvious_pattern"),
        ]
        
        for text, expected_reason in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence=0.8)
            assert is_halluc, f"Should detect '{text}' as hallucination"
            assert expected_reason in reason, f"Expected reason '{expected_reason}' in '{reason}'"
    
    # ==================== Repetition Tests ====================
    
    def test_detects_identical_repetitions(self):
        """Should detect repetitive words like 'si si si'"""
        test_cases = [
            "si si si",
            "no no no",
            "ciao ciao",
            "buongiorno buongiorno buongiorno",
        ]
        
        for text in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence=0.8)
            assert is_halluc, f"Should detect repetition in '{text}'"
            assert "repetition" in reason, f"Expected 'repetition' in reason: {reason}"
    
    def test_allows_wake_word_repetition(self):
        """Should allow 'Vector Vector!' only when it's the wake word itself"""
        # Wake word repetition - allowed
        is_halluc, _ = self.recognizer.is_likely_hallucination(
            "vector vector", 
            confidence=0.8,
            has_wake_word=True
        )
        assert not is_halluc, "Should allow wake word repetition"
        
        # Other repetitions - NOT allowed even with wake word
        is_halluc, _ = self.recognizer.is_likely_hallucination(
            "ciao ciao", 
            confidence=0.8,
            has_wake_word=True
        )
        assert is_halluc, "Should detect non-wake-word repetition even with wake word active"
    
    def test_detects_alternating_repetitions(self):
        """Should detect alternating patterns like 'si no si no'"""
        test_cases = [
            "si no si no",
            "ciao buongiorno ciao buongiorno",
        ]
        
        for text in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence=0.8)
            assert is_halluc, f"Should detect alternating repetition in '{text}'"
            assert "repetition" in reason
    
    # ==================== Common Hallucination Tests ====================
    
    def test_detects_common_single_words_without_wake_word(self):
        """Should detect common hallucinations when no wake word"""
        test_cases = [
            "Ciao",
            "Buongiorno",
            "Buonanotte", 
            "Grazie",
            "Sì",
            "No",
            "Ok",
        ]
        
        for text in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(
                text, 
                confidence=0.8,
                has_wake_word=False
            )
            assert is_halluc, f"Should detect '{text}' as hallucination without wake word"
            assert "only" in reason, f"Expected 'only' in reason: {reason}"
    
    def test_allows_common_words_with_wake_word(self):
        """Should STILL filter single common words even with wake word (prevents hallucinations after wake)"""
        single_words = ["Ciao", "Grazie", "Sì", "No", "Buonanotte"]
        
        for text in single_words:
            is_halluc, _ = self.recognizer.is_likely_hallucination(
                text,
                confidence=0.8,
                has_wake_word=True
            )
            # Changed: Now we filter even with wake word for single words
            assert is_halluc, f"Should filter single word '{text}' even with wake word to prevent hallucinations"
        
        # But allow if part of longer phrase
        is_halluc, _ = self.recognizer.is_likely_hallucination(
            "Sì grazie",
            confidence=0.8,
            has_wake_word=True
        )
        assert not is_halluc, "Should allow multi-word phrase with wake word"
    
    # ==================== Confidence Tests ====================
    
    def test_detects_extremely_low_confidence(self):
        """Should flag transcriptions with very low confidence"""
        is_halluc, reason = self.recognizer.is_likely_hallucination(
            "qualche parola",
            confidence=0.2,  # Very low
            has_wake_word=False
        )
        assert is_halluc, "Should detect low confidence transcription"
        assert "confidence" in reason
    
    def test_detects_short_text_with_medium_confidence(self):
        """Should flag single words with medium confidence"""
        is_halluc, reason = self.recognizer.is_likely_hallucination(
            "parola",
            confidence=0.55,  # Medium-low
            has_wake_word=False
        )
        assert is_halluc, "Should detect short text with medium confidence"
        assert "confidence" in reason
    
    # ==================== Lexical Diversity Tests ====================
    
    def test_detects_low_lexical_diversity(self):
        """Should detect phrases with low vocabulary variety"""
        # 10 words, only 3 unique = 30% diversity
        text = "la la la casa casa casa bella bella bella casa"
        
        is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence=0.8)
        assert is_halluc, "Should detect low lexical diversity"
        assert "lexical" in reason or "diversity" in reason
    
    # ==================== Legitimate Speech Tests ====================
    
    def test_allows_legitimate_questions(self):
        """Should allow real questions"""
        test_cases = [
            ("Quanto fa due più due?", 0.9),
            ("Che ore sono adesso?", 0.85),
            ("Come ti chiami tu?", 0.8),
            ("Raccontami una storia per favore", 0.9),
        ]
        
        for text, confidence in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence)
            assert not is_halluc, f"Should allow legitimate question: '{text}' (reason: {reason})"
    
    def test_allows_legitimate_commands(self):
        """Should allow real commands"""
        test_cases = [
            ("Accendi la luce per favore", 0.85),
            ("Dimmi qualcosa di interessante", 0.9),
            ("Mostrami un trucco divertente", 0.88),
        ]
        
        for text, confidence in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence)
            assert not is_halluc, f"Should allow legitimate command: '{text}' (reason: {reason})"
    
    def test_allows_normal_conversation(self):
        """Should allow normal conversational phrases"""
        test_cases = [
            ("Buongiorno come stai oggi?", 0.9),
            ("No grazie non mi serve", 0.85),
            ("Sì va bene d'accordo", 0.8),
        ]
        
        for text, confidence in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text, confidence)
            assert not is_halluc, f"Should allow conversation: '{text}' (reason: {reason})"
    
    # ==================== Edge Cases ====================
    
    def test_handles_empty_text(self):
        """Should handle empty text gracefully"""
        test_cases = ["", "   ", None]
        
        for text in test_cases:
            is_halluc, reason = self.recognizer.is_likely_hallucination(text or "", confidence=0.8)
            assert is_halluc, f"Should detect empty text as hallucination"
            assert "empty" in reason
    
    def test_handles_mixed_case(self):
        """Should handle case-insensitive matching"""
        test_cases = [
            "CIAO",
            "BuOnGiOrNo",
            "SI SI SI",
        ]
        
        for text in test_cases:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence=0.8)
            assert is_halluc, f"Should detect '{text}' regardless of case"
    
    def test_handles_punctuation(self):
        """Should handle text with punctuation"""
        test_cases = [
            ("Ciao!", 0.8, False),  # Single word with punctuation - still hallucination
            ("Come stai?", 0.9, False),  # Real question - legitimate
        ]
        
        for text, confidence, has_wake in test_cases:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake)
            expected = text == "Ciao!"
            assert is_halluc == expected, f"Unexpected result for '{text}'"


class TestHallucinationIntegration:
    """Test hallucination detection in real-world scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.recognizer = SpeechRecognizer(api_key="dummy", language="it")
    
    def test_user_scenario_background_tv(self):
        """Scenario: User watching TV in background"""
        # Typical TV/video hallucinations
        tv_phrases = [
            ("Al prossimo episodio!", 0.85),
            ("Grazie per l'attenzione", 0.8),
            ("00:15", 0.9),
        ]
        
        for text, confidence in tv_phrases:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake_word=False)
            assert is_halluc, f"Should ignore TV audio: '{text}'"
    
    def test_user_scenario_conversation_with_others(self):
        """Scenario: User talking to another person (not Vector)"""
        # Common conversational fragments
        fragments = [
            ("Ciao", 0.8),
            ("Sì", 0.85),
            ("Buonanotte", 0.9),
            ("Ok ok", 0.75),
        ]
        
        for text, confidence in fragments:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake_word=False)
            assert is_halluc, f"Should ignore conversation fragment: '{text}'"
    
    def test_user_scenario_intentional_command(self):
        """Scenario: User intentionally talking to Vector"""
        # User says wake word + command
        commands = [
            ("Ciao Vector dimmi che ore sono", 0.9, True),
            ("Vector quanto fa due più due", 0.85, True),
            ("Ehi Vector raccontami una barzelletta", 0.88, True),
        ]
        
        for text, confidence, has_wake in commands:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake)
            assert not is_halluc, f"Should process intentional command: '{text}'"
    
    def test_user_scenario_follow_up_response(self):
        """Scenario: User responding to Vector's question (wake word active)"""
        # During active conversation (has_wake_word=True represents conversation_active)
        # Single-word responses should STILL be filtered (likely hallucinations)
        single_responses = [
            ("Sì", 0.85, True),      # Should be FILTERED even in conversation
            ("No", 0.9, True),       # Should be FILTERED even in conversation
            ("Buonanotte", 0.88, True),  # Should be FILTERED even in conversation
        ]
        
        for text, confidence, has_wake in single_responses:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake)
            assert is_halluc, f"Should filter single word '{text}' even during conversation (likely hallucination)"
        
        # Multi-word responses are allowed
        multi_word_responses = [
            ("No grazie", 0.9, True),
            ("Sì va bene", 0.88, True),
            ("Va bene d'accordo", 0.85, True),
        ]
        
        for text, confidence, has_wake in multi_word_responses:
            is_halluc, _ = self.recognizer.is_likely_hallucination(text, confidence, has_wake)
            assert not is_halluc, f"Should allow multi-word response: '{text}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
