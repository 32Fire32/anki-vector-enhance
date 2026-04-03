"""
Test suite for AnimationMapper

Tests emotion→animation mapping with weighted selection,
intensity variants, and probability thresholds.
"""

from vector_personality.behavior.animation_mapper import AnimationMapper
import pytest


def test_add_and_pick():
    """Test adding mappings and picking animations."""
    am = AnimationMapper()
    # use a temporary in-memory mapping file by swapping attribute
    am.mapping = {}
    am.add_mapping('joy', 'AlreadyAtFace', weight=2.0)
    am.add_mapping('joy', 'BlackJack_RtpPlayerYes', weight=1.0)
    picked = am.pick_animation('joy')
    assert picked in ('AlreadyAtFace', 'BlackJack_RtpPlayerYes')


def test_no_mapping():
    """Test behavior when no mapping exists for emotion."""
    am = AnimationMapper()
    am.mapping = {}
    assert am.pick_animation('nonexistent_emotion') is None


def test_should_trigger():
    """Test probabilistic triggering."""
    am = AnimationMapper()
    am.mapping = {'joy': {'probability': 0.0}}
    # With 0 probability, should never trigger
    assert am.should_trigger('joy') is False
    
    am.mapping = {'joy': {'probability': 1.0}}
    # With 1.0 probability, should always trigger
    assert am.should_trigger('joy') is True


def test_intensity_variants():
    """Test intensity-based animation selection."""
    am = AnimationMapper()
    am.mapping = {
        'joy': {
            'candidates': {'AlreadyAtFace': 1.0},
            'intensity_variants': {
                'high': {'ComeHereSuccess': 1.0},
                'low': {'BlackJack_RtpPlayerYes': 1.0}
            }
        }
    }
    # High intensity should pick from high variant
    high = am.pick_animation('joy', intensity=0.9)
    assert high == 'ComeHereSuccess'
    
    # Low intensity should pick from low variant
    low = am.pick_animation('joy', intensity=0.2)
    assert low == 'BlackJack_RtpPlayerYes'
    
    # Medium should use default
    medium = am.pick_animation('joy', intensity=0.5)
    assert medium == 'AlreadyAtFace'


def test_weighted_choice():
    """Test weighted random selection."""
    am = AnimationMapper()
    am.mapping = {
        'curiosity': {
            'candidates': {
                'BumpObjectSlowGetIn': 3.0,
                'ExploringLookAround': 2.0,
                'ExploringQuickScan': 1.0
            }
        }
    }
    # Run multiple times to verify all candidates can be selected
    results = set()
    for _ in range(100):
        pick = am.pick_animation('curiosity')
        results.add(pick)
    
    # Should have selected at least 2 of the 3 options in 100 tries
    assert len(results) >= 2
    assert all(r in ['BumpObjectSlowGetIn', 'ExploringLookAround', 'ExploringQuickScan'] 
               for r in results)


def test_real_mappings_loaded():
    """Test that real emotion mappings are loaded from JSON file."""
    am = AnimationMapper()
    
    # Verify key emotions are present
    assert 'joy' in am.mapping
    assert 'sadness' in am.mapping
    assert 'curiosity' in am.mapping
    assert 'confusion' in am.mapping
    
    # Verify joy has candidates
    assert 'candidates' in am.mapping['joy']
    assert len(am.mapping['joy']['candidates']) > 0
    
    # Verify probability is set
    assert 'probability' in am.mapping['joy']
    assert 0.0 <= am.mapping['joy']['probability'] <= 1.0


def test_observed_animations_mapping():
    """Test that observed animations are mapped correctly."""
    am = AnimationMapper()
    
    # Test joy animations from observations
    joy_candidates = am.mapping.get('joy', {}).get('candidates', {})
    assert 'AlreadyAtFace' in joy_candidates  # gioia
    assert 'ComeHereSuccess' in joy_candidates  # Molta gioia
    
    # Test confusion animations
    confusion_candidates = am.mapping.get('confusion', {}).get('candidates', {})
    assert 'AudioOnlyHuh' in confusion_candidates  # confuso
    assert 'ChargerDockingAlreadyHere' in confusion_candidates  # Confuso
    
    # Test sadness animations
    sadness_candidates = am.mapping.get('sadness', {}).get('candidates', {})
    assert 'ChargerDockingRequestGetout' in sadness_candidates  # Triste
    assert 'ConnectToCubeFailure' in sadness_candidates  # Triste 3
    
    # Test curiosity animations
    curiosity_candidates = am.mapping.get('curiosity', {}).get('candidates', {})
    assert 'ExploringLookAround' in curiosity_candidates  # Si guarda attorno
    assert 'ExploringQuickScan' in curiosity_candidates  # Perlustrare


def test_intensity_variants_for_key_emotions():
    """Test that key emotions have intensity variants configured."""
    am = AnimationMapper()
    
    # Joy should have high intensity variant
    assert 'intensity_variants' in am.mapping['joy']
    assert 'high' in am.mapping['joy']['intensity_variants']
    
    # Curiosity should have high and low variants
    if 'intensity_variants' in am.mapping['curiosity']:
        variants = am.mapping['curiosity']['intensity_variants']
        assert 'high' in variants or 'low' in variants


def test_pick_animation_returns_valid_trigger():
    """Test that picked animations are valid trigger names."""
    am = AnimationMapper()
    
    for emotion in ['joy', 'sadness', 'curiosity', 'confusion', 'fear']:
        if emotion in am.mapping:
            trigger = am.pick_animation(emotion)
            # Should return a non-empty string or None
            assert trigger is None or (isinstance(trigger, str) and len(trigger) > 0)


def test_all_emotions_have_valid_structure():
    """Test that all emotions in mapping have valid structure."""
    am = AnimationMapper()
    
    for emotion, config in am.mapping.items():
        if emotion.startswith('_'):  # Skip metadata fields
            continue
            
        # Should have candidates dict
        assert 'candidates' in config, f"{emotion} missing 'candidates'"
        assert isinstance(config['candidates'], dict), f"{emotion} candidates not a dict"
        
        # Should have probability
        if 'probability' in config:
            prob = config['probability']
            assert 0.0 <= prob <= 1.0, f"{emotion} probability out of range: {prob}"
        
        # Weights should be positive
        for trigger, weight in config['candidates'].items():
            assert weight > 0, f"{emotion}/{trigger} has invalid weight: {weight}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

