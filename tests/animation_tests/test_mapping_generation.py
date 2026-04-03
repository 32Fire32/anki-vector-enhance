"""
Validation tests for animation mapping files.

Tests schema validation, naming rules, weight consistency,
and cross-file validation for animation mapping system.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

# Import validator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

try:
    from validate_animation_mapping import AnimationMappingValidator
except ImportError:
    pytest.skip("validate_animation_mapping not available", allow_module_level=True)


@pytest.fixture
def validator():
    """Create validator instance."""
    schema_path = Path(__file__).parent.parent.parent / 'tools' / 'animation_mapping.schema.json'
    if not schema_path.exists():
        pytest.skip(f"Schema file not found: {schema_path}")
    return AnimationMappingValidator(schema_path)


@pytest.fixture
def sample_observations():
    """Create sample observations data."""
    return {
        "schema_version": "1.0",
        "observations": [
            {
                "trigger": "anim_pounce_success_02",
                "name": "Salto Vittorioso",
                "description": "Vector salta in avanti con le ruote sollevate mostrando soddisfazione",
                "safety": "safe",
                "observed_date": "2025-12-28"
            },
            {
                "trigger": "anim_fistbump_success_01",
                "name": "High Five",
                "description": "Vector solleva il braccio per fare il five con l'operatore",
                "safety": "safe",
                "observed_date": "2025-12-28"
            }
        ],
        "metadata": {
            "created_date": "2025-12-28T10:00:00",
            "last_updated": "2025-12-28T10:00:00",
            "operator": "Test Operator"
        }
    }


@pytest.fixture
def sample_mappings():
    """Create sample emotion mappings data."""
    return {
        "schema_version": "1.0",
        "mappings": [
            {
                "emotion": "joy",
                "candidates": [
                    {"trigger": "anim_pounce_success_02", "weight": 0.6},
                    {"trigger": "anim_fistbump_success_01", "weight": 0.4}
                ],
                "selection_strategy": "weighted_random"
            }
        ],
        "metadata": {
            "generated_date": "2025-12-28T10:00:00",
            "source_file": "tools/animation_observations.json"
        }
    }


class TestSchemaValidation:
    """Test JSON schema validation."""
    
    def test_valid_observations(self, validator, sample_observations, tmp_path):
        """Test that valid observations pass schema validation."""
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert is_valid, f"Validation failed: {errors}"
        assert len(errors) == 0
    
    def test_valid_mappings(self, validator, sample_mappings, tmp_path):
        """Test that valid mappings pass schema validation."""
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert is_valid, f"Validation failed: {errors}"
        assert len(errors) == 0
    
    def test_invalid_schema_version(self, validator, sample_observations, tmp_path):
        """Test that wrong schema version fails validation."""
        sample_observations["schema_version"] = "2.0"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("schema_version" in str(e).lower() for e in errors)
    
    def test_missing_required_field(self, validator, sample_observations, tmp_path):
        """Test that missing required field fails validation."""
        del sample_observations["observations"][0]["trigger"]
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid


class TestNamingRules:
    """Test naming pattern enforcement."""
    
    def test_valid_trigger_pattern(self, validator, sample_observations, tmp_path):
        """Test that valid trigger names pass validation."""
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert is_valid
    
    def test_invalid_trigger_pattern(self, validator, sample_observations, tmp_path):
        """Test that invalid trigger format fails validation."""
        # Use characters that don't match allowed patterns (no leading uppercase and not anim_)
        sample_observations["observations"][0]["trigger"] = "invalid-trigger!"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("trigger format" in str(e).lower() for e in errors)

    def test_camelcase_trigger_allowed(self, validator, sample_observations, tmp_path):
        """Test that CamelCase trigger names from docs are accepted."""
        sample_observations["observations"][0]["trigger"] = "AlreadyAtFace"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert is_valid, f"CamelCase trigger should be accepted but failed: {errors}"
    
    def test_name_length_limits(self, validator, sample_observations, tmp_path):
        """Test that name length limits are enforced."""
        # Too short
        sample_observations["observations"][0]["name"] = "AB"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("too short" in str(e).lower() for e in errors)
        
        # Too long
        sample_observations["observations"][0]["name"] = "A" * 51
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("too long" in str(e).lower() for e in errors)
    
    def test_italian_characters_allowed(self, validator, sample_observations, tmp_path):
        """Test that Italian accented characters are allowed."""
        sample_observations["observations"][0]["name"] = "Più Felice"
        sample_observations["observations"][1]["name"] = "Città Grande"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert is_valid
    
    def test_invalid_characters_rejected(self, validator, sample_observations, tmp_path):
        """Test that invalid characters are rejected."""
        sample_observations["observations"][0]["name"] = "Name@Invalid!"
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("name format" in str(e).lower() for e in errors)


class TestWeightConsistency:
    """Test weight validation in emotion mappings."""
    
    def test_weights_sum_to_one(self, validator, sample_mappings, tmp_path):
        """Test that weights summing to 1.0 pass validation."""
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert is_valid
    
    def test_weights_not_summing_to_one(self, validator, sample_mappings, tmp_path):
        """Test that weights not summing to 1.0 fail validation."""
        sample_mappings["mappings"][0]["candidates"][0]["weight"] = 0.5
        sample_mappings["mappings"][0]["candidates"][1]["weight"] = 0.3
        # Sum = 0.8, not 1.0
        
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert not is_valid
        assert any("sum" in str(e).lower() for e in errors)
    
    def test_weight_out_of_range(self, validator, sample_mappings, tmp_path):
        """Test that weights outside [0, 1] fail validation."""
        sample_mappings["mappings"][0]["candidates"][0]["weight"] = 1.5
        
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert not is_valid


class TestDuplicateDetection:
    """Test duplicate detection."""
    
    def test_duplicate_triggers(self, validator, sample_observations, tmp_path):
        """Test that duplicate triggers are detected."""
        sample_observations["observations"].append(
            sample_observations["observations"][0].copy()
        )
        
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("duplicate trigger" in str(e).lower() for e in errors)
    
    def test_duplicate_names_case_insensitive(self, validator, sample_observations, tmp_path):
        """Test that duplicate names (case-insensitive) are detected."""
        sample_observations["observations"][0]["trigger"] = "anim_test_01"
        sample_observations["observations"][1]["trigger"] = "anim_test_02"
        sample_observations["observations"][0]["name"] = "Salto Felice"
        sample_observations["observations"][1]["name"] = "salto felice"  # Same name, different case
        
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("duplicate name" in str(e).lower() for e in errors)
    
    def test_duplicate_emotions(self, validator, sample_mappings, tmp_path):
        """Test that duplicate emotion mappings are detected."""
        sample_mappings["mappings"].append(
            sample_mappings["mappings"][0].copy()
        )
        
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert not is_valid
        assert any("duplicate emotion" in str(e).lower() for e in errors)


class TestCrossValidation:
    """Test cross-file validation."""
    
    def test_all_mapped_triggers_have_observations(
        self, validator, sample_observations, sample_mappings, tmp_path
    ):
        """Test that all triggers in mappings exist in observations."""
        obs_file = tmp_path / "observations.json"
        map_file = tmp_path / "mappings.json"
        
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_cross_file(obs_file, map_file)
        assert is_valid
    
    def test_missing_observations_detected(
        self, validator, sample_observations, sample_mappings, tmp_path
    ):
        """Test that missing observations are detected in cross-validation."""
        # Add a trigger to mappings that doesn't exist in observations
        sample_mappings["mappings"][0]["candidates"].append({
            "trigger": "anim_nonexistent_trigger",
            "weight": 0.0
        })
        # Adjust weights to sum to 1.0
        sample_mappings["mappings"][0]["candidates"][0]["weight"] = 0.5
        sample_mappings["mappings"][0]["candidates"][1]["weight"] = 0.5
        
        obs_file = tmp_path / "observations.json"
        map_file = tmp_path / "mappings.json"
        
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_cross_file(obs_file, map_file)
        assert not is_valid
        assert any("not in observations" in str(e).lower() for e in errors)


class TestSafetyLevels:
    """Test safety level validation."""
    
    def test_valid_safety_levels(self, validator, sample_observations, tmp_path):
        """Test that valid safety levels pass validation."""
        sample_observations["observations"][0]["safety"] = "safe"
        sample_observations["observations"][1]["safety"] = "sensitive"
        
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert is_valid
    
    def test_invalid_safety_level(self, validator, sample_observations, tmp_path):
        """Test that invalid safety level fails validation."""
        sample_observations["observations"][0]["safety"] = "invalid_level"
        
        obs_file = tmp_path / "observations.json"
        with open(obs_file, 'w') as f:
            json.dump(sample_observations, f)
        
        is_valid, errors = validator.validate_file(obs_file)
        assert not is_valid
        assert any("safety" in str(e).lower() for e in errors)


class TestEmotionCategories:
    """Test emotion category validation."""
    
    def test_valid_emotion_categories(self, validator, sample_mappings, tmp_path):
        """Test that valid emotion categories pass validation."""
        valid_emotions = ["joy", "sadness", "anger", "fear", "surprise"]
        
        for i, emotion in enumerate(valid_emotions):
            mapping = {
                "emotion": emotion,
                "candidates": [
                    {"trigger": f"anim_test_{i}", "weight": 1.0}
                ],
                "selection_strategy": "weighted_random"
            }
            sample_mappings["mappings"].append(mapping)
        
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        # May have weight validation errors, but emotion categories should be valid
        # Check that no emotion category errors exist
        emotion_errors = [e for e in errors if "emotion" in str(e).lower() and "invalid emotion" in str(e).lower()]
        assert len(emotion_errors) == 0
    
    def test_invalid_emotion_category(self, validator, sample_mappings, tmp_path):
        """Test that invalid emotion category fails validation."""
        sample_mappings["mappings"][0]["emotion"] = "invalid_emotion_xyz"
        
        map_file = tmp_path / "mappings.json"
        with open(map_file, 'w') as f:
            json.dump(sample_mappings, f)
        
        is_valid, errors = validator.validate_file(map_file)
        assert not is_valid
        assert any("invalid emotion" in str(e).lower() for e in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
