#!/usr/bin/env python3
"""
Validator for animation mapping JSON files.

Validates both animation_observations.json and emotion_animation_mapping.json
against the schema and enforces naming rules, weight consistency, and safety requirements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("Error: jsonschema library not installed. Run: pip install jsonschema")
    sys.exit(1)


class AnimationMappingValidator:
    """Validates animation mapping files against schema and business rules."""
    
    # Naming rules
    ANIMATION_PATTERN = re.compile(r'^(?:anim_[a-z0-9_]+|[A-Z][A-Za-z0-9_]+)$')
    HUMAN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 _\-'àèéìòù]+$")
    HUMAN_NAME_MIN_LENGTH = 3
    HUMAN_NAME_MAX_LENGTH = 50
    DESCRIPTION_MIN_LENGTH = 10
    DESCRIPTION_MAX_LENGTH = 200
    
    # Valid emotion categories
    VALID_EMOTIONS = {
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        'contentment', 'excitement', 'curiosity', 'confusion',
        'frustration', 'affection', 'pride', 'shame', 'neutral'
    }
    
    # Valid safety levels
    VALID_SAFETY = {'safe', 'sensitive', 'unknown'}
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator with JSON schema."""
        if schema_path is None:
            schema_path = Path(__file__).parent / 'animation_mapping.schema.json'
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
    
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a JSON file against schema and business rules.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Load JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]
        
        # Validate against JSON Schema
        try:
            validate(instance=data, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            errors.append(f"  Path: {' > '.join(str(p) for p in e.path)}")
        
        # Determine file type and run specific validations
        if 'observations' in data:
            errors.extend(self._validate_observations(data))
        elif 'mappings' in data:
            errors.extend(self._validate_mappings(data))
        else:
            errors.append("Unknown file type: must contain 'observations' or 'mappings'")
        
        return len(errors) == 0, errors
    
    def _validate_observations(self, data: Dict) -> List[str]:
        """Validate animation observations file."""
        errors = []
        
        observations = data.get('observations', [])
        seen_triggers = set()
        seen_names = set()
        
        for idx, obs in enumerate(observations):
            prefix = f"Observation {idx + 1}"
            
            # Check for duplicate triggers
            trigger = obs.get('trigger', '')
            if trigger in seen_triggers:
                errors.append(f"{prefix}: Duplicate trigger '{trigger}'")
            seen_triggers.add(trigger)
            
            # Check for duplicate human names
            name = obs.get('name', '')
            if name.lower() in seen_names:
                errors.append(f"{prefix}: Duplicate name '{name}' (case-insensitive)")
            seen_names.add(name.lower())
            
            # Validate naming patterns
            if not self.ANIMATION_PATTERN.match(trigger):
                errors.append(
                    f"{prefix}: Invalid trigger format '{trigger}'. "
                    f"Must match pattern: anim_[a-z0-9_]+"
                )
            
            if not self.HUMAN_NAME_PATTERN.match(name):
                errors.append(
                    f"{prefix}: Invalid name format '{name}'. "
                    f"Allowed: letters, numbers, spaces, dashes, apostrophes, Italian accents"
                )
            
            if len(name) < self.HUMAN_NAME_MIN_LENGTH:
                errors.append(
                    f"{prefix}: Name too short '{name}'. "
                    f"Minimum {self.HUMAN_NAME_MIN_LENGTH} characters"
                )
            
            if len(name) > self.HUMAN_NAME_MAX_LENGTH:
                errors.append(
                    f"{prefix}: Name too long '{name}'. "
                    f"Maximum {self.HUMAN_NAME_MAX_LENGTH} characters"
                )
            
            # Validate description
            description = obs.get('description', '')
            if len(description) < self.DESCRIPTION_MIN_LENGTH:
                errors.append(
                    f"{prefix}: Description too short. "
                    f"Minimum {self.DESCRIPTION_MIN_LENGTH} characters"
                )
            
            if len(description) > self.DESCRIPTION_MAX_LENGTH:
                errors.append(
                    f"{prefix}: Description too long. "
                    f"Maximum {self.DESCRIPTION_MAX_LENGTH} characters"
                )
            
            # Validate safety level
            safety = obs.get('safety', '')
            if safety not in self.VALID_SAFETY:
                errors.append(
                    f"{prefix}: Invalid safety level '{safety}'. "
                    f"Must be one of: {', '.join(self.VALID_SAFETY)}"
                )
            
            # Validate date format
            try:
                datetime.strptime(obs.get('observed_date', ''), '%Y-%m-%d')
            except ValueError:
                errors.append(
                    f"{prefix}: Invalid date format. Must be YYYY-MM-DD"
                )
        
        return errors
    
    def _validate_mappings(self, data: Dict) -> List[str]:
        """Validate emotion animation mapping file."""
        errors = []
        
        mappings = data.get('mappings', [])
        seen_emotions = set()
        
        for idx, mapping in enumerate(mappings):
            prefix = f"Mapping {idx + 1}"
            
            emotion = mapping.get('emotion', '')
            
            # Check for duplicate emotions
            if emotion in seen_emotions:
                errors.append(f"{prefix}: Duplicate emotion '{emotion}'")
            seen_emotions.add(emotion)
            
            # Validate emotion category
            if emotion not in self.VALID_EMOTIONS:
                errors.append(
                    f"{prefix}: Invalid emotion '{emotion}'. "
                    f"Must be one of: {', '.join(sorted(self.VALID_EMOTIONS))}"
                )
            
            # Validate candidates
            candidates = mapping.get('candidates', [])
            if not candidates:
                errors.append(f"{prefix}: No candidates specified for emotion '{emotion}'")
                continue
            
            # Check weight sum
            total_weight = sum(c.get('weight', 0) for c in candidates)
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
                errors.append(
                    f"{prefix}: Weights for emotion '{emotion}' sum to {total_weight:.4f}, "
                    f"must sum to 1.0"
                )
            
            # Validate individual candidates
            for cidx, candidate in enumerate(candidates):
                trigger = candidate.get('trigger', '')
                weight = candidate.get('weight', 0)
                
                if not self.ANIMATION_PATTERN.match(trigger):
                    errors.append(
                        f"{prefix}.{cidx + 1}: Invalid trigger format '{trigger}'"
                    )
                
                if not (0.0 <= weight <= 1.0):
                    errors.append(
                        f"{prefix}.{cidx + 1}: Weight {weight} out of range [0.0, 1.0]"
                    )
        
        # Check coverage - warn if standard emotions are missing
        missing_emotions = self.VALID_EMOTIONS - seen_emotions
        if missing_emotions:
            # This is a warning, not an error
            print(f"Warning: Missing mappings for emotions: {', '.join(sorted(missing_emotions))}")
        
        return errors
    
    def validate_cross_file(
        self,
        observations_path: Path,
        mappings_path: Path
    ) -> Tuple[bool, List[str]]:
        """
        Validate consistency between observations and mappings files.
        
        Checks that all triggers referenced in mappings exist in observations,
        and warns about unused observations.
        """
        errors = []
        
        # Load both files
        try:
            with open(observations_path, 'r', encoding='utf-8') as f:
                obs_data = json.load(f)
            with open(mappings_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
        except Exception as e:
            return False, [f"Error loading files: {e}"]
        
        # Extract triggers from observations
        observed_triggers = {
            obs['trigger']
            for obs in obs_data.get('observations', [])
        }
        
        # Extract triggers from mappings
        mapped_triggers = set()
        for mapping in map_data.get('mappings', []):
            for candidate in mapping.get('candidates', []):
                mapped_triggers.add(candidate['trigger'])
        
        # Check for missing observations
        missing = mapped_triggers - observed_triggers
        if missing:
            errors.append(
                f"Mappings reference {len(missing)} triggers not in observations: "
                f"{', '.join(sorted(list(missing)[:5]))}"
                + ("..." if len(missing) > 5 else "")
            )
        
        # Warn about unused observations
        unused = observed_triggers - mapped_triggers
        if unused:
            print(
                f"Warning: {len(unused)} observed animations not used in mappings: "
                f"{', '.join(sorted(list(unused)[:5]))}"
                + ("..." if len(unused) > 5 else "")
            )
        
        return len(errors) == 0, errors


def main():
    """Command-line interface for validator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Vector animation mapping JSON files'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='JSON file(s) to validate'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        help='Path to JSON schema file (default: animation_mapping.schema.json in same dir)'
    )
    parser.add_argument(
        '--cross-check',
        nargs=2,
        metavar=('OBSERVATIONS', 'MAPPINGS'),
        type=Path,
        help='Cross-validate observations and mappings files for consistency'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    try:
        validator = AnimationMappingValidator(args.schema)
    except Exception as e:
        print(f"Error initializing validator: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate files
    all_valid = True
    
    for file_path in args.files:
        print(f"\nValidating: {file_path}")
        print("-" * 60)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            all_valid = False
            continue
        
        is_valid, errors = validator.validate_file(file_path)
        
        if is_valid:
            print(f"✅ Valid")
        else:
            print(f"❌ Validation failed with {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
            all_valid = False
    
    # Cross-validation if requested
    if args.cross_check:
        obs_path, map_path = args.cross_check
        print(f"\nCross-validating: {obs_path.name} ↔ {map_path.name}")
        print("-" * 60)
        
        is_valid, errors = validator.validate_cross_file(obs_path, map_path)
        
        if is_valid:
            print(f"✅ Cross-validation passed")
        else:
            print(f"❌ Cross-validation failed:")
            for error in errors:
                print(f"  - {error}")
            all_valid = False
    
    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
