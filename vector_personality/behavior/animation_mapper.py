"""Animation mapper: map emotions/events -> animation triggers.

This module provides a lightweight mapping utility that loads emotion→trigger
mappings from `tools/emotion_animation_mapping.json`.

Structure: {emotion: {candidates: {trigger: weight}, probability: 0.0-1.0, intensity_variants: {...}}}
"""
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[2]
MAPPING_FILE = ROOT / 'tools' / 'emotion_animation_mapping.json'


class AnimationMapper:
    def __init__(self, mapping_file: Path = MAPPING_FILE):
        self.mapping_file = Path(mapping_file)
        self.load()

    def load(self):
        try:
            self.mapping = json.loads(self.mapping_file.read_text(encoding='utf-8'))
        except Exception:
            self.mapping = {}

    def save(self):
        self.mapping_file.write_text(json.dumps(self.mapping, indent=2, ensure_ascii=False), encoding='utf-8')

    def add_mapping(self, emotion: str, trigger: str, weight: float = 1.0):
        self.mapping.setdefault(emotion, {})
        self.mapping[emotion].setdefault('candidates', {})
        self.mapping[emotion]['candidates'][trigger] = float(weight)
        self.save()

    def should_trigger(self, emotion: str) -> bool:
        """Check if animation should trigger based on probability."""
        emo = self.mapping.get(emotion)
        if not emo:
            return False
        prob = emo.get('probability', 0.3)  # default 30%
        return random.random() < prob

    def pick_animation(self, emotion: str, intensity: float = 1.0):
        """Pick a trigger name for the given emotion.

        Args:
            emotion: emotion name (joy, sadness, anger, etc.)
            intensity: 0.0-1.0 scale (can select different animations for high intensity)

        Returns trigger or None if no mapping available.
        """
        emo = self.mapping.get(emotion)
        if not emo:
            return None
        
        # Check intensity variants first
        if intensity > 0.7:
            variants = emo.get('intensity_variants', {}).get('high', {})
            if variants:
                return self._weighted_choice(variants)
        elif intensity < 0.3:
            variants = emo.get('intensity_variants', {}).get('low', {})
            if variants:
                return self._weighted_choice(variants)
        
        # Fallback to default candidates
        candidates = emo.get('candidates', {})
        if not candidates:
            return None
        return self._weighted_choice(candidates)
    
    def _weighted_choice(self, candidates: dict) -> str:
        """Weighted random choice from candidates dict."""
        if not candidates:
            return None
        items = list(candidates.items())
        triggers, weights = zip(*items)
        total = sum(weights)
        if total <= 0:
            return random.choice(triggers)
        r = random.random() * total
        upto = 0
        for t, w in items:
            upto += w
            if r <= upto:
                return t
        return triggers[-1]


animation_mapper = AnimationMapper()
