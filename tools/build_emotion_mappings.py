#!/usr/bin/env python3
"""Helper to build emotion_animation_mapping.json from observations.

Usage:
  python tools/build_emotion_mappings.py

This script reads animation_observations.json and helps you interactively
assign triggers to emotions. You can also edit emotion_animation_mapping.json directly.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBS_FILE = ROOT / 'tools' / 'animation_observations.json'
EMO_FILE = ROOT / 'tools' / 'emotion_animation_mapping.json'

EMOTIONS = [
    'joy', 'celebration', 'sadness', 'anger', 'surprise', 'fear',
    'greeting', 'apology', 'thanks', 'confused', 'bored', 'excited'
]


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    observations = load_json(OBS_FILE)
    emotions = load_json(EMO_FILE)

    print('=== Build Emotion→Animation Mappings ===')
    print(f'Loaded {len(observations)} observed triggers from {OBS_FILE.name}')
    print(f'Current emotion mappings: {len(emotions)} emotions\n')

    if not observations:
        print('No observations found. Run tools/run_animation_tests.py first.')
        return

    for trigger, data in observations.items():
        if trigger.startswith('_'):
            continue
        
        observed = data.get('observed', 'no description')
        print(f'\nTrigger: {trigger}')
        print(f'Observed: {observed}')
        
        resp = input('Assign to emotion (comma-separated) or skip [Enter]: ').strip()
        if not resp:
            continue
        
        assigned = [e.strip() for e in resp.split(',')]
        
        for emo in assigned:
            if emo not in emotions:
                emotions[emo] = {
                    'candidates': {},
                    'probability': 0.3
                }
            
            weight = input(f'  Weight for {emo} (default 1.0): ').strip()
            try:
                weight = float(weight) if weight else 1.0
            except ValueError:
                weight = 1.0
            
            emotions[emo]['candidates'][trigger] = weight
    
    save_json(EMO_FILE, emotions)
    print(f'\nSaved to {EMO_FILE}')


if __name__ == '__main__':
    main()
