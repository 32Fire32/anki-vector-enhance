#!/usr/bin/env python3
"""
Interactive animation testing tool for Vector.

This script helps you observe Vector animations and record observations
in Italian following the animation mapping schema.

Usage:
    # Interactive mode (recommended)
    python tools/run_animation_tests.py --interactive
    
    # Resume from last session
    python tools/run_animation_tests.py --interactive --resume
    
    # Test specific animations
    python tools/run_animation_tests.py --interactive --filter "pounce"
    
    # Headless mode (requires observations file)
    python tools/run_animation_tests.py --headless --count 50
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import argparse

# Compatibility shim: some anki_vector code passes loop=<event_loop> to asyncio.Event
# newer Python versions removed the loop parameter; provide a thin wrapper to accept it.
import asyncio
_orig_Event = asyncio.Event
class _EventCompat(_orig_Event):
    def __init__(self, loop=None, *args, **kwargs):
        # Accept loop for compatibility but ignore it, call original initializer
        super().__init__(*args, **kwargs)
asyncio.Event = _EventCompat

try:
    import anki_vector
    from anki_vector.util import degrees
except ImportError:
    print("Error: anki_vector SDK not installed")
    print("Install with: pip install anki_vector")
    sys.exit(1)

# Reduce noisy warnings from SDK event handler
import logging
logging.getLogger("events.EventHandler").setLevel(logging.ERROR)
logging.getLogger("anki_vector.events").setLevel(logging.ERROR)

# Python 3.10+ compatibility fix
import sys
if sys.version_info >= (3, 10):
    import asyncio
    import collections
    # Patch for asyncio compatibility
    try:
        collections.Callable = collections.abc.Callable
    except AttributeError:
        pass


ROOT = Path(__file__).resolve().parents[1]
TRIGGERS_PATH = ROOT / 'tools' / 'animation_triggers.json'
OBSERVATIONS_PATH = ROOT / 'tools' / 'animation_observations.json'


def is_robot_ready(robot) -> bool:
    """Return True if robot appears to have a usable anim interface."""
    if robot is None:
        return False
    try:
        anim = getattr(robot, 'anim', None)
        if anim is None:
            return False
        if not hasattr(anim, 'play_animation_trigger'):
            return False
        return True
    except Exception:
        return False


def reconnect_to_vector(timeout: int = 30):
    """Attempt to create a new Robot connection and return it or None."""
    try:
        print(f"Attempting to reconnect to Vector (timeout={timeout}s)...")
        new_robot = anki_vector.Robot(enable_nav_map_feed=False, enable_face_detection=False)
        new_robot.connect(timeout=timeout)
        print("✅ Reconnected to Vector")
        return new_robot
    except Exception as e:
        print(f"❌ Reconnect failed: {e}")
        return None


def load_triggers() -> List[Dict]:
    """Load animation triggers from JSON file."""
    if not TRIGGERS_PATH.exists():
        print(f"Error: {TRIGGERS_PATH} not found")
        print("Run: python tools/extract_animation_triggers.py")
        sys.exit(1)
    
    with open(TRIGGERS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('triggers', [])


def _normalize_observations(raw: Dict) -> Dict:
    """Normalize older or malformed observations formats into the canonical schema.

    - If file is a dict mapping trigger->info, convert to observations list
    - Ensure keys: schema_version, observations (list), metadata (dict)
    """
    if raw is None:
        raw = {}

    # Case: legacy dict mapping trigger -> {observed: ...}
    if isinstance(raw, dict) and 'observations' not in raw and any(isinstance(v, dict) for v in raw.values()):
        observations_list = []
        for trigger, info in raw.items():
            obs = {
                'trigger': trigger,
                'name': info.get('name', ''),
                'description': info.get('observed') or info.get('description', ''),
                'safety': info.get('safety', 'unknown'),
                'observed_date': info.get('observed_date', datetime.now().strftime('%Y-%m-%d'))
            }
            if 'notes' in info:
                obs['notes'] = info['notes']
            observations_list.append(obs)

        normalized = {
            'schema_version': '1.0',
            'observations': observations_list,
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'operator': ''
            }
        }
        return normalized

    # If raw already has observations but metadata missing, add defaults
    normalized = {
        'schema_version': raw.get('schema_version', '1.0'),
        'observations': raw.get('observations', []) if isinstance(raw.get('observations', []), list) else [],
        'metadata': raw.get('metadata', {}) if isinstance(raw.get('metadata', {}), dict) else {}
    }

    # Ensure metadata keys
    if 'created_date' not in normalized['metadata']:
        normalized['metadata']['created_date'] = datetime.now().isoformat()
    if 'last_updated' not in normalized['metadata']:
        normalized['metadata']['last_updated'] = datetime.now().isoformat()
    if 'operator' not in normalized['metadata']:
        normalized['metadata']['operator'] = ''

    return normalized


def load_observations() -> Dict:
    """Load existing observations and normalize format."""
    if not OBSERVATIONS_PATH.exists():
        return {
            'schema_version': '1.0',
            'observations': [],
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'operator': ''
            }
        }

    try:
        with open(OBSERVATIONS_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception:
        # If file is corrupted or unreadable, return a fresh structure
        return {
            'schema_version': '1.0',
            'observations': [],
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'operator': ''
            }
        }

    return _normalize_observations(raw)


def save_observations(data: Dict):
    """Save observations to JSON file."""
    data['metadata']['last_updated'] = datetime.now().isoformat()
    
    with open(OBSERVATIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_observed_triggers(observations: Dict) -> set:
    """Get set of already observed triggers."""
    return {obs['trigger'] for obs in observations.get('observations', [])}


def play_animation(robot, trigger: str) -> tuple[bool, str]:
    """
    Play animation on Vector.

    Returns:
        Tuple of (success, message)
    """
    # Handle dry-run / disconnected robot gracefully
    if robot is None:
        print(f"  ▶ (dry-run) Would play: {trigger}")
        return False, "dry-run (no robot connected)"

    if not hasattr(robot, 'anim'):
        return False, "robot connection lost or missing 'anim' interface"

    try:
        print(f"  ▶ Playing animation: {trigger}")
        robot.anim.play_animation_trigger(trigger)
        try:
            robot.behavior.drive_off_charger()  # Ensure Vector can move
        except Exception:
            # Non-fatal if drive_off_charger not available or fails
            pass
        return True, "Played successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"


def interactive_observation(trigger: str) -> Optional[Dict]:
    """
    Prompt user for animation observation details.
    
    Returns observation dict or None if skipped.
    """
    print(f"\n{'='*60}")
    print(f"Animation Trigger: {trigger}")
    print(f"{'='*60}")
    
    # Italian name
    while True:
        name = input("Nome in italiano (3-50 char, invio per saltare): ").strip()
        if not name:
            print("⏭ Saltato")
            return None
        if 3 <= len(name) <= 50:
            break
        print("❌ Nome deve essere 3-50 caratteri")
    
    # Description
    while True:
        desc = input("Descrizione (10-200 char): ").strip()
        if 10 <= len(desc) <= 200:
            break
        print("❌ Descrizione deve essere 10-200 caratteri")
    
    # Safety
    while True:
        print("Safety level:")
        print("  1) safe - sempre appropriata")
        print("  2) sensitive - dipende dal contesto")
        print("  3) unknown - non ancora classificata")
        safety_choice = input("Scelta (1/2/3): ").strip()
        
        safety_map = {'1': 'safe', '2': 'sensitive', '3': 'unknown'}
        if safety_choice in safety_map:
            safety = safety_map[safety_choice]
            break
        print("❌ Scelta non valida")
    
    # Notes (optional)
    notes = input("Note aggiuntive (opzionale): ").strip()
    
    observation = {
        'trigger': trigger,
        'name': name,
        'description': desc,
        'safety': safety,
        'observed_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    if notes:
        observation['notes'] = notes
    
    return observation


def interactive_mode(
    robot,
    triggers: List[Dict],
    observations: Dict,
    resume: bool = False,
    filter_pattern: Optional[str] = None,
    start_trigger: Optional[str] = None
):
    """Run interactive animation testing.

    If `start_trigger` is provided, skip all triggers up to the first occurrence of
    that trigger (inclusive) within the filtered list. This makes it easy to resume
    from a specific point in a large run.
    """
    observed = get_observed_triggers(observations)
    
    # Filter triggers
    pending = [
        t for t in triggers
        if (not resume or t['trigger'] not in observed)
        and (not filter_pattern or filter_pattern.lower() in t['trigger'].lower())
    ]
    
    print(f"\n📊 Status:")
    print(f"  Total triggers: {len(triggers)}")
    print(f"  Already observed: {len(observed)}")
    print(f"  Pending: {len(pending)}")
    
    if filter_pattern:
        print(f"  Filter: '{filter_pattern}'")
    
    if not pending:
        print("\n✅ All animations already observed!")
        return

    # If user provided a start trigger, skip up to that trigger in the pending list
    if start_trigger:
        # Try exact or case-insensitive match within pending list first
        def match_name(name, target):
            return name == target or name.lower() == target.lower()

        start_index = None
        for idx, t in enumerate(pending):
            if match_name(t['trigger'], start_trigger):
                start_index = idx
                break

        if start_index is None:
            # Not in pending: search in the full triggers list to start after it
            full_index = None
            for idx, t in enumerate(triggers):
                if match_name(t['trigger'], start_trigger):
                    full_index = idx
                    break

            if full_index is None:
                print(f"⚠️ Start trigger '{start_trigger}' not found in trigger list. Proceeding from the beginning.")
            else:
                # Build pending list starting just after the found trigger, re-applying resume/filter rules
                new_pending = []
                for t in triggers[full_index + 1:]:
                    name = t['trigger']
                    if resume and name in observed:
                        continue
                    if filter_pattern and filter_pattern.lower() not in name.lower():
                        continue
                    new_pending.append(t)

                pending = new_pending
                print(f"⏩ Starting AFTER trigger '{start_trigger}' (now {len(pending)} pending)")
        else:
            pending = pending[start_index:]
            print(f"⏩ Starting from trigger '{start_trigger}' (now {len(pending)} pending)")
    
    # Get operator name
    if not observations['metadata'].get('operator'):
        operator = input("\nIl tuo nome (operatore): ").strip()
        observations['metadata']['operator'] = operator or 'Unknown'
    
    print(f"\n🎬 Starting interactive testing...")
    print(f"Commands: 'q' = quit, 's' = skip, 'r' = replay")
    
    for i, trigger_data in enumerate(pending):
        trigger = trigger_data['trigger']
        
        print(f"\n[{i+1}/{len(pending)}] Testing: {trigger}")
        
        # Check robot readiness and offer reconnect if needed
        if not is_robot_ready(robot):
            print("⚠️ Robot connection appears to be lost or anim interface is unavailable.")
            reconnect_choice = input("Attempt to reconnect now? (y/N): ").strip().lower()
            if reconnect_choice == 'y':
                new_robot = reconnect_to_vector(timeout=30)
                if new_robot:
                    robot = new_robot
                else:
                    print("Continuing in dry-run mode (you can still record observations manually)")
            else:
                print("Continuing in dry-run mode (you can still record observations manually)")

        # Play animation (may be dry-run if no robot connected)
        success, msg = play_animation(robot, trigger)
        if not success:
            print(f"❌ {msg}")
            # If dry-run, offer to continue without attempting play
            if msg.startswith('dry-run') or 'anim interface' in msg.lower():
                choice = input("Robot not connected / anim unavailable. Continue recording observations manually? (y/N): ").strip().lower()
                if choice != 'y':
                    print('Skipping recording for this trigger')
                    continue
            else:
                choice = input("Continue anyway? (y/N): ").strip().lower()
                if choice != 'y':
                    continue
        
        # Prompt for observation
        while True:
            cmd = input("\nRecord observation? (y/n/r=replay/q=quit): ").strip().lower()
            
            if cmd == 'q':
                print("💾 Saving and quitting...")
                save_observations(observations)
                return
            elif cmd == 's' or cmd == 'n':
                print("⏭ Skipped")
                break
            elif cmd == 'r':
                print("🔄 Replaying...")
                play_animation(robot, trigger)
                continue
            elif cmd == 'y' or cmd == '':
                # Record observation
                obs = interactive_observation(trigger)
                if obs:
                    observations['observations'].append(obs)
                    save_observations(observations)
                    print("✅ Saved!")
                break
    
    print(f"\n✅ Session complete! Observed {len(observations['observations'])} animations")


def headless_mode(robot, observations: Dict, count: int):
    """Run animations without user input (for testing)."""
    observed = get_observed_triggers(observations)
    
    print(f"Running {count} random animations from observations...")
    
    if not observations['observations']:
        print("Error: No observations found. Run interactive mode first.")
        return
    
    import random
    sample = random.sample(observations['observations'], min(count, len(observations['observations'])))
    
    for i, obs in enumerate(sample):
        trigger = obs['trigger']
        print(f"\n[{i+1}/{len(sample)}] {obs['name']} ({trigger})")
        success, msg = play_animation(robot, trigger)
        if not success:
            print(f"  ❌ {msg}")
        else:
            print(f"  ✅ {msg}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive animation testing for Vector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start interactive session
  python tools/run_animation_tests.py --interactive
  
  # Resume from last session
  python tools/run_animation_tests.py --interactive --resume
  
  # Test only "pounce" animations
  python tools/run_animation_tests.py --interactive --filter pounce
  
  # Run headless test
  python tools/run_animation_tests.py --headless --count 10
        '''
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode (recommended)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Headless mode (no user input)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last session (skip observed animations)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        help='Filter triggers by pattern'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='Number of animations to test in headless mode (default: 50)'
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Number of connection retries (default: 3)'
    )
    parser.add_argument(
        '--timeouts',
        type=str,
        default='10,30,60',
        help='Comma-separated timeouts in seconds for retries (default: 10,30,60)'
    )
    parser.add_argument(
        '--include-unknown',
        action='store_true',
        help='Include triggers not supported by this Vector (may error)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Explicit Vector IP/host to connect to (overrides auto-discovery)'
    )
    parser.add_argument(
        '--start-trigger', '--start',
        type=str,
        default=None,
        help='Trigger name to start from (skip earlier triggers). Example: --start ConnectToCubeGetIn'
    )

    args = parser.parse_args()
    
    if not args.interactive and not args.headless:
        parser.print_help()
        print("\nError: Specify --interactive or --headless")
        sys.exit(1)
    
    # Load data
    triggers = load_triggers()
    observations = load_observations()
    
    # Connect to Vector with retries and diagnostics
    print("🤖 Connecting to Vector...")

    def try_connect(retries: int, timeouts: list, host: Optional[str] = None):
        """Attempt to connect to Vector with increasing timeouts. Optionally target explicit host."""
        last_exc = None
        for attempt in range(1, retries + 1):
            to = timeouts[min(attempt - 1, len(timeouts) - 1)]
            target_info = f" (ip={host})" if host else " (auto-discovery)"
            print(f"Attempt {attempt}/{retries} - timeout={to}s{target_info}")
            try:
                if host:
                    robot = anki_vector.Robot(ip=host, enable_nav_map_feed=False, enable_face_detection=False)
                else:
                    robot = anki_vector.Robot(enable_nav_map_feed=False, enable_face_detection=False)
                robot.connect(timeout=to)
                return robot
            except Exception as e:
                last_exc = e
                msg = str(e)
                print(f"  ⚠️ Attempt {attempt} failed: {msg}")
                # Specific known failure that can be retried
                if 'ListAnimations' in msg or 'ListAnimationTriggers' in msg or 'timed out' in msg or 'Unable to establish a connection' in msg:
                    print("  ⏳ Retryable error — retrying with longer timeout...")
                    continue
                # Other exceptions are likely fatal
                break
        # All attempts failed
        return None, last_exc

    # Read retry settings from args
    req_retries = args.retries
    try:
        timeout_sequence = [int(x) for x in args.timeouts.split(',') if x.strip()]
    except Exception:
        timeout_sequence = [10, 30, 60]
    include_unknown = getattr(args, 'include_unknown', False)

    connect_result = try_connect(req_retries, timeout_sequence, host=args.host)
    if isinstance(connect_result, tuple):
        robot, last_error = connect_result
    else:
        robot = connect_result
        last_error = None

    if robot is None and args.host:
        # If explicit host was provided but connection failed, show clearer message
        print(f"❌ Failed to connect to specified host: {args.host}")

    # If connected, query device-supported animation triggers and filter list
    if robot is not None:
        try:
            available_triggers = set(robot.anim.anim_trigger_list)
        except Exception:
            try:
                available_triggers = set(robot.anim.list_animation_triggers())
            except Exception:
                available_triggers = set()

        # Normalize triggers to dict format
        normalized_triggers = []
        for t in triggers:
            if isinstance(t, dict) and 'trigger' in t:
                normalized_triggers.append(t)
            elif isinstance(t, str):
                normalized_triggers.append({'trigger': t, 'source': 'docs', 'status': 'unobserved', 'notes': ''})
            else:
                # Fallback to string representation
                normalized_triggers.append({'trigger': str(t), 'source': 'docs', 'status': 'unobserved', 'notes': ''})

        supported = [t for t in normalized_triggers if t['trigger'] in available_triggers]
        unsupported = [t for t in normalized_triggers if t['trigger'] not in available_triggers]

        print(f"\n🎯 Animations supported by device: {len(supported)}")
        if unsupported:
            print(f"⚠️ {len(unsupported)} triggers not supported by this Vector (will be skipped unless --include-unknown is used)")
            # Write unsupported list to tools/unsupported_triggers.json for review
            try:
                (ROOT / 'tools' / 'unsupported_triggers.json').write_text(json.dumps({'unsupported': [t['trigger'] for t in unsupported]}, indent=2), encoding='utf-8')
                print(f"📝 Unsupported triggers written to tools/unsupported_triggers.json")
            except Exception:
                pass

        if include_unknown:
            print("⚠️ --include-unknown specified: attempting all triggers (may produce 'Unknown animation trigger' errors on play)")
            triggers = normalized_triggers
        else:
            triggers = supported
    else:
        # Dry-run: keep original triggers (user will be entering observations manually)
        triggers = triggers

    if robot is None:
        # Provide helpful diagnostics and fallback
        print("❌ Failed to connect after retries.")
        if last_error:
            print(f"Last error: {last_error}")
        print("Suggestions:")
        print("  - Ensure Vector is powered on and connected to the same Wi-Fi network")
        print("  - Reboot Vector and retry")
        print("  - Ensure no other app is connected to Vector (mobile app may conflict)")
        print("  - Try pinging the device IP to confirm reachability: ping <VECTOR_IP>")
        print("Running in dry-run mode (no robot connection). You can still record observations manually and replay later.")
        robot = None
    else:
        print("✅ Connected!")

    
    try:
        if args.interactive:
            try:
                interactive_mode(robot, triggers, observations, args.resume, args.filter, args.start_trigger)
            except KeyboardInterrupt:
                # Save observations on Ctrl+C
                print("\n💾 Keyboard interrupt detected — saving observations...")
                save_observations(observations)
                print("✅ Observations saved. Exiting.")
        elif args.headless:
            headless_mode(robot, observations, args.count)
    finally:
        print("\n🔌 Disconnecting from Vector...")
        try:
            if robot is not None:
                robot.disconnect()
        except Exception:
            pass
    
    print("✅ Done!")


if __name__ == '__main__':
    main()
