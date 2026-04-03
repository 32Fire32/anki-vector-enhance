#!/usr/bin/env python3
"""
Extract animation triggers from Vector documentation.

Scans vector_docs/ for animation trigger names and generates a JSON file
for use in animation testing and mapping workflows.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import sys


class AnimationTriggerExtractor:
    """Extracts animation trigger names from Vector documentation."""
    
    # Pattern for animation trigger lines in documentation
    # Matches lines that look like animation names (CamelCase or with underscores)
    TRIGGER_PATTERNS = [
        # Direct trigger names (from animation trigger table)
        re.compile(r'^([A-Z][A-Za-z0-9_]+)$', re.MULTILINE),
        # Animation file names (anim_xxx format)
        re.compile(r'anim_([a-z0-9_]+)', re.IGNORECASE),
        # Animation group references
        re.compile(r'ag_([a-z0-9_]+)', re.IGNORECASE),
    ]
    
    # Known sections containing animation triggers
    ANIMATION_SECTIONS = [
        '13.1 Animation Triggers',
        'Animation Triggers',
        'Trigger Name',
    ]
    
    def __init__(self, docs_dir: Path):
        """Initialize extractor with documentation directory."""
        self.docs_dir = docs_dir
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documentation directory not found: {docs_dir}")
    
    def extract_from_file(self, file_path: Path) -> Set[str]:
        """Extract animation triggers from a single file."""
        triggers = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return triggers
        
        # Check if file contains animation trigger section
        has_animation_section = any(
            section in content for section in self.ANIMATION_SECTIONS
        )
        
        if not has_animation_section:
            return triggers
        
        # Extract triggers from animation section
        # Look for the table format in Vector-Wiki.txt
        in_trigger_section = False
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Detect start of animation trigger section
            if '13.1 Animation Triggers' in line or 'Trigger Name' in line:
                in_trigger_section = True
                continue
            
            # Detect end of section (next numbered section)
            if in_trigger_section and re.match(r'^\d+\.\d+\s+[A-Z]', line):
                if 'Animation' not in line:
                    in_trigger_section = False
                    continue
            
            # Extract trigger names from table
            if in_trigger_section:
                # Skip table headers and separators
                if any(x in line for x in ['Description', '====', '---', 'Copyright']):
                    continue
                
                # Extract trigger name (clean line, no leading/trailing whitespace)
                trigger = line.strip()
                
                # Validate trigger format
                if self._is_valid_trigger(trigger):
                    triggers.add(trigger)
        
        return triggers
    
    def _is_valid_trigger(self, trigger: str) -> bool:
        """Check if a string is a valid animation trigger name."""
        if not trigger or len(trigger) < 3:
            return False
        
        # Must start with uppercase letter or 'anim_'
        if not (trigger[0].isupper() or trigger.startswith('anim_')):
            return False
        
        # Must contain only alphanumeric and underscores
        if not re.match(r'^[A-Za-z0-9_]+$', trigger):
            return False
        
        # Filter out common false positives
        false_positives = {
            'Summary', 'Table', 'Figure', 'Note', 'Warning', 'See', 'This',
            'For', 'The', 'Vector', 'Cozmo', 'Animation', 'Trigger', 'Name',
            'Description', 'Copyright', 'Project', 'Victor', 'Match'
        }
        
        if trigger in false_positives:
            return False
        
        return True
    
    def extract_all(self) -> List[Dict[str, str]]:
        """
        Extract all animation triggers from documentation.
        
        Returns:
            List of trigger dictionaries with 'trigger' and 'source' keys
        """
        all_triggers = {}  # Use dict to track source
        
        # Scan all text files in docs directory
        for file_path in self.docs_dir.glob('*.txt'):
            print(f"Scanning: {file_path.name}", file=sys.stderr)
            triggers = self.extract_from_file(file_path)
            
            for trigger in triggers:
                if trigger not in all_triggers:
                    all_triggers[trigger] = file_path.name
            
            print(f"  Found {len(triggers)} triggers", file=sys.stderr)
        
        # Convert to list format for JSON output
        result = [
            {
                'trigger': trigger,
                'source': source,
                'status': 'unobserved',
                'notes': ''
            }
            for trigger, source in sorted(all_triggers.items())
        ]
        
        return result
    
    def generate_json(self, output_path: Path) -> int:
        """
        Generate JSON file with extracted triggers.
        
        Returns:
            Number of triggers extracted
        """
        triggers = self.extract_all()
        
        output = {
            'schema_version': '1.0',
            'metadata': {
                'generated_date': datetime.now().isoformat(),
                'source_directory': str(self.docs_dir),
                'extraction_method': 'documentation_scan',
                'total_triggers': len(triggers)
            },
            'triggers': triggers,
            'usage_notes': [
                'This file contains all animation triggers found in Vector documentation',
                'Use tools/run_animation_tests.py to observe and name these animations',
                'Status values: unobserved, observed, tested, validated',
                'Add your observations to animation_observations.json'
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return len(triggers)


def main():
    """Command-line interface for extraction tool."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract animation triggers from Vector documentation'
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('vector_docs'),
        help='Path to vector_docs directory (default: vector_docs)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('tools/animation_triggers.json'),
        help='Output JSON file path (default: tools/animation_triggers.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed extraction progress'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.docs_dir.exists():
        print(f"Error: Documentation directory not found: {args.docs_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract triggers
    try:
        extractor = AnimationTriggerExtractor(args.docs_dir)
        count = extractor.generate_json(args.output)
        
        print(f"\n✅ Successfully extracted {count} animation triggers")
        print(f"📄 Output: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review {args.output} and remove any false positives")
        print(f"  2. Run: python tools/run_animation_tests.py --interactive")
        print(f"  3. Observe animations and provide names in Italian")
        print(f"  4. Build emotion mappings from your observations")
        
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
