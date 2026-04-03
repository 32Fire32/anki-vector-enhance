"""
Personality traits and management system.
Tracks base traits and learned adjustments.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class PersonalityTraits:
    """Six-dimensional personality trait system."""
    
    curiosity: float = 0.7      # Drive to explore and ask questions
    touchiness: float = 0.6     # Sensitivity to physical contact
    vitality: float = 0.8       # Energy level and API call budget
    friendliness: float = 0.7   # Warmth and engagement
    courage: float = 0.5        # Willingness to try new things
    sassiness: float = 0.3      # Boldness and sarcasm in responses
    
    def __post_init__(self):
        """Clamp all traits to [0.0, 1.0]."""
        for field in ['curiosity', 'touchiness', 'vitality', 'friendliness', 'courage', 'sassiness']:
            value = getattr(self, field)
            setattr(self, field, max(0.0, min(1.0, value)))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __add__(self, other: 'PersonalityTraits') -> 'PersonalityTraits':
        """Add two trait sets (for base + delta)."""
        return PersonalityTraits(
            curiosity=self.curiosity + other.curiosity,
            touchiness=self.touchiness + other.touchiness,
            vitality=self.vitality + other.vitality,
            friendliness=self.friendliness + other.friendliness,
            courage=self.courage + other.courage,
            sassiness=self.sassiness + other.sassiness,
        )


class PersonalityModule:
    """
    Manage personality traits with base + learned components.
    Base traits are configuration defaults.
    Learned component accumulates user feedback over time.
    """
    
    def __init__(self, memory_module, base_traits: Optional[PersonalityTraits] = None):
        """
        Initialize personality system.
        
        Args:
            memory_module: Database connector for persistence
            base_traits: Base trait configuration (defaults to standard)
        """
        self.memory = memory_module
        self.base_traits = base_traits or PersonalityTraits()
        self.learned_delta = PersonalityTraits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    @property
    def effective_traits(self) -> PersonalityTraits:
        """
        Return effective traits = base + learned (clamped to 0-1).
        
        Returns:
            PersonalityTraits with combined values
        """
        combined = self.base_traits + self.learned_delta
        return combined
    
    async def learn_from_interaction(self, feedback: str, magnitude: float = 0.02):
        """
        Update traits based on user feedback.
        
        Examples:
            "curious" → curiosity +0.02
            "too_talkative" → sassiness -0.02, friendliness -0.01
            "clever" → sassiness +0.03
        
        Args:
            feedback: Feedback keyword or phrase
            magnitude: Amount to adjust (typically 0.01-0.05)
        """
        feedback_lower = feedback.lower().strip()
        
        # Define feedback → trait mappings
        feedback_map = {
            'curious': {'curiosity': magnitude},
            'curious!': {'curiosity': magnitude},
            'not_curious': {'curiosity': -magnitude},
            'too_quiet': {'friendliness': magnitude, 'sassiness': magnitude},
            'too_talkative': {'sassiness': -magnitude, 'curiosity': -magnitude},
            'clever': {'sassiness': magnitude},
            'nice': {'friendliness': magnitude},
            'rude': {'friendliness': -magnitude, 'sassiness': -magnitude},
            'brave': {'courage': magnitude},
            'scared': {'courage': -magnitude},
            'energetic': {'vitality': magnitude},
            'lazy': {'vitality': -magnitude},
            'sensitive': {'touchiness': magnitude},
            'tough': {'touchiness': -magnitude},
        }
        
        if feedback_lower in feedback_map:
            adjustments = feedback_map[feedback_lower]
            for trait, delta in adjustments.items():
                current = getattr(self.learned_delta, trait)
                setattr(self.learned_delta, trait, current + delta)
            
            # Clamp learned delta to prevent runaway
            for field in ['curiosity', 'touchiness', 'vitality', 'friendliness', 'courage', 'sassiness']:
                value = getattr(self.learned_delta, field)
                setattr(self.learned_delta, field, max(-1.0, min(1.0, value)))
            
            await self.save_to_database()
    
    async def save_to_database(self):
        """
        Persist learned traits to SQL Server.
        Assumes memory_module.execute() is available.
        """
        if self.memory is None:
            return
        
        delta_json = json.dumps(self.learned_delta.to_dict())
        # Database persistence disabled - personality traits are session-only
        # await self.memory.execute(
        #     """
        #     INSERT INTO personality_traits (date, learned_delta)
        #     VALUES (GETDATE(), ?)
        #     """,
        #     {'learned_delta': delta_json}
        # )
    
    async def reset_to_base(self):
        """Reset learned component; return to base traits."""
        self.learned_delta = PersonalityTraits(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Persistence disabled - personality resets are session-only
    
    def __repr__(self) -> str:
        """String representation of effective traits."""
        traits = self.effective_traits
        return (
            f"PersonalityTraits("
            f"curiosity={traits.curiosity:.2f}, "
            f"touchiness={traits.touchiness:.2f}, "
            f"vitality={traits.vitality:.2f}, "
            f"friendliness={traits.friendliness:.2f}, "
            f"courage={traits.courage:.2f}, "
            f"sassiness={traits.sassiness:.2f})"
        )
