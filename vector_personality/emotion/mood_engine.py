"""
Mood Engine Module

Implements emotional state tracking with 6 emotion drivers.
Each driver decays over time and is influenced by personality traits.

Emotion Drivers:
- Curiosity: Driven by new faces, objects, rooms (+)
- Loneliness: Decreased by face interactions (-)
- Satisfaction: Increased by successful tasks (+)
- Confusion: Increased by failed understanding (-)
- Excitement: Spikes on interesting events (+)
- Frustration: Builds up on failures (-)

Mood Calculation: Weighted average of positive vs negative drivers
Decay: Exponential decay toward baseline (50) over time
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class MoodEngine:
    """
    Emotion engine with 6 independent drivers
    
    Calculates overall mood (0-100) from weighted combination
    of emotion drivers. Each driver decays toward baseline
    over time and can be influenced by personality traits.
    """

    # Driver baselines (neutral point)
    BASELINE = 50.0

    # Decay rates (per second toward baseline)
    DEFAULT_DECAY_RATES = {
        "curiosity": 0.05,      # Decays slowly
        "loneliness": 0.02,     # Very slow decay
        "satisfaction": 0.08,   # Decays faster
        "confusion": 0.10,      # Decays quickly
        "excitement": 0.15,     # Decays very fast
        "frustration": 0.07     # Decays moderately
    }

    # Mood calculation weights (positive vs negative influence)
    DRIVER_WEIGHTS = {
        "curiosity": +0.20,     # Positive 20%
        "loneliness": -0.15,    # Negative 15%
        "satisfaction": +0.25,  # Positive 25%
        "confusion": -0.15,     # Negative 15%
        "excitement": +0.15,    # Positive 15%
        "frustration": -0.10    # Negative 10%
    }

    def __init__(self):
        """
        Initialize MoodEngine with all drivers at baseline
        """
        # Initialize all drivers at baseline (neutral)
        self.curiosity = self.BASELINE
        self.loneliness = self.BASELINE
        self.satisfaction = self.BASELINE
        self.confusion = self.BASELINE
        self.excitement = self.BASELINE
        self.frustration = self.BASELINE
        
        # Decay rates (customizable per driver)
        self.decay_rates = self.DEFAULT_DECAY_RATES.copy()
        
        # Track last update time for decay calculation
        self.last_update = datetime.now()
        
        # History tracking
        self.mood_history = []
        
        logger.info("MoodEngine initialized with 6 drivers at baseline")

    def update_driver(self, driver_name: str, delta: float) -> None:
        """
        Update emotion driver by delta amount
        
        Args:
            driver_name: Name of driver (curiosity, loneliness, etc.)
            delta: Amount to change (+/-)
        
        Raises:
            ValueError: If driver_name is invalid
        """
        if not hasattr(self, driver_name):
            raise ValueError(f"Invalid driver name: {driver_name}")
        
        # Get current value
        current = getattr(self, driver_name)
        
        # Apply delta and clamp to 0-100
        new_value = max(0.0, min(100.0, current + delta))
        
        # Set new value
        setattr(self, driver_name, new_value)
        
        logger.debug(
            f"Driver '{driver_name}': {current:.1f} → {new_value:.1f} "
            f"(delta: {delta:+.1f})"
        )

    def apply_decay(self, elapsed_seconds: float) -> None:
        """
        Apply exponential decay to all drivers toward baseline
        
        Args:
            elapsed_seconds: Time elapsed since last decay
        """
        drivers = ["curiosity", "loneliness", "satisfaction",
                   "confusion", "excitement", "frustration"]
        
        for driver in drivers:
            current = getattr(self, driver)
            decay_rate = self.decay_rates[driver]
            
            # Exponential decay toward baseline
            # Formula: value += (baseline - value) * (1 - e^(-rate * time))
            diff = self.BASELINE - current
            decay_factor = 1 - math.exp(-decay_rate * elapsed_seconds)
            new_value = current + (diff * decay_factor)
            
            setattr(self, driver, new_value)
        
        self.last_update = datetime.now()

    def calculate_mood(self, personality: Optional[object] = None) -> float:
        """
        Calculate overall mood from weighted driver values
        
        Args:
            personality: Optional PersonalityModule for trait influence
        
        Returns:
            Mood value 0-100 (0=very sad, 100=very happy)
        """
        # Start at baseline (neutral)
        mood = 50.0
        
        # Apply each driver with its weight
        for driver, weight in self.DRIVER_WEIGHTS.items():
            driver_value = getattr(self, driver)
            
            # Convert driver (0-100) to contribution (-50 to +50)
            contribution = (driver_value - 50.0) * weight
            
            # Apply personality influence if available
            if personality:
                contribution *= self._get_personality_multiplier(
                    driver, personality
                )
            
            mood += contribution
        
        # Clamp to valid range
        mood = max(0.0, min(100.0, mood))
        
        # Track history
        self.mood_history.append({
            "timestamp": datetime.now(),
            "mood": mood,
            "drivers": self.get_driver_snapshot()
        })
        
        # Keep history bounded
        if len(self.mood_history) > 100:
            self.mood_history = self.mood_history[-100:]
        
        return mood

    def _get_personality_multiplier(
        self,
        driver: str,
        personality: object
    ) -> float:
        """
        Get personality trait multiplier for driver
        
        Args:
            driver: Driver name
            personality: PersonalityModule instance
        
        Returns:
            Multiplier (0.5 to 1.5)
        """
        # Map drivers to personality traits
        trait_map = {
            "curiosity": "curiosity",
            "satisfaction": "vitality",
            "excitement": "vitality",
            "frustration": "touchiness",
            "confusion": "courage",
            "loneliness": "friendliness"
        }
        
        if driver not in trait_map:
            return 1.0
        
        trait_name = trait_map[driver]
        
        # Get effective trait value (0-1)
        traits = personality.effective_traits  # Property, not method
        trait_value = getattr(traits, trait_name, 0.5)
        
        # Convert to multiplier (0.5 to 1.5)
        # trait=0.0 → 0.5x, trait=0.5 → 1.0x, trait=1.0 → 1.5x
        multiplier = 0.5 + trait_value
        
        return multiplier

    def set_decay_rate(self, driver_name: str, rate: float) -> None:
        """
        Set custom decay rate for a driver
        
        Args:
            driver_name: Name of driver
            rate: Decay rate (0.0 = no decay, 1.0 = very fast)
        """
        if driver_name not in self.decay_rates:
            raise ValueError(f"Invalid driver name: {driver_name}")
        
        self.decay_rates[driver_name] = max(0.0, rate)
        logger.debug(f"Decay rate for '{driver_name}' set to {rate:.3f}")

    def get_driver_snapshot(self) -> Dict[str, float]:
        """
        Get current values of all drivers
        
        Returns:
            Dictionary of driver name → value
        """
        return {
            "curiosity": self.curiosity,
            "loneliness": self.loneliness,
            "satisfaction": self.satisfaction,
            "confusion": self.confusion,
            "excitement": self.excitement,
            "frustration": self.frustration
        }

    def get_dominant_emotion(self) -> str:
        """
        Get the most dominant emotion driver
        
        Returns:
            Driver name with highest deviation from baseline
        """
        drivers = self.get_driver_snapshot()
        
        # Find driver with max absolute deviation from baseline
        max_deviation = 0.0
        dominant = "neutral"
        
        for name, value in drivers.items():
            deviation = abs(value - self.BASELINE)
            if deviation > max_deviation:
                max_deviation = deviation
                dominant = name
        
        return dominant

    def reset_all_drivers(self) -> None:
        """Reset all drivers to baseline"""
        self.curiosity = self.BASELINE
        self.loneliness = self.BASELINE
        self.satisfaction = self.BASELINE
        self.confusion = self.BASELINE
        self.excitement = self.BASELINE
        self.frustration = self.BASELINE
        
        logger.info("All emotion drivers reset to baseline")

    def get_mood_trend(self, lookback: int = 10) -> float:
        """
        Calculate mood trend from recent history
        
        Args:
            lookback: Number of recent entries to analyze
        
        Returns:
            Trend value (positive = improving, negative = declining)
        """
        if len(self.mood_history) < 2:
            return 0.0
        
        recent = self.mood_history[-lookback:]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate linear trend
        first_mood = recent[0]["mood"]
        last_mood = recent[-1]["mood"]
        
        trend = (last_mood - first_mood) / len(recent)
        
        return trend

    def __repr__(self) -> str:
        """String representation"""
        mood = self.calculate_mood()
        dominant = self.get_dominant_emotion()
        
        return (
            f"MoodEngine(mood={mood:.1f}, dominant={dominant}, "
            f"curiosity={self.curiosity:.1f}, "
            f"satisfaction={self.satisfaction:.1f}, "
            f"excitement={self.excitement:.1f})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_mood_engine(config: dict = None) -> MoodEngine:
    """
    Factory function to create MoodEngine with config
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured MoodEngine instance
    """
    engine = MoodEngine()
    
    if config and "decay_rates" in config:
        for driver, rate in config["decay_rates"].items():
            engine.set_decay_rate(driver, rate)
    
    return engine


def mood_to_descriptor(mood: float) -> str:
    """
    Convert mood value to human-readable descriptor
    
    Args:
        mood: Mood value (0-100)
    
    Returns:
        Descriptor string
    """
    if mood >= 80:
        return "very happy"
    elif mood >= 65:
        return "happy"
    elif mood >= 55:
        return "content"
    elif mood >= 45:
        return "neutral"
    elif mood >= 30:
        return "sad"
    elif mood >= 15:
        return "very sad"
    else:
        return "depressed"


def simulate_mood_cycle(engine: MoodEngine, events: list) -> list:
    """
    Simulate mood changes over a series of events
    
    Args:
        engine: MoodEngine instance
        events: List of (driver, delta) tuples
    
    Returns:
        List of mood values after each event
    """
    moods = []
    
    for driver, delta in events:
        engine.update_driver(driver, delta)
        mood = engine.calculate_mood()
        moods.append(mood)
        
        # Apply decay between events
        engine.apply_decay(1.0)
    
    return moods
