"""
Eye Color Mapper Module

Maps emotional states to RGB colors for Vector's LED eyes.
Implements smooth color transitions and intensity scaling.

Color Mappings:
- Happy (mood >70): Green
- Sad (mood <30): Blue
- Neutral (mood 40-60): Yellow/Orange
- Curious: Purple
- Frustrated: Red
- Confused: Cyan
- Excited: Bright Yellow

Features:
- Linear interpolation for smooth transitions
- Intensity scaling based on emotion strength
- Direct Vector SDK integration
"""

import logging
from typing import Tuple, Optional
import asyncio

logger = logging.getLogger(__name__)


class EyeColorMapper:
    """
    Maps emotions to RGB colors for Vector's eyes
    
    Provides smooth color transitions and intensity scaling
    for visual feedback of emotional state.
    """

    # Emotion color definitions (R, G, B) in 0-255 range
    EMOTION_COLORS = {
        "happy": (0, 255, 0),          # Green
        "sad": (0, 100, 255),          # Blue
        "neutral": (255, 200, 0),      # Yellow/Orange
        "curious": (180, 0, 255),      # Purple
        "frustrated": (255, 0, 0),     # Red
        "confused": (0, 255, 255),     # Cyan
        "excited": (255, 255, 0),      # Bright Yellow
        "bored": (100, 100, 100),      # Gray
        "scared": (128, 0, 128),       # Dark Purple
        "content": (0, 200, 100)       # Teal/Green
    }

    def __init__(self):
        """Initialize EyeColorMapper with color mappings"""
        self.emotion_colors = self.EMOTION_COLORS.copy()
        self.current_color = (255, 200, 0)  # Start neutral
        self.transition_speed = 0.5  # seconds for full transition
        
        logger.info(f"EyeColorMapper initialized with {len(self.emotion_colors)} colors")

    def mood_to_rgb(self, mood: float) -> Tuple[int, int, int]:
        """
        Convert mood value to RGB color
        
        Mood ranges:
        - 0-20: Very sad (dark blue)
        - 20-40: Sad (blue)
        - 40-60: Neutral (yellow/orange)
        - 60-80: Happy (green)
        - 80-100: Very happy (bright green)
        
        Args:
            mood: Mood value (0-100)
        
        Returns:
            RGB tuple (R, G, B) in 0-255 range
        """
        # Clamp mood to valid range
        mood = max(0.0, min(100.0, mood))
        
        if mood >= 80:
            # Very happy: Bright green
            intensity = (mood - 80) / 20
            return self._scale_color((0, 255, 0), 0.8 + intensity * 0.2)
        
        elif mood >= 60:
            # Happy: Transition yellow → green
            progress = (mood - 60) / 20
            return self.interpolate_color(
                self.emotion_colors["neutral"],  # Yellow
                self.emotion_colors["happy"],    # Green
                progress
            )
        
        elif mood >= 40:
            # Neutral: Yellow/Orange
            return self.emotion_colors["neutral"]
        
        elif mood >= 20:
            # Sad: Transition yellow → blue
            progress = (mood - 20) / 20
            return self.interpolate_color(
                self.emotion_colors["sad"],       # Blue
                self.emotion_colors["neutral"],   # Yellow
                progress
            )
        
        else:
            # Very sad: Dark blue
            intensity = mood / 20
            return self._scale_color((0, 100, 255), 0.5 + intensity * 0.5)

    def emotion_to_rgb(
        self,
        emotion: str,
        intensity: float = 1.0
    ) -> Tuple[int, int, int]:
        """
        Convert emotion name to RGB color
        
        Args:
            emotion: Emotion name (happy, sad, curious, etc.)
            intensity: Color intensity (0.0-1.0)
        
        Returns:
            RGB tuple scaled by intensity
        
        Raises:
            ValueError: If emotion name is invalid
        """
        if emotion not in self.emotion_colors:
            logger.warning(f"Unknown emotion '{emotion}', using neutral")
            emotion = "neutral"
        
        base_color = self.emotion_colors[emotion]
        return self._scale_color(base_color, intensity)

    def interpolate_color(
        self,
        start_color: Tuple[int, int, int],
        end_color: Tuple[int, int, int],
        progress: float
    ) -> Tuple[int, int, int]:
        """
        Linear interpolation between two colors
        
        Args:
            start_color: Starting RGB color
            end_color: Ending RGB color
            progress: Transition progress (0.0-1.0)
        
        Returns:
            Interpolated RGB color
        """
        # Clamp progress
        progress = max(0.0, min(1.0, progress))
        
        # Interpolate each channel
        r = int(start_color[0] + (end_color[0] - start_color[0]) * progress)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * progress)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * progress)
        
        return (r, g, b)

    def _scale_color(
        self,
        color: Tuple[int, int, int],
        intensity: float
    ) -> Tuple[int, int, int]:
        """
        Scale color by intensity factor
        
        Args:
            color: Base RGB color
            intensity: Scaling factor (0.0-1.0)
        
        Returns:
            Scaled RGB color
        """
        intensity = max(0.0, min(1.0, intensity))
        
        r = int(color[0] * intensity)
        g = int(color[1] * intensity)
        b = int(color[2] * intensity)
        
        return (r, g, b)

    def mood_to_emotion_name(self, mood: float) -> str:
        """
        Map mood value to emotion name
        
        Args:
            mood: Mood value (0-100)
        
        Returns:
            Emotion name string
        """
        if mood >= 70:
            return "happy"
        elif mood >= 55:
            return "content"
        elif mood >= 45:
            return "neutral"
        elif mood >= 30:
            return "sad"
        else:
            return "sad"

    async def set_vector_eyes(
        self,
        robot,
        rgb: Tuple[int, int, int],
        transition_time: float = 0.5
    ) -> None:
        """
        Set Vector's eye color via SDK
        
        Args:
            robot: anki_vector.Robot instance
            rgb: RGB color tuple
            transition_time: Transition duration in seconds
        """
        try:
            # Unpack RGB
            r, g, b = rgb
            
            # Set eye color using Vector SDK
            # Note: set_eye_color returns a response object, not a coroutine
            robot.behavior.set_eye_color(
                hue=self._rgb_to_hue(rgb),
                saturation=1.0
            )
            
            # Update current color
            self.current_color = rgb
            
            logger.debug(f"Set Vector eyes to RGB({r}, {g}, {b})")
            
        except Exception as e:
            logger.error(f"Failed to set Vector eye color: {e}")

    def _rgb_to_hue(self, rgb: Tuple[int, int, int]) -> float:
        """
        Convert RGB to HSV hue for Vector SDK
        
        Args:
            rgb: RGB tuple (0-255)
        
        Returns:
            Hue value (0.0-1.0)
        """
        # Normalize to 0-1
        r = rgb[0] / 255.0
        g = rgb[1] / 255.0
        b = rgb[2] / 255.0
        
        # Find min/max
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        if delta == 0:
            return 0.0
        
        # Calculate hue
        if max_c == r:
            hue = ((g - b) / delta) % 6
        elif max_c == g:
            hue = ((b - r) / delta) + 2
        else:
            hue = ((r - g) / delta) + 4
        
        hue = hue / 6.0  # Normalize to 0-1
        
        return hue

    async def animate_transition(
        self,
        robot,
        start_rgb: Tuple[int, int, int],
        end_rgb: Tuple[int, int, int],
        duration: float = 1.0,
        steps: int = 10
    ) -> None:
        """
        Animate smooth color transition
        
        Args:
            robot: anki_vector.Robot instance
            start_rgb: Starting RGB color
            end_rgb: Ending RGB color
            duration: Total animation duration in seconds
            steps: Number of interpolation steps
        """
        step_duration = duration / steps
        
        for i in range(steps + 1):
            progress = i / steps
            color = self.interpolate_color(start_rgb, end_rgb, progress)
            
            await self.set_vector_eyes(robot, color, transition_time=0)
            await asyncio.sleep(step_duration)

    def get_mood_gradient(
        self,
        start_mood: float,
        end_mood: float,
        steps: int = 10
    ) -> list:
        """
        Generate color gradient between two mood values
        
        Args:
            start_mood: Starting mood (0-100)
            end_mood: Ending mood (0-100)
            steps: Number of gradient steps
        
        Returns:
            List of RGB colors
        """
        gradient = []
        
        for i in range(steps + 1):
            progress = i / steps
            mood = start_mood + (end_mood - start_mood) * progress
            rgb = self.mood_to_rgb(mood)
            gradient.append(rgb)
        
        return gradient

    def add_custom_emotion(
        self,
        emotion_name: str,
        rgb: Tuple[int, int, int]
    ) -> None:
        """
        Add custom emotion color mapping
        
        Args:
            emotion_name: Name for the emotion
            rgb: RGB color tuple
        """
        self.emotion_colors[emotion_name] = rgb
        logger.info(f"Added custom emotion '{emotion_name}': RGB{rgb}")

    def __repr__(self) -> str:
        """String representation"""
        r, g, b = self.current_color
        return f"EyeColorMapper(current=RGB({r}, {g}, {b}))"


# ============================================================================
# Utility Functions
# ============================================================================

def create_eye_color_mapper(config: dict = None) -> EyeColorMapper:
    """
    Factory function to create EyeColorMapper with config
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured EyeColorMapper instance
    """
    mapper = EyeColorMapper()
    
    if config and "custom_colors" in config:
        for emotion, rgb in config["custom_colors"].items():
            mapper.add_custom_emotion(emotion, rgb)
    
    if config and "transition_speed" in config:
        mapper.transition_speed = config["transition_speed"]
    
    return mapper


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string
    
    Args:
        rgb: RGB tuple (0-255)
    
    Returns:
        Hex color string (e.g., "#FF0000")
    """
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
    
    Returns:
        RGB tuple (0-255)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


async def demo_color_cycle(robot, mapper: EyeColorMapper) -> None:
    """
    Demonstrate color cycling through emotions
    
    Args:
        robot: anki_vector.Robot instance
        mapper: EyeColorMapper instance
    """
    emotions = ["happy", "excited", "curious", "neutral", 
                "confused", "sad", "frustrated"]
    
    for emotion in emotions:
        rgb = mapper.emotion_to_rgb(emotion, intensity=0.8)
        await mapper.set_vector_eyes(robot, rgb)
        logger.info(f"Showing {emotion}: RGB{rgb}")
        await asyncio.sleep(2.0)
