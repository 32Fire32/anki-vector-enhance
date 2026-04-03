"""
Idle Controller Module

Provides idle behaviors for Vector when no tasks are pending.
Keeps Vector looking "alive" with animations, head movements, and subtle motions.

Key Features:
- Random head movements (look around)
- Idle animations from SDK animation list
- Subtle drive movements (shift weight, small turns)
- Prevents "frozen" appearance
- Respects behavior control levels

Performance Target: Execute idle behavior every 5-10 seconds
"""

import logging
import asyncio
import random
from typing import Optional, Any
from datetime import datetime, timedelta
from anki_vector.util import degrees

logger = logging.getLogger(__name__)


class IdleController:
    """
    Controls Vector's idle behavior to keep him looking alive
    
    When no tasks are pending, executes random idle behaviors:
    - Head movements (look around)
    - Idle animations (blink, shift eyes, wiggle)
    - Subtle body movements (shift weight)
    """

    def __init__(
        self,
        robot: Any,
        min_interval_seconds: float = 5.0,
        max_interval_seconds: float = 10.0
    ):
        """
        Initialize IdleController
        
        Args:
            robot: Vector robot instance
            min_interval_seconds: Minimum time between idle behaviors
            max_interval_seconds: Maximum time between idle behaviors
        """
        self.robot = robot
        self.min_interval = min_interval_seconds
        self.max_interval = max_interval_seconds
        
        self.last_idle_time = datetime.now()
        self.idle_count = 0
        
        # Idle behavior weights (higher = more frequent)
        self.behavior_weights = {
            "head_movement": 25,
            "animation": 20,
            "look_around": 15,
            "eye_shift": 10,
            "small_drive": 20,  # Short forward/backward drives
            "turn_in_place": 10  # Rotate to look around
        }
        
        logger.info(f"IdleController initialized: interval={min_interval_seconds}-{max_interval_seconds}s")
    
    def should_execute_idle(self) -> bool:
        """
        Check if enough time has passed to execute idle behavior
        
        Returns:
            True if idle behavior should execute
        """
        elapsed = (datetime.now() - self.last_idle_time).total_seconds()
        # Random interval between min and max
        target_interval = random.uniform(self.min_interval, self.max_interval)
        return elapsed >= target_interval
    
    async def execute_idle_behavior(self):
        """
        Execute a random idle behavior
        
        Randomly selects and executes one of:
        - Head movement (tilt up/down, turn left/right)
        - Idle animation (from SDK animation list)
        - Look around (scan environment)
        - Eye shift (change eye position without head movement)
        - Small drive (forward/backward movement)
        - Turn in place (rotation)
        """
        if not self.should_execute_idle():
            return
        
        try:
            # Select behavior based on weights
            behavior = random.choices(
                list(self.behavior_weights.keys()),
                weights=list(self.behavior_weights.values()),
                k=1
            )[0]
            
            logger.info(f"🎬 Idle behavior: {behavior}")
            
            if behavior == "head_movement":
                await self._random_head_movement()
            elif behavior == "animation":
                await self._play_idle_animation()
            elif behavior == "look_around":
                await self._look_around()
            elif behavior == "eye_shift":
                await self._eye_shift()
            elif behavior == "small_drive":
                await self._small_drive()
            elif behavior == "turn_in_place":
                await self._turn_in_place()
            
            self.last_idle_time = datetime.now()
            self.idle_count += 1
            
        except Exception as e:
            logger.error(f"Error executing idle behavior: {e}", exc_info=True)
    
    async def _random_head_movement(self):
        """Move head to random angle"""
        try:
            # Random head angle between -25 and 44.5 degrees (Vector's physical limits)
            angle = random.uniform(-20, 40)
            
            logger.debug(f"Moving head to {angle:.1f}°")
            self.robot.behavior.set_head_angle(degrees(angle))
            
            # Wait a moment
            await asyncio.sleep(random.uniform(0.5, 1.5))
        
        except Exception as e:
            logger.warning(f"Head movement failed: {e}")
    
    async def _play_idle_animation(self):
        """Play a random idle animation"""
        try:
            # Idle animation triggers (subtle, non-disruptive)
            idle_animations = [
                "anim_eyepose_happy",
                "anim_eyepose_sad",
                "anim_eyepose_surprised",
                "anim_eyepose_worried",
                "anim_eyepose_suspicious",
                "anim_eyepose_curious",
                "anim_eyepose_sleepy",
                "anim_eyepose_grumpy"
            ]
            
            animation = random.choice(idle_animations)
            
            logger.debug(f"Playing animation: {animation}")
            
            # Note: play_animation_trigger is non-blocking
            self.robot.anim.play_animation_trigger(animation)
            
            # Small delay
            await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.warning(f"Animation playback failed: {e}")
    
    async def _look_around(self):
        """Scan environment with head movements"""
        try:
            logger.debug("Looking around")
            
            # Sequence: center → left → center → right → center
            angles = [0, -20, 0, 20, 0]
            
            for angle in angles:
                self.robot.behavior.set_head_angle(degrees(angle))
                await asyncio.sleep(0.8)
        
        except Exception as e:
            logger.warning(f"Look around failed: {e}")
    
    async def _eye_shift(self):
        """Shift eyes without moving head"""
        try:
            logger.debug("Shifting eyes")
            
            # Use eye pose animation for subtle eye movement
            eye_animations = [
                "anim_eyepose_happy",
                "anim_eyepose_curious",
                "anim_eyepose_suspicious"
            ]
            
            animation = random.choice(eye_animations)
            self.robot.anim.play_animation_trigger(animation)
            
            await asyncio.sleep(1.0)
        
        except Exception as e:
            logger.warning(f"Eye shift failed: {e}")
    
    async def _small_drive(self):
        """Short forward or backward drive movement"""
        try:
            from anki_vector.util import distance_mm, speed_mmps
            
            # Random short distance (50-150mm forward or backward)
            distance = random.uniform(50, 150) * random.choice([1, -1])
            speed = 80  # Moderate speed
            
            direction = "forward" if distance > 0 else "backward"
            logger.info(f"🚗 Driving {direction}: {abs(distance):.0f}mm at {speed}mm/s")
            
            self.robot.behavior.drive_straight(
                distance_mm(distance),
                speed_mmps(speed),
                should_play_anim=True
            )
            
            await asyncio.sleep(2.0)  # Wait for drive to complete
            logger.info(f"✅ Drive complete")
        
        except Exception as e:
            logger.error(f"❌ Small drive failed: {e}", exc_info=True)
    
    async def _turn_in_place(self):
        """Rotate in place to scan environment"""
        try:
            from anki_vector.util import degrees
            
            # Random turn angle (-90 to +90 degrees)
            angle = random.uniform(-90, 90)
            
            direction = "left" if angle > 0 else "right"
            logger.info(f"🔄 Turning {direction}: {abs(angle):.0f}°")
            
            self.robot.behavior.turn_in_place(degrees(angle))
            
            await asyncio.sleep(2.0)  # Wait for turn to complete
            logger.info(f"✅ Turn complete")
        
        except Exception as e:
            logger.error(f"❌ Turn in place failed: {e}", exc_info=True)
    
    def reset(self):
        """Reset idle timer (call when executing non-idle tasks)"""
        self.last_idle_time = datetime.now()
        logger.debug("Idle timer reset")
    
    def get_statistics(self) -> dict:
        """
        Get idle behavior statistics
        
        Returns:
            Dict with statistics
        """
        elapsed = (datetime.now() - self.last_idle_time).total_seconds()
        
        return {
            "idle_count": self.idle_count,
            "last_idle_seconds_ago": elapsed,
            "average_interval": (self.min_interval + self.max_interval) / 2
        }
    
    def set_interval(self, min_seconds: float, max_seconds: float):
        """
        Update idle interval range
        
        Args:
            min_seconds: Minimum time between idle behaviors
            max_seconds: Maximum time between idle behaviors
        """
        if min_seconds <= 0 or max_seconds <= min_seconds:
            logger.warning(f"Invalid interval: {min_seconds}-{max_seconds}")
            return
        
        self.min_interval = min_seconds
        self.max_interval = max_seconds
        logger.info(f"Idle interval changed to {min_seconds}-{max_seconds}s")
    
    def set_behavior_weight(self, behavior: str, weight: int):
        """
        Update behavior selection weight
        
        Args:
            behavior: Behavior name (head_movement, animation, look_around, eye_shift)
            weight: Weight (0-100, higher = more frequent)
        """
        if behavior not in self.behavior_weights:
            logger.warning(f"Unknown behavior: {behavior}")
            return
        
        if weight < 0 or weight > 100:
            logger.warning(f"Invalid weight: {weight}")
            return
        
        self.behavior_weights[behavior] = weight
        logger.info(f"Behavior weight updated: {behavior} = {weight}")


# ========== Convenience Functions ==========

def create_idle_controller(
    robot: Any,
    min_interval_seconds: float = 5.0,
    max_interval_seconds: float = 10.0
) -> IdleController:
    """
    Factory function to create IdleController instance
    
    Args:
        robot: Vector robot instance
        min_interval_seconds: Minimum time between idle behaviors
        max_interval_seconds: Maximum time between idle behaviors
    
    Returns:
        IdleController instance
    
    Usage:
        idle = create_idle_controller(robot)
        
        # In main loop:
        if not task_manager.has_pending_tasks():
            await idle.execute_idle_behavior()
    """
    return IdleController(
        robot=robot,
        min_interval_seconds=min_interval_seconds,
        max_interval_seconds=max_interval_seconds
    )
