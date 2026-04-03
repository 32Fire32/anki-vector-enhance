"""
Room Inference Module

Infers room type from detected objects using pattern matching.
Integrates with database to track room visits and transitions.

Key Features:
- Pattern-based room classification (kitchen, bedroom, living room, etc.)
- Confidence scoring for room inference
- GetOrCreateRoom database integration
- Room transition detection

Accuracy Target: 80% on test scenes
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from collections import Counter
import asyncio

logger = logging.getLogger(__name__)


class RoomInference:
    """
    Room type inference from object detections
    
    Uses pattern matching on detected objects to classify room type.
    Tracks room transitions and integrates with database storage.
    """

    # Room patterns: {room_type: [characteristic_objects]}
    ROOM_PATTERNS = {
        "kitchen": {
            "strong": ["refrigerator", "oven", "microwave", "sink", "toaster", "dishwasher"],
            "weak": ["dining table", "chair", "bowl", "cup", "bottle", "knife", "fork", "spoon"]
        },
        "bedroom": {
            "strong": ["bed", "pillow"],
            "weak": ["clock", "book", "teddy bear", "suitcase", "backpack"]
        },
        "living_room": {
            "strong": ["couch", "tv", "remote"],
            "weak": ["chair", "potted plant", "vase", "book", "clock"]
        },
        "bathroom": {
            "strong": ["toilet", "sink"],
            "weak": ["toothbrush", "hair drier", "bottle"]
        },
        "dining_room": {
            "strong": ["dining table"],
            "weak": ["chair", "bowl", "cup", "wine glass", "fork", "knife", "spoon", "vase"]
        },
        "office": {
            "strong": ["laptop", "keyboard", "mouse", "desk"],
            "weak": ["chair", "book", "clock", "cell phone"]
        },
        "garage": {
            "strong": ["car", "bicycle", "motorcycle"],
            "weak": ["skateboard", "sports ball"]
        },
        "outdoor": {
            "strong": ["car", "bicycle", "traffic light", "stop sign", "bench"],
            "weak": ["person", "bird", "dog", "cat", "tree"]
        }
    }

    def __init__(self, connector):
        """
        Initialize RoomInference
        
        Args:
            connector: SQLServerConnector instance for database operations
        """
        self.connector = connector
        self.room_patterns = self.ROOM_PATTERNS
        
        # State tracking
        self.last_room_id: Optional[UUID] = None
        self.last_room_type: Optional[str] = None
        self.detection_history: List[Dict[str, Any]] = []
        
        logger.info(f"RoomInference initialized with {len(self.room_patterns)} room patterns")

    def infer_room_type(self, detections: List[Dict[str, Any]]) -> str:
        """
        Infer room type from object detections
        
        Args:
            detections: List of detections from ObjectDetector
                Each detection should have "class" and "confidence" keys
        
        Returns:
            Room type string or "unknown" if no match
        """
        if not detections:
            return "unknown"
        
        # Calculate scores for each room type
        scores = self.calculate_pattern_scores(detections)
        
        # Get room with highest score
        if scores:
            best_room = max(scores.items(), key=lambda x: x[1])
            room_type, score = best_room
            
            # Require minimum confidence
            if score >= 0.3:
                logger.info(f"Inferred room type: {room_type} (score: {score:.2f})")
                return room_type
        
        logger.debug("No confident room match, returning 'unknown'")
        return "unknown"

    def calculate_pattern_scores(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for each room pattern
        
        Args:
            detections: List of detections
        
        Returns:
            Dictionary mapping room_type to confidence score
        """
        # Extract detected classes
        detected_classes = [d["class"] for d in detections]
        detected_confidences = {d["class"]: d["confidence"] for d in detections}
        
        scores = {}
        
        for room_type, patterns in self.room_patterns.items():
            strong_objects = patterns["strong"]
            weak_objects = patterns["weak"]
            
            # Calculate strong match score
            strong_score = 0.0
            for obj in strong_objects:
                if obj in detected_classes:
                    confidence = detected_confidences[obj]
                    strong_score += confidence * 2.0  # Strong objects worth 2x
            
            # Calculate weak match score
            weak_score = 0.0
            for obj in weak_objects:
                if obj in detected_classes:
                    confidence = detected_confidences[obj]
                    weak_score += confidence * 0.5  # Weak objects worth 0.5x
            
            # Total score (normalized by max possible)
            max_strong = len(strong_objects) * 2.0
            max_weak = len(weak_objects) * 0.5
            max_possible = max_strong + max_weak
            
            if max_possible > 0:
                total_score = (strong_score + weak_score) / max_possible
                scores[room_type] = total_score
        
        return scores

    async def get_or_create_room(self, room_type: str) -> UUID:
        """
        Get existing room or create new one in database
        
        Args:
            room_type: Room type string
        
        Returns:
            Room UUID
        """
        # Use stored procedure to get or create room
        room_id = await self.connector.get_or_create_room(room_type)
        
        logger.debug(f"Room ID for '{room_type}': {room_id}")
        return room_id

    async def process_objects(
        self,
        detections: List[Dict[str, Any]]
    ) -> Optional[UUID]:
        """
        Process object detections and update room state
        
        Full pipeline:
        1. Infer room type from objects
        2. Get or create room in database
        3. Detect room transitions
        4. Update state tracking
        
        Args:
            detections: List of detections from ObjectDetector
        
        Returns:
            Room UUID or None if inference failed
        """
        # Infer room type
        room_type = self.infer_room_type(detections)
        
        if room_type == "unknown":
            logger.debug("Cannot process unknown room type")
            return None
        
        # Get or create room in database
        room_id = await self.get_or_create_room(room_type)
        
        # Check for room transition
        if self.last_room_id and self.last_room_id != room_id:
            logger.info(
                f"Room transition detected: {self.last_room_type} → {room_type}"
            )
        
        # Update state
        self.last_room_id = room_id
        self.last_room_type = room_type
        
        # Add to history
        self.detection_history.append({
            "room_id": room_id,
            "room_type": room_type,
            "detection_count": len(detections),
            "objects": [d["class"] for d in detections]
        })
        
        # Keep history bounded
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
        
        return room_id

    def get_room_transition_count(self) -> int:
        """
        Count room transitions in history
        
        Returns:
            Number of room transitions
        """
        if len(self.detection_history) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(self.detection_history)):
            prev_room = self.detection_history[i - 1]["room_id"]
            curr_room = self.detection_history[i]["room_id"]
            if prev_room != curr_room:
                transitions += 1
        
        return transitions

    def get_most_common_rooms(self, top_k: int = 3) -> List[tuple]:
        """
        Get most frequently visited room types
        
        Args:
            top_k: Number of rooms to return
        
        Returns:
            List of (room_type, count) tuples
        """
        if not self.detection_history:
            return []
        
        room_types = [entry["room_type"] for entry in self.detection_history]
        counter = Counter(room_types)
        
        return counter.most_common(top_k)

    def get_current_room_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about current room
        
        Returns:
            Dictionary with room info or None
        """
        if not self.last_room_id:
            return None
        
        return {
            "room_id": self.last_room_id,
            "room_type": self.last_room_type,
            "visit_duration": len([
                e for e in self.detection_history
                if e["room_id"] == self.last_room_id
            ])
        }

    def reset_state(self) -> None:
        """Reset room tracking state"""
        self.last_room_id = None
        self.last_room_type = None
        self.detection_history = []
        logger.info("Room inference state reset")

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"RoomInference("
            f"current_room={self.last_room_type}, "
            f"history_size={len(self.detection_history)}, "
            f"transitions={self.get_room_transition_count()})"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_room_inference(connector, config: dict = None) -> RoomInference:
    """
    Factory function to create RoomInference with config
    
    Args:
        connector: SQLServerConnector instance
        config: Optional configuration dictionary
    
    Returns:
        Configured RoomInference instance
    """
    inference = RoomInference(connector)
    
    # Apply custom patterns if provided
    if config and "custom_patterns" in config:
        inference.room_patterns.update(config["custom_patterns"])
        logger.info(f"Added {len(config['custom_patterns'])} custom room patterns")
    
    return inference


def room_type_to_emoji(room_type: str) -> str:
    """
    Map room type to emoji for display
    
    Args:
        room_type: Room type string
    
    Returns:
        Emoji string
    """
    emoji_map = {
        "kitchen": "🍳",
        "bedroom": "🛏️",
        "living_room": "🛋️",
        "bathroom": "🚿",
        "dining_room": "🍽️",
        "office": "💼",
        "garage": "🚗",
        "outdoor": "🌳",
        "unknown": "❓"
    }
    return emoji_map.get(room_type, "❓")


def describe_room_confidence(scores: Dict[str, float]) -> str:
    """
    Create human-readable description of room confidence
    
    Args:
        scores: Room type confidence scores
    
    Returns:
        Description string
    """
    if not scores:
        return "No room patterns matched"
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = sorted_scores[:3]
    descriptions = [
        f"{room_type}: {score*100:.0f}%"
        for room_type, score in top_3
        if score > 0.1
    ]
    
    if descriptions:
        return "Confidence: " + ", ".join(descriptions)
    else:
        return "Low confidence for all room types"


async def detect_and_infer_room(
    object_detector,
    room_inference,
    image: Any
) -> tuple:
    """
    Detect objects and infer room type in one call
    
    Args:
        object_detector: ObjectDetector instance
        room_inference: RoomInference instance
        image: Image as numpy array
    
    Returns:
        Tuple of (detections, room_id, room_type)
    """
    # Detect objects
    detections = object_detector.detect(image)
    
    # Infer room and update database
    room_id = await room_inference.process_objects(detections)
    room_type = room_inference.last_room_type
    
    return detections, room_id, room_type


def validate_room_inference_accuracy(
    test_cases: List[tuple],
    room_inference: RoomInference
) -> float:
    """
    Validate room inference accuracy on test cases
    
    Args:
        test_cases: List of (detections, expected_room_type) tuples
        room_inference: RoomInference instance
    
    Returns:
        Accuracy as float (0.0-1.0)
    """
    if not test_cases:
        return 0.0
    
    correct = 0
    for detections, expected in test_cases:
        inferred = room_inference.infer_room_type(detections)
        if inferred == expected:
            correct += 1
    
    accuracy = correct / len(test_cases)
    logger.info(f"Validation accuracy: {accuracy*100:.1f}% ({correct}/{len(test_cases)})")
    
    return accuracy
