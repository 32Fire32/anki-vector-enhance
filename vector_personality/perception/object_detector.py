"""
Object Detection Module

Integrates YOLOv5 for real-time object detection from Vector's camera.
Ports functionality from vector_photo_analyzer.py with Vector SDK integration.

Key Features:
- YOLOv5 inference on Vector camera frames
- Confidence filtering (>0.5 threshold)
- COCO class mapping to readable names
- Batch processing support

Performance Target: >5 FPS on CPU
Model: yolov5n.pt (nano) for speed, yolov5s.pt (small) for accuracy
"""

import logging
import warnings
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

# Suppress FutureWarning from YOLOv5 library about deprecated torch.cuda.amp.autocast
warnings.filterwarnings('ignore', category=FutureWarning, module='.*yolov5.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detection using YOLOv5
    
    Detects objects in images from Vector's camera and filters
    by confidence threshold. Maps COCO class IDs to readable names.
    """

    # COCO dataset class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        model_path: str = "yolov5n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        img_size: int = 640
    ):
        """
        Initialize ObjectDetector
        
        Args:
            model_path: Path to YOLOv5 model weights
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device for inference ("cpu" or "cuda")
            img_size: Input image size for model
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.img_size = img_size
        
        # Validate model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load YOLOv5 model
        try:
            logger.info(f"Loading YOLOv5 model from {model_path}...")
            # PyTorch 2.6+ changed weights_only default to True, causing YOLOv5 loading to fail
            # Temporarily patch torch.load to use weights_only=False for YOLOv5 models
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False for YOLOv5 model loading
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            try:
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=model_path,
                    force_reload=False,
                    trust_repo=True
                )
            finally:
                # Restore original torch.load
                torch.load = original_load
            
            self.model.conf = confidence_threshold
            self.model.to(device)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        # Statistics
        self.total_detections = 0
        self.total_frames = 0
        self.inference_times = []
        
        logger.info(
            f"ObjectDetector initialized: threshold={confidence_threshold}, "
            f"device={device}, img_size={img_size}"
        )

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image
        
        Args:
            image: Image as numpy array (H, W, 3) in RGB format
        
        Returns:
            List of detections, each containing:
                - class: Object class name (str)
                - confidence: Detection confidence (float)
                - bbox: Bounding box [x1, y1, x2, y2] (list)
                - class_id: COCO class ID (int)
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(image, size=self.img_size)
        
        # Extract detections
        detections = []
        for detection in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls = detection.tolist()
            
            # Apply confidence threshold
            if conf >= self.confidence_threshold:
                class_id = int(cls)
                class_name = self.get_class_name(class_id)
                
                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": class_id
                })
        
        # Update statistics
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_detections += len(detections)
        self.total_frames += 1
        
        logger.debug(
            f"Detected {len(detections)} objects in {inference_time*1000:.1f}ms "
            f"({1/inference_time:.1f} FPS)"
        )
        
        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in batch of images
        
        Args:
            images: List of images as numpy arrays
        
        Returns:
            List of detection lists (one per image)
        """
        start_time = time.time()
        
        # Run batch inference
        results = self.model(images, size=self.img_size)
        
        # Extract detections for each image
        all_detections = []
        for result in results.xyxy:
            detections = []
            for detection in result:
                x1, y1, x2, y2, conf, cls = detection.tolist()
                
                if conf >= self.confidence_threshold:
                    class_id = int(cls)
                    class_name = self.get_class_name(class_id)
                    
                    detections.append({
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": class_id
                    })
            
            all_detections.append(detections)
        
        # Update statistics
        batch_time = time.time() - start_time
        fps = len(images) / batch_time
        
        logger.info(
            f"Batch detection: {len(images)} images in {batch_time:.2f}s "
            f"({fps:.1f} FPS)"
        )
        
        return all_detections

    def get_class_name(self, class_id: int) -> str:
        """
        Get readable class name from COCO class ID
        
        Args:
            class_id: COCO class ID (0-79)
        
        Returns:
            Class name string
        """
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        else:
            return f"unknown_{class_id}"

    def filter_by_class(
        self,
        detections: List[Dict[str, Any]],
        class_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter detections by class names
        
        Args:
            detections: List of detections
            class_names: List of class names to keep
        
        Returns:
            Filtered detection list
        """
        return [d for d in detections if d["class"] in class_names]

    def get_most_confident(
        self,
        detections: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top-k most confident detections
        
        Args:
            detections: List of detections
            top_k: Number of detections to return
        
        Returns:
            Top-k detections sorted by confidence
        """
        sorted_detections = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True
        )
        return sorted_detections[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = 0.0
        avg_fps = 0.0
        
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        avg_detections_per_frame = 0.0
        if self.total_frames > 0:
            avg_detections_per_frame = self.total_detections / self.total_frames
        
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": avg_detections_per_frame,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "avg_fps": avg_fps,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics"""
        self.total_detections = 0
        self.total_frames = 0
        self.inference_times = []
        logger.info("Statistics reset")

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ObjectDetector("
            f"model={Path(self.model_path).name}, "
            f"threshold={self.confidence_threshold}, "
            f"device={self.device}, "
            f"frames={self.total_frames}, "
            f"detections={self.total_detections})"
        )


# ============================================================================
# Vector Camera Integration
# ============================================================================

def vector_camera_to_numpy(camera_image) -> np.ndarray:
    """
    Convert Vector SDK camera image to numpy array
    
    Args:
        camera_image: anki_vector.camera.CameraImage object
    
    Returns:
        NumPy array (H, W, 3) in RGB format
    """
    # Vector camera returns PIL Image
    from PIL import Image
    
    if hasattr(camera_image, 'raw_image'):
        pil_image = camera_image.raw_image
    else:
        pil_image = camera_image
    
    # Convert to RGB numpy array
    image_array = np.array(pil_image.convert('RGB'))
    
    return image_array


async def detect_from_vector_camera(
    detector: ObjectDetector,
    robot
) -> List[Dict[str, Any]]:
    """
    Capture frame from Vector and detect objects
    
    Args:
        detector: ObjectDetector instance
        robot: anki_vector.Robot instance
    
    Returns:
        List of detections
    """
    # Capture camera frame
    camera_image = await robot.camera.capture_single_image()
    
    # Convert to numpy
    image_array = vector_camera_to_numpy(camera_image)
    
    # Detect objects
    detections = detector.detect(image_array)
    
    return detections


# ============================================================================
# Utility Functions
# ============================================================================

def create_object_detector(config: dict = None) -> ObjectDetector:
    """
    Factory function to create ObjectDetector with config
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured ObjectDetector instance
    """
    if config is None:
        config = {}
    
    return ObjectDetector(
        model_path=config.get('model_path', 'yolov5n.pt'),
        confidence_threshold=config.get('confidence_threshold', 0.5),
        device=config.get('device', 'cpu'),
        img_size=config.get('img_size', 640)
    )


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Image as numpy array
        detections: List of detections
    
    Returns:
        Image with drawn bounding boxes
    """
    import cv2
    
    image_copy = image.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        class_name = detection["class"]
        confidence = detection["confidence"]
        
        # Draw rectangle
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image_copy,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return image_copy


def detection_summary(detections: List[Dict[str, Any]]) -> str:
    """
    Create human-readable summary of detections
    
    Args:
        detections: List of detections
    
    Returns:
        Summary string
    """
    if not detections:
        return "No objects detected"
    
    # Count objects by class
    class_counts = {}
    for detection in detections:
        class_name = detection["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Build summary
    parts = []
    for class_name, count in sorted(class_counts.items()):
        if count == 1:
            parts.append(f"1 {class_name}")
        else:
            parts.append(f"{count} {class_name}s")
    
    return "Detected: " + ", ".join(parts)
