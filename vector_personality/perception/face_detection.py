"""
Face detection integration for Vector Personality Project.
Connects Vector SDK face events to memory system.

Principle I: Authentic Perception + Principle II: Persistent Memory
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import anki_vector
from anki_vector.events import Events

from vector_personality.memory import SQLServerConnector, WorkingMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetectionHandler:
    """
    Handles face detection events from Vector SDK.
    Integrates with dual-tier memory architecture.
    
    Principle I: Authentic Perception
    - Vector SDK provides raw face detection events
    - We don't hallucinate faces - only respond to actual SDK events
    
    Principle II: Persistent Memory
    - New faces stored in SQL Server (long-term)
    - Face observations tracked in WorkingMemory (session)
    """
    
    def __init__(
        self,
        robot: anki_vector.Robot,
        db_connector: SQLServerConnector,
        working_memory: WorkingMemory,
        tts: Optional[Any] = None
    ):
        """
        Initialize face detection handler.
        
        Args:
            robot: Anki Vector robot instance
            db_connector: SQL Server connector
            working_memory: Working memory instance
            tts: Optional text-to-speech module for greetings
        """
        self.robot = robot
        self.db = db_connector
        self.working_memory = working_memory
        self.tts = tts
        
        # Face ID mapping (SDK face_id -> database face_id)
        self._face_id_map: Dict[int, str] = {}  # SDK int ID -> database UUID
        
        # Track which faces we've greeted this session (to avoid repeated greetings)
        self._greeted_faces: set = set()
        
        # Startup mode flag - disables continuous greeting during startup sequence
        self.startup_mode = True
        
        # Event throttling - only process face events every N seconds
        self._last_event_time: Dict[int, float] = {}  # SDK face_id -> timestamp
        self._event_throttle_seconds = 2.0  # Process same face max once per 2 seconds
        
        # Event subscription
        self._event_handler = None
        self._polling_task = None
        
        logger.info("Face detection handler initialized")
    
    async def start(self):
        """Start listening to face events from Vector SDK."""
        # Subscribe to face events
        self._event_handler = self.robot.events.subscribe(
            self._on_face_appeared,
            Events.robot_observed_face
        )
        
        # Start polling task as backup (since events seem unreliable)
        self._polling_task = asyncio.create_task(self._poll_faces())
        
        logger.info("🎭 Face detection event subscription ACTIVE - waiting for faces...")
        logger.info(f"🎭 Event handler registered: {self._event_handler}")
    
    async def stop(self):
        """Stop listening to face events."""
        if self._event_handler:
            self.robot.events.unsubscribe(self._event_handler)
            self._event_handler = None
            
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
        logger.info("Face detection stopped")
    
    async def _poll_faces(self):
        """Poll for faces in case events don't fire."""
        logger.info("👀 Face polling started")
        while True:
            try:
                # Skip polling during startup to avoid flood
                if self.startup_mode:
                    await asyncio.sleep(1.0)
                    continue
                
                # Check visible faces
                # robot.world.visible_faces returns a generator of Face objects
                import time
                current_time = time.time()
                
                for face in self.robot.world.visible_faces:
                    sdk_face_id = face.face_id
                    
                    # Apply same throttling as event handler
                    if sdk_face_id in self._last_event_time:
                        time_since_last = current_time - self._last_event_time[sdk_face_id]
                        if time_since_last < self._event_throttle_seconds:
                            continue  # Skip - too soon
                    
                    self._last_event_time[sdk_face_id] = current_time
                    await self._process_face(face)
                
                await asyncio.sleep(2.0)  # Poll every 2 seconds (was 1s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Face polling error: {e}")
                await asyncio.sleep(1.0)

    async def _on_face_appeared(self, robot, event_type, event):
        """
        Callback for face appearance events from SDK.
        """
        sdk_face_id = event.face_id
        
        # Throttle: Only process same face every N seconds to reduce spam
        import time
        current_time = time.time()
        if sdk_face_id in self._last_event_time:
            time_since_last = current_time - self._last_event_time[sdk_face_id]
            if time_since_last < self._event_throttle_seconds:
                return  # Skip this event (too soon)
        
        self._last_event_time[sdk_face_id] = current_time
        logger.debug(f"🎭 Processing face event: SDK Face ID={sdk_face_id}")
        
        # Use the face object from the world if available, or construct a minimal one
        # The event usually has the face object attached or we can look it up
        face = None
        try:
            face = self.robot.world.get_face(event.face_id)
        except:
            pass
            
        if face:
            await self._process_face(face)
        else:
            logger.warning(f"Could not retrieve face object for ID {event.face_id}")

    async def _process_face(self, face):
        """
        Process a detected face (from event or polling).
        
        Args:
            face: Anki Vector Face object
        """
        try:
            sdk_face_id = face.face_id
            face_name = face.name if hasattr(face, 'name') and face.name else None
            
            # Check if we've seen this face before in this session
            db_face_id = self._face_id_map.get(sdk_face_id)
            
            if not db_face_id:
                # New face in this session - check database
                db_face_id = await self._get_or_create_face(sdk_face_id, face_name)
                self._face_id_map[sdk_face_id] = db_face_id
            
            # Get face history from database
            face_record = await self.db.get_face_by_id(db_face_id)
            
            # Determine if this is a new face (first time seeing)
            is_new_face = face_record and face_record['total_interactions'] == 0
            
            # Determine mood impact
            if face_record and face_record['total_interactions'] > 0:
                # Familiar face = positive mood
                mood_impact = 10
            else:
                # New face = neutral/curious
                mood_impact = 5
            
            # Update working memory
            self.working_memory.observe_face(
                face_id=db_face_id,
                name=face_name or face_record.get('name'),
                mood_impact=mood_impact
            )
            
            # Update database (last_seen, interaction count)
            await self.db.update_face_interaction(
                face_id=db_face_id,
                mood_change=mood_impact
            )
            
            # Greet face if we haven't greeted them yet this session
            await self._maybe_greet_face(db_face_id, face_name or face_record.get('name'), is_new_face)
            
        except Exception as e:
            logger.error(f"Error handling face event: {e}", exc_info=True)
    
    async def _get_or_create_face(self, sdk_face_id: int, name: Optional[str]) -> str:
        """
        Get database face_id for SDK face, or create new record.

        This version also captures a face embedding (if possible) and attempts to
        reuse an existing face record via similarity matching (T128).

        Args:
            sdk_face_id: Vector SDK face ID (int)
            name: Optional person name

        Returns:
            Database face_id (UUID string)
        """
        import os
        import numpy as np
        from datetime import datetime
        from vector_personality.perception.object_detector import vector_camera_to_numpy

        # Environment-based thresholds
        merge_window = int(os.getenv('FACE_MERGE_WINDOW_SECONDS', '3600'))
        duplicate_threshold = float(os.getenv('FACE_DUPLICATE_THRESHOLD', '0.85'))
        auto_merge_threshold = float(os.getenv('FACE_AUTO_MERGE_THRESHOLD', '0.95'))

        # 1) Check if we know this person by name
        known_face_id = None
        if name:
            face = await self.db.get_face_by_name(name)
            if face:
                logger.info(f"Found existing face in database: {name}")
                known_face_id = str(face['face_id'])
                # Continue to generate embedding for known faces too!

        # 2) Attempt to capture face crop and generate embedding
        embedding_vec = None
        try:
            # Capture a single camera frame (synchronous SDK call)
            logger.info("📸 Capturing face crop for embedding generation...")
            camera_image = self.robot.camera.capture_single_image()
            image_array = vector_camera_to_numpy(camera_image)

            # Lazy-import facenet components
            try:
                from facenet_pytorch import MTCNN, InceptionResnetV1
            except Exception as e:
                logger.warning(f"Face embedding libraries not available: {e}")
                MTCNN = None
                InceptionResnetV1 = None

            if MTCNN and InceptionResnetV1:
                # Create models if not already present on handler
                if not hasattr(self, '_mtcnn'):
                    self._mtcnn = MTCNN(keep_all=True)
                if not hasattr(self, '_embedder'):
                    self._embedder = InceptionResnetV1(pretrained='vggface2').eval()

                # Detect faces and pick largest box
                boxes, probs = self._mtcnn.detect(image_array)
                if boxes is not None and len(boxes) > 0:
                    # Choose box with largest area
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    best_idx = int(np.argmax(areas))
                    box = boxes[best_idx].astype(int)
                    x1, y1, x2, y2 = box
                    crop = image_array[y1:y2, x1:x2]
                    if crop.size > 0:
                        # Resize/correct to PIL image expected by embedder
                        from PIL import Image
                        crop_pil = Image.fromarray(crop).convert('RGB')
                        # Get tensor embedding
                        import torch
                        face_tensor = self._mtcnn(crop_pil)
                        if face_tensor is not None:
                            # face_tensor may be unbatched (3D) or batched (4D)
                            try:
                                if isinstance(face_tensor, torch.Tensor):
                                    if face_tensor.ndim == 3:
                                        inp = face_tensor.unsqueeze(0)  # (1,3,H,W)
                                    elif face_tensor.ndim == 4:
                                        inp = face_tensor  # already batched (N,3,H,W)
                                    elif face_tensor.ndim == 5:
                                        # Some backends may return an extra batch dimension (1,1,3,H,W)
                                        try:
                                            inp = face_tensor.squeeze(1)  # (1,3,H,W)
                                            logger.debug(f"Squeezed extra dim from face tensor: new shape {tuple(inp.shape)}")
                                        except Exception:
                                            logger.warning(f"Unexpected face tensor shape: {tuple(face_tensor.shape)}")
                                            inp = None
                                    else:
                                        logger.warning(f"Unexpected face tensor shape: {tuple(face_tensor.shape)}")
                                        inp = None
                                else:
                                    # Some backends may return a list/tuple - convert first element
                                    tensor_candidate = face_tensor[0] if len(face_tensor) > 0 else None
                                    if tensor_candidate is not None:
                                        inp = tensor_candidate.unsqueeze(0) if tensor_candidate.ndim == 3 else tensor_candidate
                                    else:
                                        inp = None

                                if inp is not None:
                                    with torch.no_grad():
                                        emb = self._embedder(inp)
                                    # Use first embedding when multiple faces returned
                                    embedding_vec = emb[0].cpu().numpy().astype(np.float32)
                                    logger.info(f"✅ Generated face embedding (512-dim)")
                            except Exception as e:
                                logger.warning(f"Error processing face tensor for embedding: {e}")
        except Exception as e:
            logger.warning(f"Failed to capture/compute face embedding: {e}")

        # 3) If we have an embedding, try to find a similar existing face
        if embedding_vec is not None and not known_face_id:  # Only search if not already known by name
            logger.info(f"🔍 Searching for similar faces (threshold={duplicate_threshold:.2f})...")
            try:
                matches = await self.db.find_similar_faces(embedding_vec, top_k=3, min_score=duplicate_threshold)
                if matches:
                    top = matches[0]
                    logger.info(f"Face embedding matched existing face {top['face_id']} (score={top['score']:.3f})")
                    # If very confident, reuse canonical face id or auto-merge
                    if top['score'] >= auto_merge_threshold:
                        # Use matched face id
                        matched_id = top['face_id']
                        # Store embedding for matched face too
                        await self.db.add_face_embedding(matched_id, embedding_vec.tobytes(), embedding_vec.size)
                        return matched_id
                    else:
                        # Reuse matched id if last seen within merge window
                        face_info = await self.db.get_face_by_id(top['face_id'])
                        if face_info:
                            last_seen = face_info.get('last_seen')
                            if isinstance(last_seen, datetime):
                                delta = (datetime.now() - last_seen).total_seconds()
                                if delta <= merge_window:
                                    matched_id = top['face_id']
                                    await self.db.add_face_embedding(matched_id, embedding_vec.tobytes(), embedding_vec.size)
                                    logger.info(f"Reusing recent unknown face {matched_id} (score={top['score']:.3f}, last_seen={delta:.0f}s)")
                                    return matched_id
            except Exception as e:
                logger.warning(f"Error while searching for similar faces: {e}")

        # 4) If no suitable match, create new face record OR use known face
        if known_face_id:
            # We found a known person by name - use that ID and ensure sdk mapping exists
            face_id = known_face_id
            try:
                # If SDK ID is provided but not stored, update the DB mapping
                if sdk_face_id is not None:
                    row = await self.db.query("SELECT sdk_face_id FROM faces WHERE face_id = ?", (face_id,))
                    if row and row[0].get('sdk_face_id') is None:
                        await self.db.execute("UPDATE faces SET sdk_face_id = ? WHERE face_id = ?", (sdk_face_id, face_id))
                        logger.info(f"🔁 Added sdk_face_id mapping for face {face_id[:8]} -> SDK {sdk_face_id}")
            except Exception as e:
                logger.warning(f"Failed to update sdk mapping for {face_id}: {e}")
        else:
            # Create new unknown face and record sdk_face_id
            face_id = await self.db.create_face(name=name, sdk_face_id=sdk_face_id)
            logger.info(f"Created new face in database: {name or 'Unknown'} (ID: {face_id}, SDK ID: {sdk_face_id})")

        # Store embedding if available (for both known and new faces)
        if embedding_vec is not None:
            try:
                await self.db.add_face_embedding(face_id, embedding_vec.tobytes(), embedding_vec.size)
                logger.info(f"🔢 Stored embedding for face {name or 'Unknown'} ({face_id[:8]}...)")
            except Exception as e:
                logger.warning(f"Failed to store embedding for {face_id}: {e}")

        return face_id
    
    async def _maybe_greet_face(self, face_id: str, name: Optional[str], is_new: bool):
        """
        Greet a face if we haven't greeted them yet this session.
        
        Args:
            face_id: Database face ID
            name: Person's name (or None/Unknown)
            is_new: True if this is first time seeing this face ever
        """
        # Don't greet during startup (startup controller handles that)
        if self.startup_mode:
            logger.debug(f"⏸️ Skipping continuous greeting (startup mode active)")
            return
        
        # Don't greet if we've already greeted this face this session
        if face_id in self._greeted_faces:
            return
        
        # Don't greet unknown faces
        if not name or name == 'Unknown':
            return
        
        try:
            # Mark as greeted
            self._greeted_faces.add(face_id)
            
            # Play greeting animation
            try:
                self.robot.anim.play_animation('anim_greeting_hello_01')
                logger.info("🎬 Played greeting animation")
            except Exception as e:
                logger.debug(f"Animation failed (non-critical): {e}")
            
            # Speak greeting
            greeting = f"Ciao {name}!"
            logger.info(f"👋 Greeting: {greeting}")
            
            if self.tts:
                await self.tts.speak(greeting)
            else:
                # Fallback to built-in TTS
                self.robot.behavior.say_text(greeting)
                
        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
    
    async def manually_identify_face(self, sdk_face_id: int, name: str) -> bool:
        """
        Manually identify a face (user provides name).
        
        Args:
            sdk_face_id: Vector SDK face ID
            name: Person's name
        
        Returns:
            True if successful
        """
        try:
            # Get or create database record
            db_face_id = await self._get_or_create_face(sdk_face_id, name)
            
            # Update mapping
            self._face_id_map[sdk_face_id] = db_face_id
            
            # Update working memory
            if db_face_id in self.working_memory.current_faces:
                self.working_memory.current_faces[db_face_id].name = name
            
            logger.info(f"Manually identified face: SDK ID {sdk_face_id} → {name}")
            return True
        except Exception as e:
            logger.error(f"Error manually identifying face: {e}", exc_info=True)
            return False
    
    def disable_startup_mode(self):
        """
        Disable startup mode, enabling continuous face greeting.
        Called by VectorAgent after startup sequence completes.
        """
        self.startup_mode = False
        logger.info("✅ Startup mode disabled - continuous greeting enabled")
    
    def get_current_faces(self) -> list:
        """Get list of faces currently in view (from working memory)."""
        active_faces = self.working_memory.get_active_faces(max_age_seconds=30)
        return [
            {
                'face_id': obs.face_id,
                'name': obs.name or 'Unknown',
                'seen_for_seconds': (datetime.now() - obs.first_seen_session).total_seconds(),
                'interaction_count': obs.interaction_count
            }
            for obs in active_faces
        ]
    
    async def get_face_conversation_history(self, face_id: str, limit: int = 10) -> list:
        """
        Get conversation history with a specific face.
        
        Args:
            face_id: Database face UUID
            limit: Maximum conversations to retrieve
        
        Returns:
            List of conversation dicts
        """
        history = await self.db.get_face_history(face_id, limit=limit)
        return [
            {
                'timestamp': conv['timestamp'],
                'text': conv['text'],
                'response': conv['response_text'],
                'mood': conv['emotional_context']
            }
            for conv in history
        ]


# ========== Convenience Functions ==========

async def setup_face_detection(
    robot: anki_vector.Robot,
    db_connector: SQLServerConnector,
    working_memory: WorkingMemory
) -> FaceDetectionHandler:
    """
    Setup and start face detection integration.
    
    Args:
        robot: Vector robot instance
        db_connector: SQL Server connector
        working_memory: Working memory instance
    
    Returns:
        FaceDetectionHandler instance (already started)
    """
    handler = FaceDetectionHandler(robot, db_connector, working_memory)
    await handler.start()
    return handler
