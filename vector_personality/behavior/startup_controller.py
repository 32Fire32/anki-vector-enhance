"""
Startup Controller - Face-First Initialization Sequence

Implements intelligent startup behavior where Vector prioritizes human interaction
before exploring the environment. Follows the principle of social-first robotics.

Startup Sequence:
1. Look for person/face (scan environment)
2. Attempt face recognition via SDK
3. If recognized: Greet by name
4. If unknown: Trigger face enrollment + ask for name
5. Only after face interaction: Signal ready for environment scanning

Author: Vector Personality Enhancement Team
Date: 2025-12-16
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import anki_vector
from anki_vector.util import degrees

logger = logging.getLogger(__name__)


class StartupController:
    """
    Manages Vector's startup sequence with face-first priority.
    
    Ensures Vector establishes human connection before exploring environment.
    Implements social robotics principle: people before places.
    """
    
    def __init__(
        self,
        robot: anki_vector.Robot,
        db_connector: Any,
        tts: Optional[Any] = None,
        face_detection_handler: Optional[Any] = None,
        face_scan_timeout: float = 15.0,
        head_scan_angle_range: Tuple[float, float] = (-20.0, 30.0)
    ):
        """
        Initialize startup controller.
        
        Args:
            robot: Vector robot instance
            db_connector: Database connector for face lookup
            tts: Text-to-speech module for greetings
            face_detection_handler: Face detection handler to mark greeted faces
            face_scan_timeout: Seconds to scan for faces before giving up
            head_scan_angle_range: (min_angle, max_angle) in degrees for head scanning
        """
        self.robot = robot
        self.db = db_connector
        self.tts = tts
        self.face_detection_handler = face_detection_handler
        self.face_scan_timeout = face_scan_timeout
        self.head_scan_angle_range = head_scan_angle_range
        
        self.startup_complete = False
        self.face_found = False
        self.face_id = None
        self.face_name = None
        
    async def execute_startup_sequence(self) -> Dict[str, Any]:
        """
        Execute the complete face-first startup sequence.
        
        Returns:
            Dict with startup results:
            {
                'success': bool,
                'face_found': bool,
                'face_recognized': bool,
                'face_id': Optional[str],
                'face_name': Optional[str],
                'message': str
            }
        """
        logger.info("🚀 Starting face-first initialization sequence...")
        
        result = {
            'success': False,
            'face_found': False,
            'face_recognized': False,
            'face_id': None,
            'face_name': None,
            'message': ''
        }
        
        try:
            # Step 1: Look for face
            logger.info("👁️ Step 1: Scanning for faces...")
            face_data = await self._scan_for_face()
            
            if not face_data:
                logger.info("❌ No face detected during startup scan")
                result['message'] = "No person found during startup - proceeding with autonomous behavior"
                # Don't announce being alone - just proceed silently
                return result
            
            result['face_found'] = True
            self.face_found = True
            logger.info(f"✅ Face detected: ID={face_data['face_id']}")
            
            # Step 2: Attempt recognition
            logger.info("🔍 Step 2: Attempting face recognition...")
            recognized = await self._recognize_face(face_data)
            
            if recognized:
                # Step 3a: Greet by name
                result['face_recognized'] = True
                result['face_id'] = self.face_id
                result['face_name'] = self.face_name
                logger.info(f"👋 Step 3a: Greeting recognized person: {self.face_name}")
                await self._greet_known_person(self.face_name)
                logger.info(f"✅ Greeting complete for {self.face_name}")
                result['success'] = True
                result['message'] = f"Recognized and greeted {self.face_name}"
            else:
                # Step 3b: Enroll new face and ask for name
                logger.info("❓ Step 3b: Unknown face - initiating enrollment...")
                enrolled_name = await self._enroll_and_ask_name(face_data)
                
                if enrolled_name:
                    result['face_recognized'] = False  # Was unknown, now enrolled
                    result['face_id'] = self.face_id
                    result['face_name'] = enrolled_name
                    result['success'] = True
                    result['message'] = f"Enrolled new person: {enrolled_name}"
                else:
                    result['message'] = "Face enrollment failed or declined"
            
            # Step 4: Mark startup complete (ready for environment scanning)
            self.startup_complete = True
            logger.info("✅ Face-first startup sequence complete")
            
        except Exception as e:
            logger.error(f"❌ Startup sequence error: {e}", exc_info=True)
            result['message'] = f"Error during startup: {e}"
        
        return result
    
    async def _scan_for_face(self) -> Optional[Dict[str, Any]]:
        """
        Scan environment for faces by moving head and checking visible faces.
        
        Returns:
            Face data dict if found, None otherwise
        """
        try:
            logger.info("🔄 Scanning for faces (moving head)...")

            # Brief delay to let the gRPC connection stabilise after initial SDK connect
            # (avoids "connection has been closed" on the first behavior command)
            await asyncio.sleep(1.5)

            # Move head through range to scan for faces
            min_angle, max_angle = self.head_scan_angle_range
            
            # Start scan
            start_time = asyncio.get_event_loop().time()
            
            # Scan positions: high to medium (typical human face heights when standing/sitting)
            # Focus on realistic heights: 35° (standing), 25° (sitting), 15° (child/crouching)
            scan_angles = [35.0, 25.0, 15.0]
            
            for angle in scan_angles:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > self.face_scan_timeout:
                    logger.info("⏱️ Face scan timeout reached")
                    break
                
                # Move head to scan position
                logger.info(f"📐 Scanning at head angle: {angle}°")
                self.robot.behavior.set_head_angle(degrees(angle))
                
                # Wait for head to settle and camera to process face recognition
                # Face recognition takes several seconds to process
                await asyncio.sleep(5.0)  # 5 seconds per angle for reliable face detection
                
                # Check for visible faces (convert generator to list)
                visible_faces = list(self.robot.world.visible_faces)
                if visible_faces and len(visible_faces) > 0:
                    face = visible_faces[0]  # Take first detected face
                    logger.info(f"✅ Face found at angle {angle}°")
                    
                    # Return face data
                    return {
                        'face_id': face.face_id,
                        'name': face.name if face.name else 'Unknown',
                        'expression': face.expression,
                        'timestamp': datetime.now()
                    }
            
            logger.info("❌ No faces found during scan")
            return None
            
        except Exception as e:
            logger.warning(f"Face scan skipped (connection not ready): {e}")
            return None
    
    async def _recognize_face(self, face_data: Dict[str, Any]) -> bool:
        """
        Check if face is recognized (in database).
        
        Args:
            face_data: Face information from SDK
            
        Returns:
            True if recognized, False if unknown
        """
        try:
            # Check if SDK already has name
            sdk_name = face_data.get('name')
            sdk_face_id = face_data['face_id']
            
            if sdk_name and sdk_name != 'Unknown':
                logger.info(f"✅ SDK recognizes face: {sdk_name} (SDK ID: {sdk_face_id})")
                self.face_name = sdk_name
                
                # Get database UUID for this face
                if self.db:
                    rows = await self.db.query(
                        "SELECT face_id FROM faces WHERE name = ?",
                        (sdk_name,)
                    )
                    if rows and len(rows) > 0:
                        self.face_id = rows[0]['face_id']  # Database UUID
                        logger.debug(f"Database UUID: {self.face_id}")
                    else:
                        self.face_id = None
                else:
                    self.face_id = None
                
                return True
            
            # Query database for face by SDK ID
            if self.db:
                rows = await self.db.query(
                    "SELECT face_id, name FROM faces WHERE sdk_face_id = ?",
                    (sdk_face_id,)
                )
                
                if rows and len(rows) > 0:
                    self.face_id = rows[0]['face_id']  # Database UUID
                    self.face_name = rows[0]['name']
                    logger.info(f"✅ Database recognizes face: {self.face_name} (UUID: {self.face_id})")
                    return True
            
            logger.info("❓ Face not recognized (unknown)")
            return False
            
        except Exception as e:
            logger.error(f"Face recognition error: {e}", exc_info=True)
            return False
    
    async def _greet_known_person(self, name: str):
        """
        Greet a recognized person by name.
        
        Args:
            name: Person's name
        """
        try:
            greeting = f"Ciao {name}!"
            logger.info(f"👋 Greeting: {greeting}")
            
            # Speak greeting first for immediate feedback
            if self.tts:
                await self.tts.speak(greeting)
            else:
                # Fallback to built-in TTS
                self.robot.behavior.say_text(greeting)
            
            # Play greeting animation after speaking (non-blocking)
            try:
                # Don't await - let it play in background
                self.robot.anim.play_animation('anim_greeting_hello_01')
                logger.info("🎬 Started greeting animation")
            except Exception as e:
                logger.debug(f"Animation failed (non-critical): {e}")
            
            # Mark face as greeted to prevent duplicate greeting from continuous system
            if self.face_detection_handler and self.face_id:
                self.face_detection_handler._greeted_faces.add(self.face_id)
                logger.debug(f"✅ Marked face as greeted: {self.face_id}")

                # Enable announce_faces flag in working memory so the next LLM response may
                # mention the recognized person if relevant (one-time, expires)
                try:
                    if hasattr(self.face_detection_handler, 'working_memory') and self.face_detection_handler.working_memory:
                        self.face_detection_handler.working_memory.set_announce_faces(180)
                        logger.debug('✅ Enabled face announcement window (startup greeting)')
                except Exception as e:
                    logger.debug(f"Could not enable face announcement flag: {e}")
                
        except Exception as e:
            logger.error(f"Greeting error: {e}", exc_info=True)
    
    async def _enroll_and_ask_name(self, face_data: Dict[str, Any]) -> Optional[str]:
        """
        Enroll unknown face and ask for name.
        
        Vector SDK should handle face enrollment automatically when it sees
        an unknown face multiple times. We'll prompt the user to interact.
        
        Args:
            face_data: Face information from SDK
            
        Returns:
            Person's name if successfully enrolled, None otherwise
        """
        try:
            logger.info("📝 Requesting face enrollment...")
            
            # Speak enrollment request
            enrollment_request = "Non ti conosco. Come ti chiami?"
            logger.info(f"❓ Asking: {enrollment_request}")
            
            if self.tts:
                await self.tts.speak(enrollment_request)
            else:
                self.robot.behavior.say_text(enrollment_request)
            
            # Note: Actual enrollment happens via Vector SDK's built-in mechanism
            # User should use Vector's button + voice command or app to enroll
            # We'll wait a moment for them to respond
            logger.info("⏳ Waiting for user to provide name via voice/app...")
            
            # Give user time to respond (they should say "My name is [name]" or use app)
            await asyncio.sleep(8.0)
            
            # Check if face now has a name (SDK updated it)
            visible_faces = list(self.robot.world.visible_faces)
            if visible_faces:
                for face in visible_faces:
                    if face.face_id == face_data['face_id'] and face.name and face.name != 'Unknown':
                        logger.info(f"✅ Face enrolled with name: {face.name}")
                        self.face_id = face.face_id
                        self.face_name = face.name
                        
                        # Greet the newly enrolled person
                        greeting = f"Piacere di conoscerti, {face.name}!"
                        if self.tts:
                            await self.tts.speak(greeting)
                        else:
                            self.robot.behavior.say_text(greeting)
                        
                        return face.name
            
            logger.warning("❌ Face enrollment did not complete (no name provided)")
            
            # Store as "Unknown" in database for now (store sdk_face_id mapping)
            if self.db:
                try:
                    sdk_id = face_data['face_id']
                    # Use the connector's create_face helper to correctly insert UUID and sdk mapping
                    new_face_id = await self.db.create_face(name='Unknown', sdk_face_id=sdk_id)
                    logger.info(f"💾 Stored unknown face in database (UUID: {new_face_id}, SDK ID: {sdk_id})")
                except Exception as e:
                    logger.error(f"Failed to store unknown face: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Face enrollment error: {e}", exc_info=True)
            return None
    
    def is_startup_complete(self) -> bool:
        """Check if startup sequence has completed."""
        return self.startup_complete
    
    def get_detected_face(self) -> Optional[Dict[str, Any]]:
        """
        Get information about face detected during startup.
        
        Returns:
            Dict with face_id and face_name if face was found, None otherwise
        """
        if self.face_found:
            return {
                'face_id': self.face_id,
                'face_name': self.face_name
            }
        return None
    
    def reset(self):
        """Reset controller state for new startup sequence."""
        self.startup_complete = False
        self.face_found = False
        self.face_id = None
        self.face_name = None
        logger.info("🔄 Startup controller reset")
