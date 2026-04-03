"""
SQL Server connector for Vector Personality Project.
Handles database connectivity with Windows authentication and async queries.

Principle II: Persistent Memory Architecture
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import json
import pyodbc
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLServerConnector:
    """
    SQL Server connector with Windows authentication support.
    Provides async query interface using ThreadPoolExecutor.
    
    Principle II: Persistent Memory Architecture
    All long-term storage goes through this connector.
    """
    
    def __init__(
        self,
        server: str = 'localhost',
        database: str = 'vector_memory',
        trusted_connection: bool = True,
        timeout: int = 30,
        driver: str = 'ODBC Driver 17 for SQL Server'
    ):
        """
        Initialize SQL Server connector.
        
        Args:
            server: SQL Server hostname (default: localhost)
            database: Database name (default: vector_memory)
            trusted_connection: Use Windows authentication (default: True)
            timeout: Connection timeout in seconds
            driver: ODBC driver name
        """
        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection
        self.timeout = timeout
        self.driver = driver
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        # Connection string
        if trusted_connection:
            self.connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout={timeout};"
            )
        else:
            raise NotImplementedError("Username/password auth not implemented (use Windows Auth)")
        
        logger.info(f"SQL Server connector initialized: {server}/{database}")
    
    def _get_connection(self) -> pyodbc.Connection:
        """Create a new database connection (synchronous)."""
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except pyodbc.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = await self.query("SELECT 1 AS test")
            return len(result) == 1 and result[0]['test'] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def query(self, sql: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute SELECT query and return results as list of dicts.
        
        Args:
            sql: SQL SELECT statement
            params: Optional parameters (tuple, list, or single value)
        
        Returns:
            List of rows as dictionaries
        """
        def _execute_query():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if params is not None:
                    # Ensure params is a tuple for pyodbc
                    if not isinstance(params, (tuple, list)):
                        cursor.execute(sql, (params,))
                    else:
                        cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                columns = [column[0] for column in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _execute_query)
    
    async def execute(self, sql: str, params: Optional[Any] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE statement.
        
        Args:
            sql: SQL statement
            params: Optional parameters (tuple, list, or single value)
        
        Returns:
            Number of rows affected
        """
        def _execute_statement():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if params is not None:
                    # Ensure params is a tuple for pyodbc
                    if not isinstance(params, (tuple, list)):
                        cursor.execute(sql, (params,))
                    else:
                        cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                conn.commit()
                return cursor.rowcount
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _execute_statement)
    
    async def execute_many(self, sql: str, params_list: List[Dict[str, Any]]) -> int:
        """
        Execute multiple statements in batch.
        
        Args:
            sql: SQL statement
            params_list: List of parameter dictionaries
        
        Returns:
            Total rows affected
        """
        def _execute_batch():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                total_rows = 0
                
                for params in params_list:
                    cursor.execute(sql, params)
                    total_rows += cursor.rowcount
                
                conn.commit()
                return total_rows
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _execute_batch)
    
    # ========== Face Operations (Principle I & II) ==========
    
    async def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face record by UUID."""
        results = await self.query(
            "SELECT * FROM faces WHERE face_id = ?",
            (face_id,)
        )
        return results[0] if results else None
    
    async def get_face_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get face record by name."""
        results = await self.query(
            "SELECT * FROM faces WHERE name = ?",
            (name,)
        )
        return results[0] if results else None
    
    async def create_face(self, name: Optional[str] = None, face_id: Optional[str] = None, sdk_face_id: Optional[int] = None) -> str:
        """
        Create new face record.
        
        Args:
            name: Optional person name
            face_id: Optional UUID (generated if not provided)
            sdk_face_id: Optional SDK integer id to map SDK face -> DB record
        
        Returns:
            face_id (UUID as string)
        """
        if face_id:
            # Insert with provided UUID and optional SDK id
            if sdk_face_id is not None:
                await self.execute(
                    """
                    INSERT INTO faces (face_id, name, first_seen, last_seen, sdk_face_id)
                    VALUES (?, ?, GETDATE(), GETDATE(), ?)
                    """,
                    (face_id, name, sdk_face_id)
                )
            else:
                await self.execute(
                    """
                    INSERT INTO faces (face_id, name, first_seen, last_seen)
                    VALUES (?, ?, GETDATE(), GETDATE())
                    """,
                    (face_id, name)
                )
            return face_id
        else:
            # Let SQL Server generate UUID; include sdk_face_id when provided
            if sdk_face_id is not None:
                await self.execute(
                    """
                    INSERT INTO faces (name, first_seen, last_seen, sdk_face_id)
                    VALUES (?, GETDATE(), GETDATE(), ?)
                    """,
                    (name, sdk_face_id)
                )
            else:
                await self.execute(
                    """
                    INSERT INTO faces (name, first_seen, last_seen)
                    VALUES (?, GETDATE(), GETDATE())
                    """,
                    (name,)
                )
            # Retrieve generated ID
            results = await self.query("SELECT TOP 1 face_id FROM faces ORDER BY created_at DESC")
            return str(results[0]['face_id'])
    
    async def update_face_interaction(self, face_id: str, mood_change: int = 0) -> bool:
        """
        Update face last_seen and increment interaction count.
        
        Args:
            face_id: UUID of face
            mood_change: Mood delta caused by this face (-100 to +100)
        
        Returns:
            True if successful
        """
        rows = await self.execute(
            "EXEC UpdateFaceInteraction @face_id = ?, @mood_change = ?",
            (face_id, mood_change)
        )
        return rows > 0
    
    async def get_face_history(self, face_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a face."""
        return await self.query(
            """
            SELECT TOP (?) * FROM conversations
            WHERE speaker_id = ?
            ORDER BY timestamp DESC
            """,
            (limit, face_id)
        )

    # ========== Face Embedding Operations (T128) ==========

    async def get_latest_embedding_for_face(self, face_id: str) -> Optional[Tuple[bytes, int, 'datetime']]:
        """Return the latest embedding for a face as (bytes, dim, created_at) or None."""
        results = await self.query(
            """
            SELECT TOP 1 embedding, vector_dim, created_at FROM face_embeddings
            WHERE face_id = ?
            ORDER BY created_at DESC
            """,
            (face_id,)
        )
        if not results:
            return None
        r = results[0]
        return (r['embedding'], r['vector_dim'], r['created_at'])

    async def add_face_embedding(self, face_id: str, embedding: bytes, vector_dim: int) -> bool:
        """Store raw embedding bytes for a face (latest embedding).

        Performs de-duplication: if the most recent embedding for this face
        is nearly identical (cosine similarity >= FACE_EMBEDDING_DUPLICATE_SIMILARITY)
        or was stored less than FACE_EMBEDDING_MIN_SECONDS ago, the insert is skipped.
        """
        import numpy as np
        import os
        try:
            # Configurable thresholds
            dup_sim = float(os.getenv('FACE_EMBEDDING_DUPLICATE_SIMILARITY', '0.999'))
            min_secs = float(os.getenv('FACE_EMBEDDING_MIN_SECONDS', '5'))

            latest = await self.get_latest_embedding_for_face(face_id)
            if latest is not None:
                latest_bytes, latest_dim, latest_created = latest
                # Exact byte match -> skip
                if latest_bytes == embedding:
                    logger.info(f"🔇 Skipping identical embedding for face {face_id[:8]}...")
                    return False

                # Dimensionality check
                try:
                    arr_new = np.frombuffer(embedding, dtype=np.float32)
                    arr_old = np.frombuffer(latest_bytes, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to decode embeddings for dedupe check: {e}")
                    arr_new = None
                    arr_old = None

                if arr_new is not None and arr_old is not None and arr_new.size == arr_old.size:
                    # Cosine similarity
                    arr_new_norm = arr_new / (np.linalg.norm(arr_new) + 1e-10)
                    arr_old_norm = arr_old / (np.linalg.norm(arr_old) + 1e-10)
                    score = float(np.dot(arr_new_norm, arr_old_norm))
                    if score >= dup_sim:
                        # Check time window
                        from datetime import datetime
                        if isinstance(latest_created, str):
                            # If the DB returns string, parse
                            try:
                                latest_ts = datetime.fromisoformat(latest_created)
                            except Exception:
                                latest_ts = None
                        else:
                            latest_ts = latest_created

                        if latest_ts is None or (datetime.now() - latest_ts).total_seconds() < min_secs:
                            logger.info(f"🔇 Skipping near-duplicate embedding for face {face_id[:8]} (score={score:.4f})")
                            return False

            # Insert new embedding
            await self.execute(
                """
                INSERT INTO face_embeddings (face_id, embedding, vector_dim)
                VALUES (?, ?, ?)
                """, (face_id, embedding, vector_dim)
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to add_face_embedding: {e}")
            # Fallback to inserting to avoid losing data if dedupe check fails
            try:
                await self.execute(
                    """
                    INSERT INTO face_embeddings (face_id, embedding, vector_dim)
                    VALUES (?, ?, ?)
                    """, (face_id, embedding, vector_dim)
                )
                return True
            except Exception as e2:
                logger.error(f"Failed to insert embedding after dedupe error: {e2}")
                return False

    async def get_latest_embeddings(self) -> List[Tuple[str, bytes, int]]:
        """Return the latest embedding for each face as (face_id, bytes, dim)."""
        results = await self.query(
            """
            SELECT f.face_id, fe.embedding, fe.vector_dim FROM faces f
            JOIN (
                SELECT face_id, MAX(created_at) as maxt FROM face_embeddings GROUP BY face_id
            ) latest ON f.face_id = latest.face_id
            JOIN face_embeddings fe ON fe.face_id = latest.face_id AND fe.created_at = latest.maxt
            """
        )
        # Return list of tuples
        return [(r['face_id'], r['embedding'], r['vector_dim']) for r in results]

    async def find_similar_faces(self, embedding: 'np.ndarray', top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Find similar faces by comparing against stored embeddings (cosine similarity).

        Returns list of dicts: [{'face_id': id, 'score': float}, ...] sorted descending by score.
        """
        import numpy as np

        encoded = np.asarray(embedding, dtype=np.float32)
        all_embeddings = await self.get_latest_embeddings()
        matches = []
        for face_id, emb_bytes, dim in all_embeddings:
            try:
                arr = np.frombuffer(emb_bytes, dtype=np.float32)
                if arr.size == 0:
                    continue
                # Normalize
                arr_norm = arr / (np.linalg.norm(arr) + 1e-10)
                q_norm = encoded / (np.linalg.norm(encoded) + 1e-10)
                score = float(np.dot(arr_norm, q_norm))
                if score >= min_score:
                    matches.append({'face_id': face_id, 'score': score})
            except Exception:
                continue

        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches

    async def merge_faces(self, canonical_face_id: str, merged_face_id: str) -> bool:
        """Merge `merged_face_id` into `canonical_face_id`.

        This updates conversations to point to canonical id, consolidates interaction counts, and marks merged row.
        """
        # Update conversations
        await self.execute(
            """
            UPDATE conversations SET speaker_id = ? WHERE speaker_id = ?
            """, (canonical_face_id, merged_face_id)
        )
        # Consolidate counts and last_seen
        await self.execute(
            """
            UPDATE faces SET
                total_interactions = ISNULL(a.total_interactions,0) + ISNULL(b.total_interactions,0),
                last_seen = CASE WHEN a.last_seen > b.last_seen THEN a.last_seen ELSE b.last_seen END
            FROM faces a JOIN faces b ON a.face_id = ? AND b.face_id = ?
            """, (canonical_face_id, merged_face_id)
        )
        # Mark merged row
        await self.execute(
            """
            UPDATE faces SET merged_to = ?, merged_at = GETDATE() WHERE face_id = ?
            """, (canonical_face_id, merged_face_id)
        )
        return True

    # ========== Room Operations (Principle I & V) ==========

    async def get_or_create_room(self, room_name: str) -> Dict[str, Any]:
        """Get room by name or create if not exists."""
        results = await self.query(
            "EXEC GetOrCreateRoom @room_name = ?",
            (room_name,)
        )
        return results[0] if results else None
    
    async def update_room_objects(self, room_id: str, typical_objects: List[str]) -> bool:
        """Update typical objects list for a room."""
        objects_json = json.dumps(typical_objects)
        rows = await self.execute(
            """
            UPDATE rooms
            SET typical_objects = ?, updated_at = GETDATE()
            WHERE room_id = ?
            """,
            (objects_json, room_id)
        )
        return rows > 0
    
    # ========== Object Detection (Principle I) ==========
    
    async def store_object_detection(
        self,
        object_type: str,
        confidence: float,
        room_id: Optional[str] = None,
        location_description: Optional[str] = None
    ) -> str:
        """
        Store detected object.
        
        Args:
            object_type: Class name (e.g., "fridge", "desk")
            confidence: Detection confidence (0.0-1.0)
            room_id: Optional room UUID
            location_description: Optional location text
        
        Returns:
            object_id (UUID as string)
        """
        # Check if object already exists in this room (update if so)
        existing = await self.query(
            """
            SELECT object_id FROM objects
            WHERE object_type = ? AND room_id = ?
            """,
            (object_type, room_id)
        )
        
        if existing:
            # Update existing
            object_id = str(existing[0]['object_id'])
            await self.execute(
                """
                UPDATE objects
                SET last_detected = GETDATE(),
                    detection_count = detection_count + 1,
                    confidence = ?,
                    updated_at = GETDATE()
                WHERE object_id = ?
                """,
                (confidence, object_id)
            )
            return object_id
        else:
            # Insert new
            await self.execute(
                """
                INSERT INTO objects (object_type, room_id, confidence, location_description)
                VALUES (?, ?, ?, ?)
                """,
                (object_type, room_id, confidence, location_description)
            )
            results = await self.query("SELECT TOP 1 object_id FROM objects ORDER BY created_at DESC")
            return str(results[0]['object_id'])
    
    # ========== Conversation Storage (Principle II & III) ==========
    
    async def store_conversation(
        self,
        speaker_id: str,
        text: str,
        room_id: Optional[str] = None,
        emotional_context: int = 50,
        response_text: Optional[str] = None,
        response_type: Optional[str] = None,
        vector_db: Optional[Any] = None,
        embedding_gen: Optional[Any] = None
    ) -> str:
        """
        Store conversation turn (with optional embedding generation - T140).
        
        Args:
            speaker_id: Face UUID
            text: User's speech text
            room_id: Optional room UUID
            emotional_context: Vector's mood at time of speech (0-100)
            response_text: Vector's response
            response_type: 'sdk', 'api_cheap', 'api_moderate', 'api_expensive'
            vector_db: Optional VectorDBConnector for semantic search (T140)
            embedding_gen: Optional EmbeddingGenerator for creating embeddings (T140)
        
        Returns:
            conversation_id (UUID as string)
        """
        # 1. Store in SQL Server (existing logic)
        await self.execute(
            """
            INSERT INTO conversations (speaker_id, text, room_id, emotional_context, response_text, response_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (speaker_id, text, room_id, emotional_context, response_text, response_type)
        )
        results = await self.query("SELECT TOP 1 conversation_id FROM conversations ORDER BY created_at DESC")
        conversation_id = str(results[0]['conversation_id'])

        # 2. Generate and store embedding (T140 - NEW)
        if vector_db and embedding_gen and text:
            try:
                # Combine user text + Vector response for richer context
                full_text = f"User: {text}"
                if response_text:
                    full_text += f"\nVector: {response_text}"
                
                embedding = await embedding_gen.generate_embedding(full_text)
                
                if embedding:
                    # Get timestamp from SQL result
                    timestamp = results[0].get('timestamp') or datetime.now()
                    
                    vector_db.add_embedding(
                        conversation_id=conversation_id,
                        embedding=embedding,
                        metadata={
                            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                            'speaker_id': speaker_id,
                            'has_response': response_text is not None,
                            'text_preview': text[:100]
                        }
                    )
                    logger.info(f"🔢 Embedding stored for conversation {conversation_id[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ Failed to store embedding for {conversation_id}: {e}")
                # Don't fail the whole operation if embedding storage fails

        return conversation_id

    # ========== Context Queries (T123) ==========

    async def get_recent_conversations(self, hours: int = 72, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations (joined with face names) within the last N hours."""
        return await self.query(
            """
            SELECT TOP (?)
                c.conversation_id,
                c.timestamp,
                c.text,
                c.response_text,
                c.speaker_id,
                f.name AS speaker_name,
                c.room_id
            FROM conversations c
            LEFT JOIN faces f ON c.speaker_id = f.face_id
            WHERE c.timestamp >= DATEADD(hour, -?, GETDATE())
            ORDER BY c.timestamp DESC
            """,
            (limit, int(hours))
        )

    async def get_known_faces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get known faces ordered by last seen."""
        return await self.query(
            """
            SELECT TOP (?)
                face_id,
                name,
                total_interactions,
                last_seen
            FROM faces
            ORDER BY last_seen DESC
            """,
            (limit,)
        )

    async def get_recent_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently detected objects ordered by last_detected."""
        return await self.query(
            """
            SELECT TOP (?)
                object_type,
                confidence,
                location_description,
                last_detected,
                room_id
            FROM objects
            ORDER BY last_detected DESC
            """,
            (limit,)
        )

    async def search_old_conversations(
        self,
        keywords: List[str],
        days_back: int = 90,
        limit: int = 15,
        exclude_recent_hours: int = 72,
    ) -> List[Dict[str, Any]]:
        """Search conversations by keywords with RELEVANCE RANKING.

        By default this excludes the most recent 72 hours to surface "older" memories,
        but callers can set exclude_recent_hours=0 to include recent matches too.

        Relevance scoring:
        - +10 points: Keyword in Vector's response (stored fact)
        - +5 points: Keyword in user text
        - -3 points: Contains "ricord" in user text (meta-question, not original fact)
        
        Orders by relevance_score DESC, then timestamp DESC.
        """
        keywords = [k for k in (keywords or []) if k and k.strip()]
        if not keywords:
            return []

        like_clauses: List[str] = []
        score_clauses: List[str] = []
        params: List[Any] = [limit, int(days_back)]
        
        for kw in keywords[:6]:
            like_clauses.append("(c.text LIKE ? OR c.response_text LIKE ?)")
            pattern = f"%{kw}%"
            params.extend([pattern, pattern])
            
            # Scoring: +10 for response, +5 for user text
            score_clauses.append(f"CASE WHEN c.response_text LIKE '{pattern}' THEN 10 ELSE 0 END")
            score_clauses.append(f"CASE WHEN c.text LIKE '{pattern}' THEN 5 ELSE 0 END")

        where_like = " OR ".join(like_clauses)
        score_sum = " + ".join(score_clauses)
        
        # Heavy penalty for meta-questions containing "ricord" (pushes them below fact-statements)
        score_sum += " - CASE WHEN c.text LIKE '%ricord%' THEN 10 ELSE 0 END"

        recent_exclusion = ""
        if exclude_recent_hours and int(exclude_recent_hours) > 0:
            recent_exclusion = "AND c.timestamp < DATEADD(hour, -?, GETDATE())"
            params.insert(2, int(exclude_recent_hours))

        sql = f"""
            SELECT TOP (?)
                c.timestamp,
                c.text,
                c.response_text,
                f.name AS person,
                DATEDIFF(day, c.timestamp, GETDATE()) AS days_ago,
                ({score_sum}) AS relevance_score
            FROM conversations c
            LEFT JOIN faces f ON c.speaker_id = f.face_id
            WHERE c.timestamp > DATEADD(day, -?, GETDATE())
              {recent_exclusion}
              AND ({where_like})
              AND (c.response_text NOT LIKE '%non mi ricordo%'
                   AND c.response_text NOT LIKE '%non lo so%'
                   AND c.response_text NOT LIKE '%non l''ho mai%'
                   AND c.response_text NOT LIKE '%non so chi%')
            ORDER BY relevance_score DESC, c.timestamp DESC
        """

        return await self.query(sql, tuple(params))
    
    # ========== Personality Learning (Principle V) ==========
    
    async def save_personality_adjustment(
        self,
        curiosity_delta: float = 0.0,
        touchiness_delta: float = 0.0,
        vitality_delta: float = 0.0,
        friendliness_delta: float = 0.0,
        courage_delta: float = 0.0,
        sassiness_delta: float = 0.0,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Save personality trait adjustment from user feedback.
        
        Args:
            *_delta: Trait adjustments (-1.0 to +1.0)
            feedback_text: User feedback that triggered adjustment
        
        Returns:
            True if successful
        """
        rows = await self.execute(
            """
            INSERT INTO personality_learned (
                curiosity_delta, touchiness_delta, vitality_delta,
                friendliness_delta, courage_delta, sassiness_delta,
                feedback_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (curiosity_delta, touchiness_delta, vitality_delta,
             friendliness_delta, courage_delta, sassiness_delta,
             feedback_text)
        )
        return rows > 0
    
    async def get_cumulative_personality_deltas(self) -> Dict[str, float]:
        """
        Get cumulative personality adjustments.
        
        Returns:
            Dict with delta values for each trait
        """
        results = await self.query("EXEC GetCurrentPersonalityDeltas")
        if results:
            return {
                'curiosity': float(results[0]['curiosity_delta'] or 0.0),
                'touchiness': float(results[0]['touchiness_delta'] or 0.0),
                'vitality': float(results[0]['vitality_delta'] or 0.0),
                'friendliness': float(results[0]['friendliness_delta'] or 0.0),
                'courage': float(results[0]['courage_delta'] or 0.0),
                'sassiness': float(results[0]['sassiness_delta'] or 0.0),
            }
        return {trait: 0.0 for trait in ['curiosity', 'touchiness', 'vitality', 'friendliness', 'courage', 'sassiness']}
    
    async def get_conversation_by_id(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single conversation by ID (T140 - semantic search support).
        
        Args:
            conversation_id: Conversation ID from conversations table
            
        Returns:
            Conversation dict with text, response_text, timestamp, speaker_name, etc.
            None if not found
        """
        results = await self.query(
            """
            SELECT 
                c.conversation_id,
                c.text,
                c.response_text,
                c.timestamp,
                c.speaker_id,
                COALESCE(f.name, 'Unknown') AS speaker_name,
                DATEDIFF(day, c.timestamp, GETDATE()) AS days_ago
            FROM conversations c
            LEFT JOIN faces f ON c.speaker_id = f.face_id
            WHERE c.conversation_id = ?
            """,
            (conversation_id,)
        )
        return results[0] if results else None
    
    async def close(self):
        """Shutdown executor and cleanup."""
        self._executor.shutdown(wait=True)
        logger.info("SQL Server connector closed")


# ========== Convenience Functions ==========

async def initialize_database(connector: SQLServerConnector, schema_file: str = 'schema.sql') -> bool:
    """
    Initialize database schema from SQL file.
    
    Args:
        connector: SQLServerConnector instance
        schema_file: Path to schema.sql file
    
    Returns:
        True if successful
    """
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Split by GO statements and execute each batch
        # GO must be case-insensitive and on its own line
        import re
        batches = re.split(r'^\s*GO\s*$', schema_sql, flags=re.IGNORECASE | re.MULTILINE)
        batches = [batch.strip() for batch in batches if batch.strip()]
        
        logger.info(f"Found {len(batches)} batches to execute")
        
        def _execute_batch(batch_sql):
            """Execute a single batch synchronously."""
            with connector._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Split by semicolons for individual statements within a batch
                    statements = [s.strip() for s in batch_sql.split(';') if s.strip()]
                    for statement in statements:
                        if statement:
                            logger.debug(f"Executing: {statement[:100]}...")
                            cursor.execute(statement)
                    conn.commit()
                except Exception as e:
                    logger.error(f"Batch execution failed: {e}")
                    logger.error(f"Failed batch: {batch_sql[:200]}...")
                    raise
        
        loop = asyncio.get_event_loop()
        for i, batch in enumerate(batches, 1):
            logger.info(f"Executing batch {i}/{len(batches)}")
            await loop.run_in_executor(connector._executor, _execute_batch, batch)
        
        logger.info("Database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False
