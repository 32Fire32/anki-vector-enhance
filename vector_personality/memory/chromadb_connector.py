"""
ChromaDB connector for Vector Personality Project.
Replaces SQL Server with ChromaDB as the sole persistence layer.

Principle II: Persistent Memory Architecture
All long-term storage goes through this connector.

Collections:
- faces: Face records and metadata
- face_embeddings: Face embedding vectors for similarity matching
- conversations: Conversation records with text embeddings
- objects: Detected objects with metadata
- rooms: Room profiles with behavior adjustments
- personality_learned: Personality trait adjustments over time
"""

import logging
import json
import uuid
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return current datetime as ISO string."""
    return datetime.now().isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO datetime string back to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None


class ChromaDBConnector:
    """
    ChromaDB connector providing the same async interface as the former
    SQLServerConnector. Uses ChromaDB collections to store faces,
    conversations, objects, rooms, and personality adjustments.

    All public methods are async (using ThreadPoolExecutor for blocking
    ChromaDB calls) so callers don't need to change.
    """

    def __init__(
        self,
        persist_directory: str = "./vector_memory_chroma",
        timeout: int = 30,
    ):
        """
        Initialize ChromaDB persistent client and all collections.

        Args:
            persist_directory: Path to ChromaDB storage directory
            timeout: Not used (kept for interface compat)
        """
        self.persist_directory = persist_directory
        self._executor = ThreadPoolExecutor(max_workers=5)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # ── Collections ──────────────────────────────────────────
        self.faces = self.client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"},
        )
        self.face_embeddings_col = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        self.conversations = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )
        self.objects = self.client.get_or_create_collection(
            name="objects",
            metadata={"hnsw:space": "cosine"},
        )
        self.rooms = self.client.get_or_create_collection(
            name="rooms",
            metadata={"hnsw:space": "cosine"},
        )
        self.personality_learned = self.client.get_or_create_collection(
            name="personality_learned",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB connector initialized: {persist_directory} "
            f"(faces={self.faces.count()}, conversations={self.conversations.count()}, "
            f"objects={self.objects.count()})"
        )

    # ── helpers ────────────────────────────────────────────────

    def _run(self, fn, *args, **kwargs):
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @staticmethod
    def _dummy_embedding(dim: int = 3) -> List[float]:
        """Return a zero-vector used as a placeholder embedding.

        ChromaDB requires an embedding for every document. When we don't
        have a real embedding (e.g. structured metadata records), we
        store a small dummy vector so that the record can still be
        retrieved by ID or metadata filter.
        """
        return [0.0] * dim

    # ── Connection test ───────────────────────────────────────

    async def test_connection(self) -> bool:
        """Test ChromaDB connectivity."""
        try:
            _ = self.faces.count()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    # ================================================================
    #  FACE OPERATIONS
    # ================================================================

    async def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face record by UUID."""
        try:
            result = self.faces.get(ids=[face_id], include=["metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0]
                return {"face_id": face_id, **meta}
            return None
        except Exception:
            return None

    async def get_face_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get face record by name."""
        try:
            result = self.faces.get(
                where={"name": name},
                include=["metadatas"],
            )
            if result["ids"]:
                return {"face_id": result["ids"][0], **result["metadatas"][0]}
            return None
        except Exception:
            return None

    async def create_face(
        self,
        name: Optional[str] = None,
        face_id: Optional[str] = None,
        sdk_face_id: Optional[int] = None,
    ) -> str:
        """Create new face record. Returns face_id."""
        fid = face_id or str(uuid.uuid4())
        now = _now_iso()
        meta: Dict[str, Any] = {
            "name": name or "",
            "first_seen": now,
            "last_seen": now,
            "total_interactions": 0,
            "last_mood_change": 0,
            "notes": "",
            "created_at": now,
            "updated_at": now,
        }
        if sdk_face_id is not None:
            meta["sdk_face_id"] = sdk_face_id
        self.faces.add(
            ids=[fid],
            embeddings=[self._dummy_embedding()],
            metadatas=[meta],
        )
        logger.info(f"Created face: {name or 'Unknown'} ({fid})")
        return fid

    async def update_face_interaction(self, face_id: str, mood_change: int = 0) -> bool:
        """Increment interaction count and update last_seen."""
        try:
            result = self.faces.get(ids=[face_id], include=["metadatas", "embeddings"])
            if not result["ids"]:
                return False
            meta = result["metadatas"][0]
            meta["total_interactions"] = int(meta.get("total_interactions", 0)) + 1
            meta["last_seen"] = _now_iso()
            meta["last_mood_change"] = mood_change
            meta["updated_at"] = _now_iso()
            self.faces.update(
                ids=[face_id],
                metadatas=[meta],
                embeddings=result["embeddings"],
            )
            return True
        except Exception as e:
            logger.error(f"update_face_interaction failed: {e}")
            return False

    async def get_face_history(self, face_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a face."""
        try:
            result = self.conversations.get(
                where={"speaker_id": face_id},
                include=["metadatas"],
            )
            rows = [
                {"conversation_id": cid, **meta}
                for cid, meta in zip(result["ids"], result["metadatas"])
            ]
            rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            return rows[:limit]
        except Exception:
            return []

    # ── Face Embedding Operations ─────────────────────────────

    async def get_latest_embedding_for_face(self, face_id: str) -> Optional[Tuple[bytes, int, datetime]]:
        """Return the latest embedding for a face as (bytes, dim, created_at) or None."""
        try:
            result = self.face_embeddings_col.get(
                where={"face_id": face_id},
                include=["metadatas", "embeddings"],
            )
            if not result["ids"]:
                return None
            # Find latest by created_at
            best_idx = 0
            best_ts = ""
            for i, meta in enumerate(result["metadatas"]):
                ts = meta.get("created_at", "")
                if ts > best_ts:
                    best_ts = ts
                    best_idx = i
            import numpy as np
            emb = np.array(result["embeddings"][best_idx], dtype=np.float32)
            dim = int(result["metadatas"][best_idx].get("vector_dim", emb.size))
            created = _parse_dt(best_ts) or datetime.now()
            return (emb.tobytes(), dim, created)
        except Exception as e:
            logger.warning(f"get_latest_embedding_for_face failed: {e}")
            return None

    async def add_face_embedding(self, face_id: str, embedding: bytes, vector_dim: int) -> bool:
        """Store raw embedding bytes for a face with deduplication."""
        import numpy as np
        import os

        try:
            dup_sim = float(os.getenv("FACE_EMBEDDING_DUPLICATE_SIMILARITY", "0.999"))
            min_secs = float(os.getenv("FACE_EMBEDDING_MIN_SECONDS", "5"))

            latest = await self.get_latest_embedding_for_face(face_id)
            if latest is not None:
                latest_bytes, latest_dim, latest_created = latest
                if latest_bytes == embedding:
                    logger.info(f"🔇 Skipping identical embedding for face {face_id[:8]}...")
                    return False

                try:
                    arr_new = np.frombuffer(embedding, dtype=np.float32)
                    arr_old = np.frombuffer(latest_bytes, dtype=np.float32)
                except Exception:
                    arr_new = arr_old = None

                if arr_new is not None and arr_old is not None and arr_new.size == arr_old.size:
                    arr_new_norm = arr_new / (np.linalg.norm(arr_new) + 1e-10)
                    arr_old_norm = arr_old / (np.linalg.norm(arr_old) + 1e-10)
                    score = float(np.dot(arr_new_norm, arr_old_norm))
                    if score >= dup_sim:
                        if latest_created and (datetime.now() - latest_created).total_seconds() < min_secs:
                            logger.info(f"🔇 Skipping near-duplicate embedding for face {face_id[:8]} (score={score:.4f})")
                            return False

            # Store as float32 list for ChromaDB and raw bytes in metadata
            arr = np.frombuffer(embedding, dtype=np.float32)
            emb_id = str(uuid.uuid4())
            self.face_embeddings_col.add(
                ids=[emb_id],
                embeddings=[arr.tolist()],
                metadatas=[{
                    "face_id": face_id,
                    "vector_dim": vector_dim,
                    "created_at": _now_iso(),
                }],
            )
            return True
        except Exception as e:
            logger.warning(f"add_face_embedding failed: {e}")
            return False

    async def get_latest_embeddings(self) -> List[Tuple[str, bytes, int]]:
        """Return the latest embedding for each face as (face_id, bytes, dim)."""
        import numpy as np
        try:
            result = self.face_embeddings_col.get(include=["metadatas", "embeddings"])
            if not result["ids"]:
                return []
            # Group by face_id, keep latest
            latest_map: Dict[str, int] = {}
            for i, meta in enumerate(result["metadatas"]):
                fid = meta.get("face_id", "")
                ts = meta.get("created_at", "")
                if fid not in latest_map or ts > result["metadatas"][latest_map[fid]].get("created_at", ""):
                    latest_map[fid] = i
            out = []
            for fid, idx in latest_map.items():
                emb = np.array(result["embeddings"][idx], dtype=np.float32)
                dim = int(result["metadatas"][idx].get("vector_dim", emb.size))
                out.append((fid, emb.tobytes(), dim))
            return out
        except Exception:
            return []

    async def find_similar_faces(self, embedding: Any, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Find similar faces via cosine similarity against stored face embeddings."""
        import numpy as np
        encoded = np.asarray(embedding, dtype=np.float32)
        all_embeddings = await self.get_latest_embeddings()
        matches = []
        for face_id, emb_bytes, dim in all_embeddings:
            try:
                arr = np.frombuffer(emb_bytes, dtype=np.float32)
                if arr.size == 0:
                    continue
                arr_norm = arr / (np.linalg.norm(arr) + 1e-10)
                q_norm = encoded / (np.linalg.norm(encoded) + 1e-10)
                score = float(np.dot(arr_norm, q_norm))
                if score >= min_score:
                    matches.append({"face_id": face_id, "score": score})
            except Exception:
                continue
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    async def merge_faces(self, canonical_face_id: str, merged_face_id: str) -> bool:
        """Merge `merged_face_id` into `canonical_face_id`."""
        try:
            # Re-point conversations
            convos = self.conversations.get(
                where={"speaker_id": merged_face_id},
                include=["metadatas", "embeddings"],
            )
            for i, cid in enumerate(convos["ids"]):
                meta = convos["metadatas"][i]
                meta["speaker_id"] = canonical_face_id
                self.conversations.update(
                    ids=[cid], metadatas=[meta], embeddings=[convos["embeddings"][i]]
                )

            # Consolidate faces metadata
            canon = await self.get_face_by_id(canonical_face_id)
            merged = await self.get_face_by_id(merged_face_id)
            if canon and merged:
                canon_interactions = int(canon.get("total_interactions", 0))
                merged_interactions = int(merged.get("total_interactions", 0))
                canon_last = canon.get("last_seen", "")
                merged_last = merged.get("last_seen", "")
                new_last = max(canon_last, merged_last)
                canon["total_interactions"] = canon_interactions + merged_interactions
                canon["last_seen"] = new_last
                canon["updated_at"] = _now_iso()
                # Remove non-metadata keys before updating
                canon.pop("face_id", None)
                self.faces.update(
                    ids=[canonical_face_id],
                    metadatas=[canon],
                    embeddings=[self._dummy_embedding()],
                )

            # Mark merged face
            if merged:
                merged["merged_to"] = canonical_face_id
                merged["merged_at"] = _now_iso()
                merged.pop("face_id", None)
                self.faces.update(
                    ids=[merged_face_id],
                    metadatas=[merged],
                    embeddings=[self._dummy_embedding()],
                )
            return True
        except Exception as e:
            logger.error(f"merge_faces failed: {e}")
            return False

    # ================================================================
    #  ROOM OPERATIONS
    # ================================================================

    async def get_or_create_room(self, room_name: str) -> Dict[str, Any]:
        """Get room by name or create if not exists."""
        try:
            result = self.rooms.get(
                where={"room_name": room_name}, include=["metadatas"]
            )
            if result["ids"]:
                meta = result["metadatas"][0]
                meta["room_id"] = result["ids"][0]
                # Update visit
                meta["last_visited"] = _now_iso()
                meta["visit_count"] = int(meta.get("visit_count", 0)) + 1
                self.rooms.update(
                    ids=[result["ids"][0]],
                    metadatas=[meta],
                    embeddings=[self._dummy_embedding()],
                )
                return meta
            # Create new
            rid = str(uuid.uuid4())
            now = _now_iso()
            meta = {
                "room_name": room_name,
                "typical_objects": "[]",
                "context_behavior_adjustments": "{}",
                "first_identified": now,
                "last_visited": now,
                "visit_count": 1,
                "created_at": now,
                "updated_at": now,
            }
            self.rooms.add(ids=[rid], embeddings=[self._dummy_embedding()], metadatas=[meta])
            meta["room_id"] = rid
            return meta
        except Exception as e:
            logger.error(f"get_or_create_room failed: {e}")
            return {"room_id": "", "room_name": room_name}

    async def update_room_objects(self, room_id: str, typical_objects: List[str]) -> bool:
        """Update typical objects list for a room."""
        try:
            result = self.rooms.get(ids=[room_id], include=["metadatas"])
            if not result["ids"]:
                return False
            meta = result["metadatas"][0]
            meta["typical_objects"] = json.dumps(typical_objects)
            meta["updated_at"] = _now_iso()
            self.rooms.update(ids=[room_id], metadatas=[meta], embeddings=[self._dummy_embedding()])
            return True
        except Exception:
            return False

    # ================================================================
    #  OBJECT DETECTION
    # ================================================================

    async def store_object_detection(
        self,
        object_type: str,
        confidence: float,
        room_id: Optional[str] = None,
        location_description: Optional[str] = None,
    ) -> str:
        """Store detected object (upsert by type+room)."""
        try:
            # Check if object already exists in this room
            where_filter: Dict[str, Any] = {"object_type": object_type}
            if room_id:
                where_filter = {"$and": [{"object_type": object_type}, {"room_id": room_id}]}

            result = self.objects.get(where=where_filter, include=["metadatas"])
            if result["ids"]:
                oid = result["ids"][0]
                meta = result["metadatas"][0]
                meta["last_detected"] = _now_iso()
                meta["detection_count"] = int(meta.get("detection_count", 1)) + 1
                meta["confidence"] = confidence
                meta["updated_at"] = _now_iso()
                self.objects.update(ids=[oid], metadatas=[meta], embeddings=[self._dummy_embedding()])
                return oid
            else:
                oid = str(uuid.uuid4())
                now = _now_iso()
                meta = {
                    "object_type": object_type,
                    "room_id": room_id or "",
                    "confidence": confidence,
                    "location_description": location_description or "",
                    "first_detected": now,
                    "last_detected": now,
                    "detection_count": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                self.objects.add(ids=[oid], embeddings=[self._dummy_embedding()], metadatas=[meta])
                return oid
        except Exception as e:
            logger.error(f"store_object_detection failed: {e}")
            return ""

    # ================================================================
    #  VISUAL OBJECT MEMORY (VLM-observed objects with attributes)
    # ================================================================

    async def store_visual_memory(
        self,
        object_name: str,
        description: str,
        user_label: str = "",
    ) -> str:
        """
        Upsert a visually-observed object with its full VLM description.

        ``object_name`` is the normalised name (lowercase, e.g. "penna").
        ``description`` is the raw VLM object line, e.g. "penna | rossa | Bic, sottile".
        ``user_label`` is an optional human-given name (e.g. "stampante").
        """
        try:
            result = self.objects.get(
                where={"visual_memory_name": object_name},
                include=["metadatas"],
            )
            now = _now_iso()
            if result["ids"]:
                oid = result["ids"][0]
                meta = result["metadatas"][0]
                meta["description"] = description
                meta["last_seen"] = now
                meta["seen_count"] = int(meta.get("seen_count", 1)) + 1
                meta["updated_at"] = now
                if user_label:
                    meta["user_label"] = user_label
                self.objects.update(ids=[oid], metadatas=[meta], embeddings=[self._dummy_embedding()])
                return oid
            else:
                oid = str(uuid.uuid4())
                meta = {
                    "visual_memory_name": object_name,
                    "description": description,
                    "first_seen": now,
                    "last_seen": now,
                    "seen_count": 1,
                    "user_label": user_label or "",
                    # Keep compatibility with existing object_type field
                    "object_type": f"visual:{object_name}",
                    "created_at": now,
                    "updated_at": now,
                }
                self.objects.add(ids=[oid], embeddings=[self._dummy_embedding()], metadatas=[meta])
                logger.debug(f"Visual memory stored: {object_name} → {description}")
                return oid
        except Exception as e:
            logger.error(f"store_visual_memory failed: {e}")
            return ""

    async def get_visual_memory(self) -> list:
        """
        Return all visually-memorised objects (those with ``visual_memory_name``).

        Returns list of dicts: {object_name, description, first_seen, last_seen, seen_count}
        """
        try:
            result = self.objects.get(
                where={"object_type": {"$contains": "visual:"}},
                include=["metadatas"],
            )
            entries = []
            for meta in result.get("metadatas", []):
                entries.append({
                    "object_name": meta.get("visual_memory_name", ""),
                    "description": meta.get("description", ""),
                    "first_seen": meta.get("first_seen", ""),
                    "last_seen": meta.get("last_seen", ""),
                    "seen_count": int(meta.get("seen_count", 1)),
                    "user_label": meta.get("user_label", ""),
                })
            return entries
        except Exception as e:
            logger.error(f"get_visual_memory failed: {e}")
            return []

    # ================================================================
    #  CONVERSATION STORAGE
    # ================================================================

    async def store_conversation(
        self,
        speaker_id: str,
        text: str,
        room_id: Optional[str] = None,
        emotional_context: int = 50,
        response_text: Optional[str] = None,
        response_type: Optional[str] = None,
        vector_db: Optional[Any] = None,
        embedding_gen: Optional[Any] = None,
    ) -> str:
        """Store conversation turn with optional embedding."""
        cid = str(uuid.uuid4())
        now = _now_iso()
        meta: Dict[str, Any] = {
            "speaker_id": speaker_id,
            "timestamp": now,
            "timestamp_epoch": datetime.now().timestamp(),  # numeric for ChromaDB $gte queries
            "text": text,
            "room_id": room_id or "",
            "emotional_context": emotional_context,
            "response_text": response_text or "",
            "response_type": response_type or "",
            "created_at": now,
        }

        # Generate text embedding for semantic search if components available
        embedding = None
        if vector_db and embedding_gen and text:
            try:
                full_text = f"User: {text}"
                if response_text:
                    full_text += f"\nVector: {response_text}"
                embedding = await embedding_gen.generate_embedding(full_text)
            except Exception as e:
                logger.warning(f"Failed to generate conversation embedding: {e}")

        # Use generated embedding or dummy
        emb = embedding if embedding else self._dummy_embedding()
        self.conversations.add(ids=[cid], embeddings=[emb], metadatas=[meta])

        # Also store in the semantic search collection if vector_db provided
        if vector_db and embedding and embedding_gen:
            try:
                vector_db.add_embedding(
                    conversation_id=cid,
                    embedding=embedding,
                    metadata={
                        "timestamp": now,
                        "speaker_id": speaker_id,
                        "has_response": response_text is not None,
                        "text_preview": text[:100],
                    },
                )
                logger.info(f"🔢 Embedding stored for conversation {cid[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to store embedding in vector_db: {e}")

        return cid

    # ── Context Queries ───────────────────────────────────────

    async def get_recent_conversations(self, hours: int = 72, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations within the last N hours."""
        from datetime import timedelta
        cutoff_dt = datetime.now() - timedelta(hours=hours)
        try:
            # ChromaDB $gte/$lte require numeric values; ISO strings don't work.
            # Fetch all and filter in Python — collection stays small for a personal robot.
            result = self.conversations.get(
                include=["metadatas"],
            )
            rows = []
            for cid, meta in zip(result["ids"], result["metadatas"]):
                # Filter by cutoff in Python (ChromaDB $gte requires numbers, not ISO strings)
                ts_raw = meta.get("timestamp")
                ts_dt = _parse_dt(ts_raw) if isinstance(ts_raw, str) else ts_raw
                if ts_dt is not None and ts_dt < cutoff_dt:
                    continue
                row = {"conversation_id": cid, **meta}
                # Resolve speaker name
                speaker_id = meta.get("speaker_id", "")
                if speaker_id:
                    face = await self.get_face_by_id(speaker_id)
                    row["speaker_name"] = face.get("name", "Unknown") if face else "Unknown"
                else:
                    row["speaker_name"] = "Unknown"
                # Parse timestamp to datetime for downstream code
                row["timestamp"] = ts_dt if ts_dt is not None else ts_raw
                rows.append(row)
            rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
            return rows[:limit]
        except Exception as e:
            logger.error(f"get_recent_conversations failed: {e}")
            return []

    async def get_known_faces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get known faces ordered by last seen."""
        try:
            result = self.faces.get(include=["metadatas"])
            rows = [
                {"face_id": fid, **meta}
                for fid, meta in zip(result["ids"], result["metadatas"])
                if not meta.get("merged_to")  # Exclude merged faces
            ]
            rows.sort(key=lambda r: r.get("last_seen", ""), reverse=True)
            return rows[:limit]
        except Exception:
            return []

    async def get_recent_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently detected objects."""
        try:
            result = self.objects.get(include=["metadatas"])
            rows = [
                {"object_id": oid, **meta}
                for oid, meta in zip(result["ids"], result["metadatas"])
            ]
            rows.sort(key=lambda r: r.get("last_detected", ""), reverse=True)
            return rows[:limit]
        except Exception:
            return []

    async def search_old_conversations(
        self,
        keywords: List[str],
        days_back: int = 90,
        limit: int = 15,
        exclude_recent_hours: int = 72,
    ) -> List[Dict[str, Any]]:
        """Search conversations by keywords with relevance ranking."""
        from datetime import timedelta
        keywords = [k for k in (keywords or []) if k and k.strip()]
        if not keywords:
            return []

        cutoff_old_ts = (datetime.now() - timedelta(days=days_back)).timestamp()
        try:
            # Get all conversations within range (use numeric epoch for ChromaDB $gte)
            result = self.conversations.get(
                where={"timestamp_epoch": {"$gte": cutoff_old_ts}},
                include=["metadatas"],
            )
            if not result["ids"]:
                # Fallback: no timestamp_epoch metadata — fetch all and filter in Python
                result = self.conversations.get(include=["metadatas"])
                if not result["ids"]:
                    return []

            # Apply exclude_recent filter
            if exclude_recent_hours and int(exclude_recent_hours) > 0:
                recent_cutoff_ts = (datetime.now() - timedelta(hours=exclude_recent_hours)).timestamp()
            else:
                recent_cutoff_ts = None

            matches = []
            for cid, meta in zip(result["ids"], result["metadatas"]):
                ts = meta.get("timestamp", "")
                # Filter out entries newer than cutoff_old
                ts_epoch = meta.get("timestamp_epoch")
                if ts_epoch is not None:
                    if float(ts_epoch) < cutoff_old_ts:
                        continue
                    if recent_cutoff_ts and float(ts_epoch) >= recent_cutoff_ts:
                        continue
                else:
                    # Fallback: parse ISO timestamp string
                    dt_check = _parse_dt(ts)
                    if dt_check:
                        ep = dt_check.timestamp()
                        if ep < cutoff_old_ts:
                            continue
                        if recent_cutoff_ts and ep >= recent_cutoff_ts:
                            continue

                text = (meta.get("text") or "").lower()
                response = (meta.get("response_text") or "").lower()

                # Skip hallucination responses
                skip_phrases = [
                    "non mi ricordo", "non lo so", "non l'ho mai", "non so chi",
                    "non ho dati", "non ho informazioni", "non ho osservato"
                ]
                if any(phrase in response for phrase in skip_phrases):
                    continue

                # Score relevance
                score = 0
                matched = False
                for kw in keywords[:6]:
                    kw_lower = kw.lower()
                    if kw_lower in response:
                        score += 10
                        matched = True
                    if kw_lower in text:
                        score += 5
                        matched = True
                if "ricord" in text:
                    score -= 10

                if matched:
                    row = {"conversation_id": cid, **meta}
                    row["relevance_score"] = score
                    # Compute days_ago
                    dt = _parse_dt(ts)
                    if dt:
                        row["days_ago"] = (datetime.now() - dt).days
                        row["timestamp"] = dt
                    # Resolve person name
                    speaker_id = meta.get("speaker_id", "")
                    if speaker_id:
                        face = await self.get_face_by_id(speaker_id)
                        row["person"] = face.get("name", "Unknown") if face else "Unknown"
                    else:
                        row["person"] = "Unknown"
                    matches.append(row)

            matches.sort(key=lambda x: (-x.get("relevance_score", 0), x.get("timestamp", "")))
            return matches[:limit]
        except Exception as e:
            logger.error(f"search_old_conversations failed: {e}")
            return []

    # ================================================================
    #  PERSONALITY LEARNING
    # ================================================================

    async def save_personality_adjustment(
        self,
        curiosity_delta: float = 0.0,
        touchiness_delta: float = 0.0,
        vitality_delta: float = 0.0,
        friendliness_delta: float = 0.0,
        courage_delta: float = 0.0,
        sassiness_delta: float = 0.0,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """Save personality trait adjustment."""
        try:
            pid = str(uuid.uuid4())
            now = _now_iso()
            meta = {
                "timestamp": now,
                "curiosity_delta": curiosity_delta,
                "touchiness_delta": touchiness_delta,
                "vitality_delta": vitality_delta,
                "friendliness_delta": friendliness_delta,
                "courage_delta": courage_delta,
                "sassiness_delta": sassiness_delta,
                "feedback_text": feedback_text or "",
                "created_at": now,
            }
            self.personality_learned.add(
                ids=[pid], embeddings=[self._dummy_embedding()], metadatas=[meta]
            )
            return True
        except Exception as e:
            logger.error(f"save_personality_adjustment failed: {e}")
            return False

    async def get_cumulative_personality_deltas(self) -> Dict[str, float]:
        """Get cumulative personality adjustments."""
        traits = ["curiosity", "touchiness", "vitality", "friendliness", "courage", "sassiness"]
        try:
            result = self.personality_learned.get(include=["metadatas"])
            totals: Dict[str, float] = {t: 0.0 for t in traits}
            for meta in result["metadatas"]:
                for t in traits:
                    totals[t] += float(meta.get(f"{t}_delta", 0.0))
            return totals
        except Exception:
            return {t: 0.0 for t in traits}

    async def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a single conversation by ID."""
        try:
            # Ensure conversation_id is a string
            conv_id_str = str(conversation_id)
            result = self.conversations.get(ids=[conv_id_str], include=["metadatas"])
            if not result["ids"]:
                return None
            meta = result["metadatas"][0]
            row = {"conversation_id": conv_id_str, **meta}
            # Resolve speaker name
            speaker_id = meta.get("speaker_id", "")
            if speaker_id:
                face = await self.get_face_by_id(speaker_id)
                row["speaker_name"] = face.get("name", "Unknown") if face else "Unknown"
            else:
                row["speaker_name"] = "Unknown"
            # Parse timestamp
            ts = meta.get("timestamp")
            if isinstance(ts, str):
                row["timestamp"] = _parse_dt(ts) or ts
            # Compute days_ago
            dt = _parse_dt(ts) if isinstance(ts, str) else ts
            if isinstance(dt, datetime):
                row["days_ago"] = (datetime.now() - dt).days
            return row
        except Exception as e:
            logger.error(f"get_conversation_by_id failed: {e}")
            return None

    # ── Generic query/execute for backward compat ─────────────

    async def query(self, sql_or_key: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Backward-compatible query interface.

        Supported patterns:
        - "SELECT sdk_face_id FROM faces WHERE face_id = ?"
        - "SELECT face_id FROM faces WHERE name = ?"
        """
        sql = str(sql_or_key)
        p = params[0] if isinstance(params, (tuple, list)) else params

        # sdk_face_id lookup by face_id
        if "sdk_face_id" in sql and "face_id" in sql:
            face = await self.get_face_by_id(str(p))
            if face:
                return [{"sdk_face_id": face.get("sdk_face_id")}]
            return []

        # face_id lookup by name (used in _handle_user_speech)
        if "face_id" in sql and "name" in sql and p is not None:
            try:
                result = self.faces.get(
                    where={"name": {"$eq": str(p)}},
                    include=["metadatas"],
                )
                if result["ids"]:
                    return [{"face_id": result["ids"][0]}]
            except Exception as e:
                logger.debug(f"Face name query failed: {e}")
            return []

        logger.debug(f"Ignored unrecognised query pattern: {sql[:60]}")
        return []

    async def execute(self, sql_or_key: str, params: Optional[Any] = None) -> int:
        """Backward-compatible execute interface.

        Handles the specific UPDATE pattern from face_detection.py.
        """
        if "UPDATE faces SET sdk_face_id" in str(sql_or_key):
            p = params if isinstance(params, (tuple, list)) else (params,)
            sdk_id = p[0]
            face_id = p[1]
            face = await self.get_face_by_id(str(face_id))
            if face:
                face.pop("face_id", None)
                face["sdk_face_id"] = sdk_id
                face["updated_at"] = _now_iso()
                self.faces.update(
                    ids=[str(face_id)],
                    metadatas=[face],
                    embeddings=[self._dummy_embedding()],
                )
                return 1
            return 0
        logger.debug(f"Ignored legacy SQL execute pattern: {sql_or_key[:60]}...")
        return 0

    async def close(self):
        """Shutdown executor and cleanup."""
        self._executor.shutdown(wait=True)
        logger.info("ChromaDB connector closed")


# ── Convenience functions ─────────────────────────────────────

async def initialize_database(connector: 'ChromaDBConnector', schema_file: str = 'schema.sql') -> bool:
    """Initialize database (no-op for ChromaDB - collections are auto-created)."""
    logger.info("ChromaDB collections are auto-created on first use - no schema init needed")
    return True
