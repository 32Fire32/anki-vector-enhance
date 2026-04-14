"""User Registry
================
Maps multiple channel identities (face UUID, Telegram user ID, web session ID)
to a single canonical user profile — so memories, entity profiles, and
conversation history are unified regardless of how the user talks to Vector.

Storage: ``vector_users.json`` in the project root (next to api.env).
Simple JSON — no extra dependencies.

Canonical ID format: ``user_<id>`` (e.g. ``user_1`` for the first user).

Usage::

    registry = UserRegistry()

    # Resolve any raw speaker_id to a canonical one
    canonical = registry.resolve("9deb0e64-...")     # face UUID
    canonical = registry.resolve("6067320095")       # Telegram ID (string)
    canonical = registry.resolve("some-session-uuid") # web session

    # Get display name for a canonical ID
    name = registry.get_name("user_1")  # → "Nicola"

    # CRUD (for dashboard API)
    user = registry.get_user(1)
    all_users = registry.get_all_users()
    user = registry.create_user("Marco")
    registry.update_user(1, face_uuids=["..."], telegram_ids=[123])
    registry.delete_user(2)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "vector_users.json"


class UserRegistry:
    """Unified user identity registry across voice, Telegram, and web channels."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = Path(path or os.getenv("USER_REGISTRY_PATH", str(_DEFAULT_PATH)))
        self._data: Dict[str, Any] = {"users": [], "next_id": 1}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"[UserRegistry] Loaded {len(self._data.get('users', []))} user(s) from {self._path}")
            except Exception as e:
                logger.warning(f"[UserRegistry] Failed to load {self._path}: {e}")
                self._data = {"users": [], "next_id": 1}
        else:
            logger.info(f"[UserRegistry] No registry file found — starting empty ({self._path})")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[UserRegistry] Failed to save: {e}")

    # ------------------------------------------------------------------
    # Resolution — the core lookup
    # ------------------------------------------------------------------

    def resolve(self, raw_id: str) -> str:
        """Map any raw channel identity to the canonical user ID.

        Returns the canonical_id (e.g. ``"user_1"``) if a match is found,
        or the original ``raw_id`` unchanged if no match exists.
        """
        raw_id = str(raw_id).strip()
        for user in self._data.get("users", []):
            canonical = user["canonical_id"]
            if raw_id == canonical:
                return canonical
            if raw_id in [str(x) for x in user.get("face_uuids", [])]:
                return canonical
            if raw_id in [str(x) for x in user.get("telegram_ids", [])]:
                return canonical
            if raw_id in user.get("web_session_ids", []):
                return canonical
        # No match
        return raw_id

    def get_name(self, canonical_id: str) -> Optional[str]:
        """Return the display name for a canonical ID, or None if not found."""
        for user in self._data.get("users", []):
            if user["canonical_id"] == canonical_id:
                return user.get("name")
        return None

    def get_canonical_id(self, user_id: int) -> Optional[str]:
        for user in self._data.get("users", []):
            if user["user_id"] == user_id:
                return user["canonical_id"]
        return None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_all_users(self) -> List[Dict[str, Any]]:
        return list(self._data.get("users", []))

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        for user in self._data.get("users", []):
            if user["user_id"] == user_id:
                return dict(user)
        return None

    def create_user(
        self,
        name: str,
        face_uuids: Optional[List[str]] = None,
        telegram_ids: Optional[List[int]] = None,
        web_session_ids: Optional[List[str]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Create a new user and return the new record."""
        uid = self._data.get("next_id", 1)
        canonical_id = f"user_{uid}"
        new_user: Dict[str, Any] = {
            "user_id": uid,
            "canonical_id": canonical_id,
            "name": name.strip(),
            "face_uuids": face_uuids or [],
            "telegram_ids": [int(t) for t in (telegram_ids or [])],
            "web_session_ids": web_session_ids or [],
            "notes": notes,
            "created_at": datetime.now().isoformat(),
        }
        self._data.setdefault("users", []).append(new_user)
        self._data["next_id"] = uid + 1
        self._save()
        logger.info(f"[UserRegistry] Created user {uid} ({name})")
        return dict(new_user)

    def update_user(
        self,
        user_id: int,
        name: Optional[str] = None,
        face_uuids: Optional[List[str]] = None,
        telegram_ids: Optional[List[int]] = None,
        web_session_ids: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update fields on an existing user. Pass None to leave unchanged."""
        for user in self._data.get("users", []):
            if user["user_id"] == user_id:
                if name is not None:
                    user["name"] = name.strip()
                if face_uuids is not None:
                    user["face_uuids"] = list(face_uuids)
                if telegram_ids is not None:
                    user["telegram_ids"] = [int(t) for t in telegram_ids]
                if web_session_ids is not None:
                    user["web_session_ids"] = list(web_session_ids)
                if notes is not None:
                    user["notes"] = notes
                user["updated_at"] = datetime.now().isoformat()
                self._save()
                logger.info(f"[UserRegistry] Updated user {user_id}")
                return dict(user)
        logger.warning(f"[UserRegistry] update_user: user {user_id} not found")
        return None

    def delete_user(self, user_id: int) -> bool:
        users = self._data.get("users", [])
        before = len(users)
        self._data["users"] = [u for u in users if u["user_id"] != user_id]
        if len(self._data["users"]) < before:
            self._save()
            logger.info(f"[UserRegistry] Deleted user {user_id}")
            return True
        logger.warning(f"[UserRegistry] delete_user: user {user_id} not found")
        return False

    def get_known_faces(self) -> List[Dict[str, str]]:
        """Return list of {face_uuid, user_name} for all registered face UUIDs."""
        result = []
        for user in self._data.get("users", []):
            for uuid in user.get("face_uuids", []):
                result.append({"face_uuid": uuid, "user_name": user["name"], "user_id": user["user_id"]})
        return result
