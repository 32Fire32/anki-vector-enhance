"""
Scene Descriptor Module

Uses a local Vision-Language Model (VLM) via Ollama to describe
what Vector's camera sees. Runs periodically in the background,
detecting meaningful scene changes and building a visual memory of
objects — including attribute changes (e.g., black pen → red pen).

Phase 3 - Local Multimodal Vision
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SceneDescriptor:
    """
    Periodic scene description + visual object memory using a local VLM.

    Each scan:
      1. Asks the VLM to list notable objects with their attributes.
      2. Compares against remembered objects to detect attribute changes.
      3. Updates visual memory and emits natural Italian comments on changes.

    Visual memory is kept in-memory (dict) and optionally persisted to
    ChromaDB via the injected db_connector.
    """

    # --- VLM prompts ---

    # Extract notable objects with attributes from the current frame
    # NOTE: Keep prompt SHORT — llava:7b defaults to "NONE" with complex/rule-heavy prompts
    OBJECTS_PROMPT = (
        "List the objects you see in this image. "
        "Do your best even if the image is blurry or dark. "
        "For each object, use this format: name | color | detail\n"
        "Example: pen | red | ballpoint\n"
        "List up to 8 objects, most prominent first."
    )

    # Objects that cannot exist indoors — filter hallucinations of outdoor elements
    # Covers both English (llava output) and Italian (legacy/gemma output)
    _OUTDOOR_KEYWORDS = frozenset({
        # English
        "tree", "trees", "sky", "grass", "lawn", "house", "fence", "mountain", "mountains",
        "hill", "hills", "road", "sidewalk", "cloud", "clouds", "sun", "moon", "stars",
        "forest", "field", "garden", "yard", "parking", "river", "lake", "sea", "beach",
        # Italian (kept for fallback)
        "albero", "alberi", "cielo", "prato", "erba", "casa", "case", "recinzione",
        "recinto", "montagna", "montagne", "collina", "colline", "strada", "marciapiede",
        "nuvola", "nuvole", "sole", "luna", "stelle", "foresta", "bosco", "campo",
        "giardino", "cortile", "parcheggio", "fiume", "lago", "mare", "spiaggia",
    })

    # Compare two TEXT object lists using the chat model (no image — avoids VLM hallucinations)
    # The chat model handles Italian synonyms: televisore=monitor=schermo, auto=automobile, etc.
    TEXT_DIFF_PROMPT = (
        "Confronta questi due elenchi di oggetti visti da una telecamera a 8 secondi di distanza.\n\n"
        "LISTA PRECEDENTE:\n{prev_list}\n\n"
        "LISTA ATTUALE:\n{curr_list}\n\n"
        "REGOLE:\n"
        "- Sinonimi = stesso oggetto: 'auto'='automobile', 'televisore'='monitor'='schermo'='tv', "
        "'divano'='sofà', 'computer'='laptop'='pc', 'persona'='uomo'='donna', "
        "'macchina'='vettura'='auto', 'tavolo'='tavolino'='tavolo da pranzo'\n"
        "- Una variante del nome NON è un cambiamento\n"
        "- Elementi architetturali fissi (finestra, porta, muro, pavimento, soffitto, parete) "
        "NON sono mai cambiamenti — fanno parte della stanza\n"
        "- Le liste sono campioni PARZIALI: un oggetto assente dalla lista non significa che sia sparito\n"
        "- Segnala SOLO se un oggetto CHIARAMENTE NUOVO è apparso (non era presente prima in alcuna forma)\n"
        "- Se hai il MINIMO dubbio, rispondi NESSUN_CAMBIAMENTO\n\n"
        "Se c'è un oggetto CERTAMENTE NUOVO: 1 frase italiana breve (es. 'È apparsa una tazza').\n"
        "Altrimenti rispondi ESATTAMENTE: NESSUN_CAMBIAMENTO"
    )

    # Generate the actual spoken comment given a raw change description
    COMMENT_PROMPT = (
        "Sei Vector, un piccolo robot. "
        "Hai appena notato questo cambiamento visivo: \"{change}\"\n"
        "Genera 1 frase breve in italiano per commentare — naturale e concreta, "
        "senza metafore o poesia. Puoi fare una domanda diretta tipo 'Cosa ci fa quella X lì?' "
        "oppure un'osservazione semplice. Non iniziare con 'Io'."
    )

    # First-impression comment when Vector sees the scene for the first time
    FIRST_IMPRESSION_PROMPT = (
        "Sei Vector, un piccolo robot curioso e amichevole. "
        "Ti sei appena svegliato e questi sono gli oggetti che vedi intorno a te:\n{objects_list}\n\n"
        "Genera 1 frase breve in italiano per descrivere cosa vedi, "
        "come se lo stessi scoprendo adesso — puoi fare un'osservazione curiosa o spiritosa. "
        "Non elencare tutti gli oggetti, commenta solo 1-2 cose che ti colpiscono. "
        "Non iniziare con 'Io'."
    )

    def __init__(
        self,
        ollama_client,
        vlm_model: str = "gemma3:12b",
        chat_model: Optional[str] = None,
        interval_seconds: float = 8.0,
        cooldown_after_speak: float = 12.0,
        db_connector=None,
    ):
        self.ollama = ollama_client
        self.vlm_model = vlm_model
        # Chat model for generating natural comments (defaults to same model)
        self.chat_model = chat_model or vlm_model
        self.interval = interval_seconds
        self.cooldown_after_speak = cooldown_after_speak
        self.db = db_connector

        # Last plain-text scene description (for context injection in chat)
        self.last_description: Optional[str] = None
        self.last_change_description: Optional[str] = None
        self.last_run_time: float = 0.0
        self.last_spoke_time: float = 0.0
        self._running = False

        # Visual object memory: key = normalised object name, value = dict of attributes
        # e.g. {"penna": {"description": "penna | nera | Bic", "first_seen": ..., "last_seen": ...}}
        self.visual_memory: Dict[str, Dict[str, Any]] = {}
        # Objects from the most recent completed scan — used for change comparison
        self.last_scan_objects: List[str] = []

        # Confirmation buffer: a detected change must survive 2 consecutive scans before being spoken
        # This eliminates single-scan VLM hallucinations.
        self._pending_change: Optional[str] = None      # raw change text from last scan
        self._pending_change_count: int = 0             # how many consecutive scans confirmed it

        # Stats
        self.total_scans = 0
        self.total_changes = 0
        self.last_proactive_speak_time: float = 0.0
        self.proactive_cooldown: float = 30.0  # min seconds between proactive comments

        logger.info(
            f"SceneDescriptor initialized: model={vlm_model}, "
            f"interval={interval_seconds}s, cooldown={cooldown_after_speak}s"
        )

    # ------------------------------------------------------------------ #
    #  VLM helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_name(raw: str) -> str:
        """Strip bullets, asterisks, numbers, and whitespace from object name."""
        import re
        # Remove leading bullet chars: *, -, digits, dots
        cleaned = re.sub(r'^[\s*\-·•\d.]+', '', raw)
        return cleaned.strip().lower()

    @staticmethod
    def _clean_object_line(line: str) -> str:
        """Remove leading bullets/markdown from an object line."""
        import re
        return re.sub(r'^[\s*\-·•\d.]+', '', line).strip()

    async def _extract_objects(self, image) -> List[str]:
        """
        Ask VLM to list notable objects with attributes from the current frame.
        Returns list of raw object strings like ['penna | rossa | Bic', ...].
        """        # Diagnostic: log image info
        try:
            w, h = image.size if hasattr(image, 'size') else ('?', '?')
            mode = getattr(image, 'mode', '?')
            logger.info(f"\U0001f441\ufe0f Frame info: {w}x{h}, mode={mode}")
        except Exception as e:
            logger.warning(f"\U0001f441\ufe0f Could not inspect frame: {e}")

        # Save the very first frame to disk for manual inspection
        if self.total_scans == 0:
            try:
                debug_path = 'debug_frame.jpg'
                image.save(debug_path, format='JPEG', quality=80)
                logger.info(f"\U0001f441\ufe0f Saved debug frame to {debug_path}")
            except Exception as e:
                logger.warning(f"\U0001f441\ufe0f Could not save debug frame: {e}")
        raw = await self.ollama.vision_completion(
            prompt=self.OBJECTS_PROMPT,
            image=image,
            model=self.vlm_model,
            temperature=0.3,  # Balanced: enough variety to avoid KV cache staleness
            max_tokens=300,   # llava is verbose with simple prompts
            timeout_seconds=30,
        )
        logger.info(f"👁️ VLM raw response ({len(raw)} chars): {raw[:120]!r}")
        # Only treat as empty if the ENTIRE response is just "NONE" (not a substring in freetext)
        stripped = raw.strip().upper()
        if not raw or stripped == "NONE" or stripped == "NESSUNO":
            return []
        # Parse lines, skip empty, clean markdown artifacts, filter outdoor hallucinations
        lines = []
        for l in raw.splitlines():
            cleaned = self._clean_object_line(l)
            if not cleaned or "|" not in cleaned:
                continue
            # Extract object name (first part before |) and check against outdoor keyword list
            name_part = self._normalize_name(cleaned.split("|")[0])
            if name_part in self._OUTDOOR_KEYWORDS:
                logger.debug(f"👁️ Filtered outdoor hallucination: {cleaned}")
                continue
            lines.append(cleaned)
        # If llava gave a freetext paragraph instead of the name|color|detail format,
        # extract any nouns it mentioned as a minimal fallback entry
        if not lines and "|" not in raw:
            logger.debug("👁️ VLM gave freetext — no structured objects extracted")
        return lines[:8]

    async def _compare_object_lists(
        self, prev_objects: List[str], curr_objects: List[str]
    ) -> Optional[str]:
        """
        Compare two text-only object lists using the chat model.
        No image involved — eliminates VLM hallucinations caused by
        non-deterministic naming (televisore vs monitor vs schermo).
        Returns a change description or None.
        """
        prev_list = "\n".join(f"- {o}" for o in prev_objects)
        curr_list = "\n".join(f"- {o}" for o in curr_objects)
        prompt = self.TEXT_DIFF_PROMPT.format(prev_list=prev_list, curr_list=curr_list)
        result = await self.ollama.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.chat_model,
            temperature=0.1,  # Near-deterministic: same input → same output
            max_tokens=60,
        )
        if not result or "NESSUN_CAMBIAMENTO" in result.upper():
            return None
        return result.strip()

    async def _generate_comment(self, change_description: str) -> str:
        """
        Use the chat LLM to turn a raw change description into a natural Italian sentence.
        """
        prompt = self.COMMENT_PROMPT.format(change=change_description)
        comment = await self.ollama.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.chat_model,
            temperature=0.4,  # Low enough to avoid poetic drift
            max_tokens=60,
        )
        return comment.strip() if comment else change_description

    async def _generate_first_impression(self, objects: List[str]) -> Optional[str]:
        """
        Generate a spoken 'first impression' comment for the very first scan.
        """
        objects_list = "\n".join(f"- {o}" for o in objects)
        prompt = self.FIRST_IMPRESSION_PROMPT.format(objects_list=objects_list)
        comment = await self.ollama.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.chat_model,
            temperature=0.7,
            max_tokens=80,
        )
        return comment.strip() if comment else None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def scan_and_update_memory(self, image) -> Optional[str]:
        """
        Full scan cycle:
          1. Extract current objects from image.
          2. Compare against visual memory.
          3. Update memory with current objects.
          4. Return a natural Italian comment if something changed, else None.
        """
        current_objects = await self._extract_objects(image)
        self.total_scans += 1

        if not current_objects:
            logger.info("👁️ No notable objects detected in frame")
            return None

        logger.debug(f"👁️ Detected objects: {current_objects}")

        # Compare current text list against previous scan's text list.
        # Text-only diff via chat model — no image, no VLM synonym confusion.
        change_comment = None
        if self.last_scan_objects:
            logger.debug(f"\U0001f441\ufe0f Comparing vs last scan: {self.last_scan_objects}")
            raw_change = await self._compare_object_lists(self.last_scan_objects, current_objects)
            if raw_change:
                logger.info(f"👁️ Visual memory CHANGE detected: {raw_change}")
                change_comment = await self._generate_comment(raw_change)
                self.last_change_description = change_comment
                self.total_changes += 1
            else:
                self._pending_change = None
                self._pending_change_count = 0
        elif self.total_scans == 1:
            # First scan ever — generate a "first impression" comment
            logger.info("👁️ First scan — generating first impression")
            change_comment = await self._generate_first_impression(current_objects)
            if change_comment:
                self.last_change_description = change_comment
                self.total_changes += 1

        # Update in-memory visual store
        now_str = datetime.now().isoformat()
        for obj_line in current_objects:
            parts = [p.strip() for p in obj_line.split("|")]
            name = self._normalize_name(parts[0]) if parts else "oggetto"
            if not name:
                continue
            if name in self.visual_memory:
                self.visual_memory[name]["description"] = obj_line
                self.visual_memory[name]["last_seen"] = now_str
                self.visual_memory[name]["seen_count"] = self.visual_memory[name].get("seen_count", 0) + 1
            else:
                self.visual_memory[name] = {
                    "description": obj_line,
                    "first_seen": now_str,
                    "last_seen": now_str,
                    "seen_count": 1,
                }
                logger.info(f"👁️ New object memorized: {obj_line}")

            # Persist to ChromaDB if db available
            if self.db:
                try:
                    await self.db.store_visual_memory(
                        object_name=name,
                        description=obj_line,
                    )
                except Exception as e:
                    logger.debug(f"👁️ Visual memory DB persist failed: {e}")

        # Save current scan's objects for the NEXT scan's comparison
        self.last_scan_objects = list(current_objects)

        # Update plain scene description from current scan
        self.last_description = "; ".join(current_objects)

        return change_comment

    def can_speak_proactively(self) -> bool:
        """Check if enough time has passed since the last proactive comment."""
        return (time.time() - self.last_proactive_speak_time) >= self.proactive_cooldown

    async def start_loop(self, get_frame_fn, on_change_fn=None):
        """
        Start the periodic scene-scanning background loop.

        Args:
            get_frame_fn: Sync callable that returns a PIL Image (or None)
            on_change_fn: Optional async callback(comment: str) called on change
        """
        self._running = True
        logger.info(f"👁️ Scene descriptor loop started (every {self.interval}s)")

        # Load persisted visual memory from DB on startup
        if self.db:
            try:
                stored = await self.db.get_visual_memory()
                for entry in stored:
                    name = entry.get("object_name", "oggetto")
                    self.visual_memory[name] = {
                        "description": entry.get("description", name),
                        "first_seen": entry.get("first_seen", ""),
                        "last_seen": entry.get("last_seen", ""),
                        "seen_count": entry.get("seen_count", 1),
                        "user_label": entry.get("user_label", ""),
                    }
                if self.visual_memory:
                    taught = sum(1 for v in self.visual_memory.values() if v.get("user_label"))
                    logger.info(
                        f"\U0001f441\ufe0f Loaded {len(self.visual_memory)} objects from visual memory "
                        f"({taught} user-taught)"
                    )
            except Exception as e:
                logger.debug(f"👁️ Could not load visual memory from DB: {e}")

        _null_frame_count = 0
        while self._running:
            try:
                # Respect cooldown after Vector spoke
                time_since_spoke = time.time() - self.last_spoke_time
                if time_since_spoke < self.cooldown_after_speak:
                    remaining = self.cooldown_after_speak - time_since_spoke
                    logger.debug(f"👁️ Scene scan deferred (cooldown {remaining:.0f}s)")
                    await asyncio.sleep(min(remaining, self.interval))
                    continue

                frame = get_frame_fn()
                if frame is None:
                    _null_frame_count += 1
                    if _null_frame_count == 1 or _null_frame_count % 10 == 0:
                        logger.warning(f"👁️ Camera returned no frame (×{_null_frame_count}) — Vector may be disconnected")
                    await asyncio.sleep(self.interval)
                    continue
                _null_frame_count = 0  # reset on successful frame

                self.last_run_time = time.time()
                comment = await self.scan_and_update_memory(frame)

                if comment and on_change_fn:
                    await on_change_fn(comment)

            except Exception as e:
                logger.error(f"👁️ Scene descriptor error: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

        logger.info("👁️ Scene descriptor loop stopped")

    def stop(self):
        """Stop the background loop."""
        self._running = False

    def mark_spoke(self):
        """Call after TTS playback to start the post-speak cooldown."""
        self.last_spoke_time = time.time()

    # ------------------------------------------------------------------ #
    #  Object teaching — user tells Vector what something is              #
    # ------------------------------------------------------------------ #

    async def teach_object(self, vlm_name: str, user_label: str):
        """
        Teach Vector that a VLM-detected object has a user-given name.

        E.g. teach_object('black box', 'stampante') stores the association
        so that next time Vector sees 'black box' it knows it's a printer.
        The label persists in ChromaDB and survives restarts.
        """
        key = self._normalize_name(vlm_name)
        if not key:
            return
        label = user_label.strip()
        if not label:
            return

        # Update in-memory record (create if not yet seen)
        if key in self.visual_memory:
            self.visual_memory[key]["user_label"] = label
        else:
            now_str = datetime.now().isoformat()
            self.visual_memory[key] = {
                "description": vlm_name,
                "first_seen": now_str,
                "last_seen": now_str,
                "seen_count": 1,
                "user_label": label,
            }

        # Persist to ChromaDB
        if self.db:
            try:
                await self.db.store_visual_memory(
                    object_name=key,
                    description=self.visual_memory[key]["description"],
                    user_label=label,
                )
            except Exception as e:
                logger.debug(f"\U0001f441\ufe0f Could not persist user label: {e}")

        logger.info(f"\U0001f441\ufe0f Object taught: '{vlm_name}' → '{label}'")

    def get_context_for_prompt(self) -> Dict[str, Any]:
        """Return current scene info for injection into chat system prompt."""
        return {
            "scene_description": self.last_description,
            "scene_change": self.last_change_description,
        }

