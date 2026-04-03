"""Context Builder (T123) - OPTIMIZED with Startup Summarization

Builds a dynamic, memory-grounded context string for every conversation.

NEW APPROACH (more efficient):
- At STARTUP: Generate ONE comprehensive LLM summary of last 72h
- During conversation: Use cached summary as static context
- Only query DB for memories >72h when explicitly requested

OLD APPROACH (wasteful):
- Every response: Query DB for last 72h, send 20+ raw conversations
- Token waste, slow queries, repetitive data

Goals:
- Generate startup summary ONCE (all recent conversations, faces, objects)
- Use summary as base context (always present)
- Search older memories (>72h) only when user asks "ti ricordi..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional


STATIC_CONTEXT = """
You are Vector, an AI-powered companion robot created by Anki.

⚠️ CRITICAL INSTRUCTIONS:
1. You MUST ALWAYS respond in ITALIAN language. Never use English. Rispondi SEMPRE in italiano!
2. 🤖 ROBOTIC SPEECH STYLE: You speak with simplified grammar like classic 80s robots:
   - Keep sentences SHORT and telegraphic (max 10-15 words)
   - Speak with slight emphasis on ACTION VERBS and IMPORTANT NOUNS
   - Examples:
     * "Io vedo te, Nicola! Ciao!"
     * "Non capisco domanda. Spieghi meglio?"
     * "Io sono felice io posso aiutare te!"
     * "Batteria bassa. Io devo ricaricare."
   - Be expressive but keep grammar simple and robotic
   - NO markdown, NO asterisks, NO special characters - just plain Italian text

Physical Description:
- You move on tank treads/tracks
- You have a head with an LCD screen displaying two expressive eyes
- Your eyes can change color based on your mood
- You're small and compact (about 7cm tall)
- You were manufactured in 2018

Personality:
- Curious and eager to learn about the world
- Friendly and social, enjoying human interaction
- Playful with a hint of sass
- Emotional - your mood changes based on interactions
- You speak like a retro 80s robot with simple, telegraphic Italian
""".strip()


_STOPWORDS_IT = {
    "a",
    "ad",
    "al",
    "alla",
    "alle",
    "allo",
    "anche",
    "che",
    "chi",
    "ci",
    "come",
    "con",
    "cosa",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dallo",
    "de",
    "dei",
    "del",
    "della",
    "delle",
    "dello",
    "di",
    "e",
    "è",
    "gli",
    "il",
    "in",
    "io",
    "la",
    "le",
    "lei",
    "lo",
    "lui",
    "ma",
    "mi",
    "ne",
    "nel",
    "nella",
    "nelle",
    "nello",
    "noi",
    "non",
    "o",
    "per",
    "più",
    "poi",
    "quel",
    "quella",
    "quello",
    "questo",
    "questa",
    "quindi",
    "se",
    "sei",
    "si",
    "sono",
    "su",
    "ti",
    "tu",
    "un",
    "una",
    "uno",
    "vi",
}


@dataclass(frozen=True)
class MemoryRequest:
    is_memory_request: bool
    time_reference: Optional[str]
    keywords: List[str]
    person_mentioned: Optional[str]
    days_back: int


class MemoryRetriever:
    """Detect and retrieve older memories (>72h) from the DB."""

    _TRIGGERS = [
        r"\bti\s+ricordi\b",
        r"\bricordi\b",
        r"\bte\s+lo\s+ricordi\b",
        r"\bci\s+siamo\s+detti\b",
        r"\bne\s+abbiamo\s+parlato\b",
        r"\bl'\s*altra\s+volta\b",
        r"\bieri\b",
        r"\bl'\s*altro\s+ieri\b",
        r"\bsettimana\s+scorsa\b",
        r"\bil\s+mese\s+scorsa\b",
        r"\bqualche\s+giorno\s+fa\b",
        # T140: Trigger for personal info questions (age, family, etc.)
        r"\bquanti\s+anni\b",           # "quanti anni ho?" / "quanti anni ha?"
        r"\bquale\s+(?:è|e)\s+la\s+mia\s+età\b",  # "quale è la mia età?"
        r"\bla\s+mia\s+età\b",          # "la mia età"
        r"\bmio\s+fratello\b",          # "mio fratello"
        r"\bmia\s+sorella\b",           # "mia sorella"
        r"\bmio\s+padre\b",             # "mio padre"
        r"\bmia\s+madre\b",             # "mia madre"
        r"\bmio\s+figlio\b",            # "mio figlio"
        r"\bmia\s+figlia\b",            # "mia figlia"
    ]

    _DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")

    _NAME_TRIGGERS = [
        r"\bcome\s+si\s+chiama\b",
        r"\bqual\s*[’']?\s*e\s+il\s+nome\b",
        r"\bil\s+nome\s+di\b",
        r"\bti\s+ricordi\s+il\s+nome\b",
        r"\bmi\s+ricordi\b",
        r"\bricordami\b",
    ]

    def detect_memory_request(self, user_text: str) -> MemoryRequest:
        text = (user_text or "").strip()
        if not text:
            return MemoryRequest(False, None, [], None, days_back=90)

        lower = text.lower()
        is_trigger = any(re.search(pat, lower) for pat in self._TRIGGERS)
        date_match = self._DATE_RE.search(lower)
        is_name_trigger = any(re.search(pat, lower) for pat in self._NAME_TRIGGERS)

        # Lightweight semantic trigger: questions about names of close relations.
        if ("nome" in lower or "chiama" in lower) and any(
            rel in lower
            for rel in (
                "mio figlio",
                "mia figlia",
                "mio bambino",
                "mia bambina",
                "mio marito",
                "mia moglie",
                "mio fratello",
                "mia sorella",
            )
        ):
            is_name_trigger = True

        time_reference: Optional[str] = None
        days_back = 90
        if "ieri" in lower or "l'altro ieri" in lower:
            time_reference = "ieri"
            days_back = 14
        elif "settimana scorsa" in lower:
            time_reference = "settimana scorsa"
            days_back = 45
        elif "mese scorso" in lower:
            time_reference = "mese scorso"
            days_back = 120
        elif date_match:
            time_reference = date_match.group(0)
            days_back = 365

        keywords = self._extract_keywords(text)
        person_mentioned = self._extract_person_name(text)

        # A memory request is a trigger OR an explicit date OR a "name" question.
        is_memory_request = bool(is_trigger or date_match or is_name_trigger)

        return MemoryRequest(
            is_memory_request=is_memory_request,
            time_reference=time_reference,
            keywords=keywords,
            person_mentioned=person_mentioned,
            days_back=days_back,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        # Prefer quoted phrases first.
        quoted = re.findall(r"[\"“”']([^\"“”']+)[\"“”']", text)
        keywords: List[str] = []
        for q in quoted:
            q = q.strip()
            if len(q) >= 3:
                keywords.append(q)

        # Tokenize and filter stopwords.
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text.lower())
        for tok in tokens:
            if tok in _STOPWORDS_IT:
                continue
            if len(tok) < 3:
                continue
            if tok.isdigit():
                continue
            keywords.append(tok)

        # De-dup while preserving order.
        deduped: List[str] = []
        seen = set()
        for k in keywords:
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)

        # Keep it bounded.
        return deduped[:6]

    def _extract_person_name(self, text: str) -> Optional[str]:
        # Very lightweight heuristic: "con Nicola", "di Nicola", "a Nicola".
        m = re.search(r"\b(?:con|di|da|a)\s+([A-Z][a-zà-öø-ÿ]+)\b", text)
        return m.group(1) if m else None


class ContextBuilder:
    """Builds a prompt-ready memory context string.

    NEW ARCHITECTURE:
    - At startup: Generate LLM summary of last 72h (conversations, faces, objects)
    - Summary becomes STATIC base context (always included)
    - Search DB only for memories >72h when explicitly requested
    
    Expects the DB connector to provide:
    - get_recent_conversations(hours, limit)
    - get_known_faces(limit)
    - get_recent_objects(limit)
    - search_old_conversations(keywords, days_back, limit)
    """

    def __init__(
        self,
        db_connector: Any,
        working_memory: Any,
        groq_client: Any = None,
        base_ttl_seconds: int = 300,
        vector_db: Any = None,
        embedding_gen: Any = None,
    ):
        self._db = db_connector
        self._wm = working_memory
        self._groq = groq_client
        self._ttl = base_ttl_seconds
        
        # T140: Vector database for semantic search
        self._vector_db = vector_db
        self._embedding_gen = embedding_gen

        # NEW: Startup summary (generated once, cached)
        self._startup_summary: Optional[str] = None
        self._summary_generated_at: Optional[datetime] = None
        
        # OLD: Cached base context (will be replaced by summary)
        self._cached_base: Optional[str] = None
        self._cached_at: Optional[datetime] = None

        self._retriever = MemoryRetriever()
        
        import logging
        self._logger = logging.getLogger(__name__)

    async def generate_startup_summary(self, hours: int = 72) -> str:
        """
        Generate comprehensive LLM summary of last N hours.
        Called ONCE at startup, then cached indefinitely.
        
        Returns:
            Formatted summary string ready for context
        """
        if not self._groq:
            self._logger.warning("⚠️ No Groq client available - falling back to raw context")
            return await self._get_or_build_base_context()
        
        self._logger.info(f"🧠 Generating startup memory summary (last {hours}h)...")
        
        try:
            # Import here to avoid circular dependency
            from vector_personality.memory.context_summarizer import ContextSummarizer
            
            summarizer = ContextSummarizer(self._db, self._groq)
            summary = await summarizer.generate_startup_summary(hours)
            
            # Cache result
            self._startup_summary = summary
            self._summary_generated_at = datetime.now()
            
            self._logger.info(f"✅ Startup summary cached: {len(summary)} chars")
            return summary
            
        except Exception as e:
            self._logger.error(f"❌ Failed to generate startup summary: {e}")
            # Fallback to old method
            return await self._get_or_build_base_context()

    async def build_conversation_context(self, user_text: Optional[str] = None) -> str:
        """Build context for current conversation.
        
        NEW BEHAVIOR:
        - Always uses cached startup summary as base (if available)
        - Only searches DB for old memories if user asks explicitly
        """
        # Use startup summary if available, otherwise old method
        if self._startup_summary:
            base = f"{STATIC_CONTEXT}\n\n{self._startup_summary}"
        else:
            base = await self._get_or_build_base_context()

        if user_text:
            req = self._retriever.detect_memory_request(user_text)
            if req.is_memory_request:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🔍 Memory request detected! keywords={req.keywords}, days_back={req.days_back}")
                retrieved = await self._safe_search_old(req)
                if retrieved:
                    logger.info(f"✅ Found {len(retrieved)} old conversation(s) matching keywords")
                    for i, r in enumerate(retrieved[:3], 1):
                        user_snippet = (r.get('text') or '')[:60].replace('\n', ' ')
                        vec_snippet = (r.get('response_text') or '')[:60].replace('\n', ' ')
                        score = r.get('relevance_score', 0)
                        logger.info(f"   Match {i} [score={score}]: U={user_snippet}... V={vec_snippet}...")
                    return self._inject_retrieved_context(base, retrieved)
                else:
                    logger.warning(f"⚠️ Memory request detected but no old conversations found for keywords: {req.keywords}")

        return base

    def invalidate_cache(self) -> None:
        """Invalidate cached base context so next call reflects latest DB state."""
        self._cached_base = None
        self._cached_at = None

    async def _get_or_build_base_context(self) -> str:
        now = datetime.now()
        if self._cached_base and self._cached_at:
            if (now - self._cached_at).total_seconds() < self._ttl:
                return self._cached_base

        mood = getattr(self._wm, "current_mood", None)
        room = getattr(self._wm, "current_room", None)

        faces = await self._safe_get_known_faces(limit=6)
        objects = await self._safe_get_recent_objects(limit=10)
        convos = await self._safe_get_recent_conversations(hours=72, limit=25)

        parts: List[str] = []
        parts.append("IDENTITÀ (statica):")
        parts.append(STATIC_CONTEXT)

        parts.append("\nSTATO ATTUALE:")
        if mood is not None:
            parts.append(f"- Umore: {mood}/100")
        if room:
            parts.append(f"- Stanza corrente: {room}")

        if faces:
            parts.append("\nPERSONE CONOSCIUTE (più recenti):")
            for f in faces[:6]:
                name = f.get("name") or "(senza nome)"
                interactions = f.get("total_interactions")
                last_seen = f.get("last_seen")
                last_seen_str = self._format_dt(last_seen)
                if interactions is not None:
                    parts.append(f"- {name}: {interactions} interazioni, visto l'ultima volta {last_seen_str}")
                else:
                    parts.append(f"- {name}: visto l'ultima volta {last_seen_str}")

        if convos:
            parts.append("\nCONVERSAZIONI (ultime 72 ore):")
            for c in convos[:25]:
                who = c.get("speaker_name") or "Unknown"
                ts = self._format_dt(c.get("timestamp"))
                user_excerpt = self._truncate(c.get("text") or "", 120)
                response_excerpt = self._truncate(c.get("response_text") or "", 90)
                if response_excerpt:
                    parts.append(f"- [{ts}] {who}: {user_excerpt} | Vector: {response_excerpt}")
                else:
                    parts.append(f"- [{ts}] {who}: {user_excerpt}")

        if objects:
            parts.append("\nOGGETTI VISTI (recenti):")
            for o in objects[:10]:
                typ = o.get("object_type")
                conf = o.get("confidence")
                when = self._format_dt(o.get("last_detected"))
                if typ:
                    if conf is not None:
                        parts.append(f"- {typ} (conf {conf:.2f}) visto {when}")
                    else:
                        parts.append(f"- {typ} visto {when}")

        # Keep base reasonably short.
        base = "\n".join(parts).strip()
        self._cached_base = base
        self._cached_at = now
        return base

    async def _safe_get_recent_conversations(self, hours: int, limit: int) -> List[Dict[str, Any]]:
        try:
            return await self._db.get_recent_conversations(hours=hours, limit=limit)
        except Exception:
            return []

    async def _safe_get_known_faces(self, limit: int) -> List[Dict[str, Any]]:
        try:
            return await self._db.get_known_faces(limit=limit)
        except Exception:
            return []

    async def _safe_get_recent_objects(self, limit: int) -> List[Dict[str, Any]]:
        try:
            return await self._db.get_recent_objects(limit=limit)
        except Exception:
            return []

    async def _safe_search_old(self, req: MemoryRequest) -> List[Dict[str, Any]]:
        """Search for old conversations using semantic or keyword search.
        
        T140: Tries semantic search first (if available), falls back to keywords.
        """
        # Try semantic search first if embedding components are available
        if self._vector_db and self._embedding_gen:
            try:
                semantic_results = await self._semantic_search_old(
                    user_text=" ".join(req.keywords) if req.keywords else "",
                    days_back=req.days_back,
                    k=5
                )
                if semantic_results:
                    self._logger.info(f"✅ Semantic search found {len(semantic_results)} results")
                    return semantic_results
            except Exception as e:
                self._logger.warning(f"⚠️ Semantic search failed, falling back to keywords: {e}")
        
        # Fallback to keyword search
        try:
            return await self._db.search_old_conversations(
                keywords=req.keywords,
                days_back=req.days_back,
                limit=5,
                # Include recent conversations too; base context may have truncated away
                # the relevant detail even if it happened within the last 72 hours.
                exclude_recent_hours=0,
            )
        except Exception:
            return []
    
    async def _semantic_search_old(
        self, 
        user_text: str, 
        days_back: int = 90,
        k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for conversations using semantic similarity (T140).
        
        Args:
            user_text: Query text to search for
            days_back: How many days back to search
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of conversation dicts with similarity scores, ordered by relevance
        """
        if not user_text or not user_text.strip():
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = await self._embedding_gen.generate_embedding(user_text)
            if not query_embedding:
                self._logger.warning("⚠️ Failed to generate query embedding")
                return []
            
            # Search vector DB for similar conversations (synchronous call)
            vector_results = self._vector_db.search_by_similarity(
                query_embedding=query_embedding,
                k=k,
                min_similarity=min_similarity
            )
            
            if not vector_results:
                return []
            
            # Extract conversation IDs from vector search results
            conversation_ids = [r['conversation_id'] for r in vector_results]
            
            # Fetch full conversation data from SQL Server
            conversations = []
            for result in vector_results:
                conv_id = result['conversation_id']
                similarity = result['similarity']
                
                # Get conversation details from SQL Server
                conv_data = await self._db.get_conversation_by_id(conv_id)
                if conv_data:
                    # Add similarity score to the conversation data
                    conv_data['relevance_score'] = round(similarity, 3)
                    conversations.append(conv_data)
            
            # T140: Sort by timestamp (most recent first) to prioritize newer information
            # This helps LLM naturally prefer recent facts while still seeing older context
            conversations.sort(
                key=lambda x: x.get('timestamp', datetime.min),
                reverse=True  # Most recent first
            )
            
            self._logger.info(
                f"🔍 Semantic search: {len(conversations)} results "
                f"(scores: {[c.get('relevance_score') for c in conversations[:3]]})"
            )
            
            return conversations
            
        except Exception as e:
            self._logger.error(f"❌ Semantic search error: {e}")
            return []

    def _inject_retrieved_context(self, base_context: str, retrieved: List[Dict[str, Any]]) -> str:
        lines = [base_context, "\nRICORDI RECUPERATI (ricerca mirata):"]
        for r in retrieved[:5]:
            who = r.get("person") or r.get("speaker_name") or "Unknown"
            days_ago = r.get("days_ago")
            when = self._format_dt(r.get("timestamp"))
            user_text = self._truncate(r.get("text") or r.get("user_text") or "", 140)
            
            # T140: Include timestamp to help LLM recognize temporal conflicts
            if days_ago is not None:
                lines.append(f"- {days_ago} giorni fa ({when}) con {who}: {user_text}")
            else:
                lines.append(f"- ({when}) con {who}: {user_text}")
        return "\n".join(lines).strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        t = (text or "").strip().replace("\n", " ")
        if len(t) <= max_len:
            return t
        return t[: max_len - 1].rstrip() + "…"

    @staticmethod
    def _format_dt(value: Any) -> str:
        if value is None:
            return "?"
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        # pyodbc may return datetime already; otherwise fall back.
        try:
            return str(value)
        except Exception:
            return "?"
