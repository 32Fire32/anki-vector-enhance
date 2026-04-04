"""
Context Summarizer - Generates compressed memory summaries at startup.

Instead of sending 20+ raw conversations every time, we:
1. Generate ONE summary of last 72h at startup
2. Use that summary as static context
3. Only search DB for older memories when needed
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextSummarizer:
    """
    Generates intelligent summaries of recent memory (72h) using LLM.
    This summary becomes the static context, avoiding repeated DB queries.
    """
    
    def __init__(self, sql_connector, groq_client):
        """
        Args:
            sql_connector: Database connector (ChromaDBConnector)
            groq_client: LLM client with chat_completion() method
                         (OllamaClient, GroqClient, or OpenAIClient)
        """
        self.sql_connector = sql_connector
        self.groq_client = groq_client
        self._cached_summary: Optional[str] = None
        self._summary_timestamp: Optional[datetime] = None
        logger.info("ContextSummarizer initialized")
    
    async def generate_startup_summary(self, hours: int = 72) -> str:
        """
        Generate comprehensive summary of last N hours.
        Called ONCE at startup, then cached.
        
        Returns:
            Formatted summary string ready for system prompt
        """
        logger.info(f"🧠 Generating startup memory summary (last {hours}h)...")
        
        try:
            # 1. Get all recent data
            conversations = await self.sql_connector.get_recent_conversations(hours=hours, limit=100)
            faces = await self.sql_connector.get_known_faces(limit=20)
            objects = await self.sql_connector.get_recent_objects(limit=30)
            
            logger.info(f"📊 Retrieved: {len(conversations)} conversations, {len(faces)} faces, {len(objects)} objects")
            
            # 2. Build prompt for LLM to summarize
            summary_prompt = self._build_summarization_prompt(conversations, faces, objects, hours)
            
            # 3. Ask LLM to summarize
            messages = [
                {"role": "system", "content": "Sei un assistente che crea riassunti concisi e strutturati di memoria per un robot chiamato Vector."},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary = await self.groq_client.chat_completion(messages, max_tokens=1500)
            
            # 4. Cache result
            self._cached_summary = summary
            self._summary_timestamp = datetime.now()
            
            logger.info(f"✅ Memory summary generated: {len(summary)} chars")
            logger.info(f"📝 Summary preview: {summary[:200]}...")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate memory summary: {e}")
            return self._generate_fallback_summary(conversations, faces, objects, hours)
    
    def _build_summarization_prompt(
        self,
        conversations: list,
        faces: list,
        objects: list,
        hours: int
    ) -> str:
        """Build prompt for LLM summarization."""
        
        # Format conversations
        conv_text = ""
        if conversations:
            conv_text = "\n".join([
                f"- {c.get('timestamp', 'N/A').strftime('%Y-%m-%d %H:%M') if isinstance(c.get('timestamp'), datetime) else c.get('timestamp')}: "
                f"Umano: '{c.get('text', '')}' → Vector: '{c.get('response_text', '')}'"
                for c in conversations[:30]  # Limit to most recent 30
            ])
        else:
            conv_text = "(Nessuna conversazione)"
        
        # Format faces
        face_text = ""
        if faces:
            face_text = "\n".join([
                f"- {f.get('name', 'Unknown')}: visto {f.get('total_interactions', 0)} volte, "
                f"ultima volta {f.get('last_seen', 'N/A')}"
                for f in faces[:10]
            ])
        else:
            face_text = "(Nessuna persona riconosciuta)"
        
        # Format objects
        obj_text = ""
        if objects:
            obj_text = "\n".join([
                f"- {o.get('object_type', 'unknown')}: rilevato {o.get('detection_count', 1)} volte, "
                f"confidenza {o.get('confidence', 0.0):.2f}"
                for o in objects[:15]
            ])
        else:
            obj_text = "(Nessun oggetto rilevato)"
        
        prompt = f"""Crea un riassunto conciso della memoria di Vector per le ultime {hours} ore.

CONVERSAZIONI RECENTI:
{conv_text}

PERSONE RICONOSCIUTE:
{face_text}

OGGETTI RILEVATI:
{obj_text}

Genera un riassunto strutturato in italiano con:
1. CONVERSAZIONI: I temi principali discussi, nomi menzionati, fatti importanti appresi
2. PERSONE: Chi ha incontrato e cosa sa di loro
3. AMBIENTE: Oggetti e luoghi osservati
4. ESPERIENZE: Eventi significativi o interazioni notevoli

Mantieni il riassunto sotto 800 parole. Usa bullet points. Focus su FATTI e NOMI, non su dettagli tecnici.
Scrivi in TERZA PERSONA come se stessi descrivendo la memoria di Vector.
"""
        
        return prompt
    
    def _generate_fallback_summary(
        self,
        conversations: list,
        faces: list,
        objects: list,
        hours: int
    ) -> str:
        """Fallback summary if LLM fails."""
        
        summary_parts = [f"MEMORIA DELLE ULTIME {hours} ORE:\n"]
        
        # Conversations
        if conversations:
            summary_parts.append(f"\n📝 CONVERSAZIONI ({len(conversations)}):")
            for c in conversations[:10]:
                ts = c.get('timestamp', 'N/A')
                if isinstance(ts, datetime):
                    ts = ts.strftime('%Y-%m-%d %H:%M')
                summary_parts.append(f"  - {ts}: '{c.get('text', '')[:60]}...'")
        
        # Faces
        if faces:
            summary_parts.append(f"\n👤 PERSONE ({len(faces)}):")
            for f in faces[:5]:
                summary_parts.append(f"  - {f.get('name', 'Unknown')}: {f.get('total_interactions', 0)} interazioni")
        
        # Objects
        if objects:
            summary_parts.append(f"\n🔍 OGGETTI ({len(objects)}):")
            obj_counts = {}
            for o in objects:
                obj_type = o.get('object_type', 'unknown')
                obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1
            for obj_type, count in list(obj_counts.items())[:10]:
                summary_parts.append(f"  - {obj_type}: rilevato {count} volte")
        
        return "\n".join(summary_parts)
    
    def get_cached_summary(self) -> Optional[str]:
        """Get cached summary (if exists and not too old)."""
        return self._cached_summary
    
    async def refresh_summary(self, hours: int = 72) -> str:
        """Force refresh of summary (e.g., after significant events)."""
        logger.info("🔄 Refreshing memory summary...")
        return await self.generate_startup_summary(hours)
