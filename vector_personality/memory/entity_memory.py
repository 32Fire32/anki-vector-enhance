"""
Entity Memory
=============
Structured profiles for every person, pet, place or thing the user
mentions during conversations.

Instead of burying facts in a large text dump, each entity gets a
compact, structured "card" that is always injected near the top of
the LLM context.  Small models (gemma3:4b) work dramatically better
when facts are presented as structured key-value cards rather than
scattered across thousands of tokens of conversation history.

Example card injected into the prompt:

    [Marco — figlio]
    • età: 7 anni
    • hobby: cartoni animati
    • non piace: film horror
    Menzione più recente: ieri

    [Nicola (tu) — proprietario]
    • città: Milano
    • lavoro: ingegnere
    • compleanno: 15 marzo

Storage: ChromaDB  `entity_profiles`  collection in the main
vector_memory_chroma directory.  Embeddings are 3-dim dummies so the
collection doesn't conflict with other high-dim collections; retrieval
is always done via .get() (not .query()) because the corpus is bounded
(typically 5–30 entities) and small enough to inject in full.

When a scalable semantic lookup is needed in the future, upgrade to
real embeddings — the rest of the code doesn't need to change.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _days_ago_label(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        d = (datetime.now() - dt).days
        if d == 0:
            return "oggi"
        if d == 1:
            return "ieri"
        return f"{d} giorni fa"
    except Exception:
        return ""


class EntityMemory:
    """
    Manage structured entity profiles persisted in ChromaDB.

    Each entity is identified by the key  ``{speaker_id}__{type}__{name}``.
    Facts are stored as a JSON dict inside ChromaDB metadata so that
    upsert merges new facts with existing ones rather than overwriting.
    """

    COLLECTION = "entity_profiles"

    def __init__(self, chromadb_client, embedder=None):
        """
        Parameters
        ----------
        chromadb_client : chromadb.PersistentClient
            The raw ChromaDB client (available as ``ChromaDBConnector.client``).
        embedder : EmbeddingGenerator | None
            Reserved for future semantic lookup.  Currently unused.
        """
        self._client = chromadb_client
        self._embedder = embedder
        self._col = chromadb_client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[EntityMemory] ready — {self._col.count()} profiles loaded")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_fact(
        self,
        entity_name: str,
        entity_type: str,
        attribute: str,
        value: str,
        speaker_id: str = "owner",
    ) -> None:
        """Add or update a single fact about an entity.

        If the entity already has a card, the new fact is merged into
        the existing facts dict (preserving everything else).
        """
        eid = self._eid(entity_name, entity_type, speaker_id)

        # Load existing facts
        try:
            result = self._col.get(ids=[eid], include=["metadatas"])
            if result["ids"]:
                existing_facts: Dict[str, str] = json.loads(
                    result["metadatas"][0].get("facts_json", "{}")
                )
            else:
                existing_facts = {}
        except Exception:
            existing_facts = {}

        existing_facts[attribute] = value
        now = _now_iso()

        meta: Dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "speaker_id": speaker_id,
            "facts_json": json.dumps(existing_facts, ensure_ascii=False),
            "updated_at": now,
        }

        self._col.upsert(
            ids=[eid],
            embeddings=[[0.0, 0.0, 0.0]],
            metadatas=[meta],
        )
        logger.info(f"[EntityMemory] 💾 {eid} → {attribute} = {value!r}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_all_cards(
        self, speaker_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all entity cards, optionally filtered by speaker."""
        try:
            if speaker_id:
                result = self._col.get(
                    where={"speaker_id": speaker_id}, include=["metadatas"]
                )
            else:
                result = self._col.get(include=["metadatas"])

            cards = []
            for eid, meta in zip(result["ids"], result["metadatas"]):
                cards.append(
                    {
                        "entity_id": eid,
                        "entity_name": meta.get("entity_name", "?"),
                        "entity_type": meta.get("entity_type", "?"),
                        "facts": json.loads(meta.get("facts_json", "{}")),
                        "updated_at": meta.get("updated_at", ""),
                    }
                )

            # Sort: most recently updated first
            cards.sort(key=lambda c: c["updated_at"], reverse=True)
            return cards
        except Exception as e:
            logger.warning(f"[EntityMemory] get_all_cards failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_cards_for_prompt(cards: List[Dict[str, Any]]) -> str:
        """Return a structured block ready to inject at the top of the LLM prompt.

        Format prioritises readability for small models:
        - One card per entity
        - Bullet points for facts
        - Recency hint so the model understands when info was captured
        """
        if not cards:
            return ""

        lines = [
            "PROFILI (persone, animali, luoghi importanti nella tua vita con l'utente):",
            "Usa queste informazioni in modo NATURALE quando sono rilevanti — non elencarle.",
        ]
        for card in cards:
            name = card.get("entity_name", "?")
            etype = card.get("entity_type", "?")
            facts: Dict[str, str] = card.get("facts", {})
            when = _days_ago_label(card.get("updated_at", ""))

            header = f"\n  [{name} — {etype}]"
            if when:
                header += f"  (menzione: {when})"
            lines.append(header)
            for attr, val in facts.items():
                lines.append(f"    • {attr}: {val}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eid(entity_name: str, entity_type: str, speaker_id: str) -> str:
        name = entity_name.lower().strip().replace(" ", "_")
        etype = entity_type.lower().strip().replace(" ", "_")
        return f"{speaker_id}__{etype}__{name}"
