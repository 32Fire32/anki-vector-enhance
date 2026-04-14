"""Personal Fact Extractor
=========================
Silently extracts personal facts the user mentions during conversation
and stores them in a dedicated ChromaDB collection that persists forever,
independently of the 72-hour conversation window.

Examples of facts extracted:
  "Mio figlio si chiama Marco"       → figlio.nome = Marco
  "Ho un figlio di 7 anni"           → figlio.età = 7
  "Il mio compleanno è il 15 marzo"  → utente.compleanno = 15 marzo
  "Vivo a Milano"                    → utente.città = Milano
  "Faccio l'ingegnere"               → utente.lavoro = ingegnere
  "Mi chiamo Nicola"                 → utente.nome_utente = Nicola

Works as a fire-and-forget background task: callers await it but any
exception is swallowed so it never disrupts the main response flow.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight pre-filter (regex) — avoids LLM call for uninteresting turns
# ---------------------------------------------------------------------------

_FACT_SIGNALS = [
    # ── FIRST-PERSON: user talking about themselves ──────────────────────────
    r"\b(mio|mia)\s+(figlio|figlia|marito|moglie|madre|padre|fratello|sorella|nipote|nonno|nonna|gatto|cane|animale)\b",
    r"\bho\s+un[ao]?\s+(figlio|figlia|fratello|sorella|marito|moglie|gatto|cane)\b",
    r"\bmi\s+chiamo\b",
    r"\bmi\s+chiama\b",
    r"\bho\s+\d+\s+anni\b",
    r"\b(abito|vivo)\s+(a|in)\b",                   # FIX: removed [A-Z] (text is .lower())
    r"\bsono\s+(nato|nata)\b",
    r"\bnato\s+il\b",
    r"\bil\s+mio\s+compleanno\b",
    r"\bcompio\s+(gli\s+anni|il)\b",
    r"\blavoro\s+(come|in|a|da|per)\b",
    r"\bfaccio\s+(il|la|l[''\u2019]\s*)\b",
    # Preferences / Likes / Dislikes / Hobbies
    r"\bmi\s+piace\b",
    r"\bmi\s+piacciono\b",
    r"\bnon\s+mi\s+piace\b",
    r"\bamo\s+(il|la|lo|l[''\u2019]|i|gli|le)\b",
    r"\bodio\s+(il|la|lo|l[''\u2019]|i|gli|le)\b",
    r"\bpreferisco\b",
    r"\b(sport|hobby|passione|piatto|cibo)\s+preferit[oa]\b",
    r"\bgioco\s+a\b",                               # "gioco a calcio"
    r"\bfaccio\s+(yoga|palestra|corsa|nuoto|tennis|calcio|ciclismo|pilates)\b",
    r"\bmi\s+appassiona\b",
    # Health / Diet / Allergies
    r"\bsono\s+(allergic[oa]|celiaco|vegano|vegetariano|diabetico|iperteso|intollerante)\b",
    r"\bnon\s+mangio\s+(carne|pesce|glutine|lattosio|uova)\b",
    r"\bho\s+(diabete|ipertensione|asma|allergia|celiachia)\b",
    r"\bintollerante\s+al\b",
    # Dates / Events / Appointments
    r"\bcompleanno\s+di\b",                          # "il compleanno di Marco"
    r"\banniversario\b",
    r"\bappuntamento\b",
    r"\bho\s+un[ao]?\s+(appuntamento|visita|riunione|colloquio)\b",
    r"\b(domani|dopodomani|lunedì|martedì|mercoledì|giovedì|venerdì|sabato|domenica)\s+ho\b",
    r"\b(il|l[''\u2019])\s+\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\b",
    # Restaurants / Places
    r"\b(ristorante|pizzeria|trattoria|osteria|bistrot)\b",
    r"\bsi\s+mangia\s+(bene|benissimo|male)\b",
    # Explicit memory requests — always capture
    r"\bricordati\s+(che|di)\b",
    r"\bnon\s+dimenticare\b",
    r"\bsegna\s+(che|questo)\b",
    r"\btieni\s+presente\b",
    r"\bvoglio\s+che\s+(tu\s+sappia|tu\s+ricordi)\b",
    r"\bimportante\s+(che|sapere)\b",
    # ── THIRD-PERSON: user mentioning facts about other people / pets ────────
    r"\bsi\s+chiama(?:no)?\b",                       # "si chiama" / "si chiamano"
    r"\bha\s+\d+\s+anni\b",                          # "ha 7 anni"
    r"\b\d+\s+anni\b",                               # any age mention
    r"\bha\s+(un[ao]?|due|tre|\d+)\s+(figlio|figlia|figli|figlie|fratello|sorella|gatto|cane)\b",
    r"\bha\s+(due|tre|quattro|cinque|\d+)\s+figli\b",
    r"\bi\s+figli\s+di\b",
    r"\bil\s+figlio\s+di\b",
    r"\bla\s+figlia\s+di\b",
    r"\b\w+\s+ha\s+(due|tre|quattro|cinque|\d+)\b",
    r"\b\w+\s+abita\s+(a|in)\b",
    r"\b\w+\s+lavora\s+(come|a|in|da|per)\b",
    r"\bla\s+moglie\s+di\b",
    r"\bil\s+marito\s+di\b",
    r"\banni\s+fa\b",
    r"\bil\s+compleanno\s+di\b",                     # "il compleanno di Marco è..."
    r"\bcompie\s+gli\s+anni\b",                      # "Luigi compie gli anni..."
    r"\b\w+\s+fa\s+(il|la)\b",                       # "Marco fa il medico"
    r"\b\w+\s+è\s+(nato|nata)\b",                    # "Marco è nato il..."
]

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_PROMPT = """\
Estrai i fatti da questa frase. Possono riguardare l'utente, persone/animali menzionati, luoghi o eventi.
Rispondi SOLO con un JSON array (niente altro testo).
Se non ci sono fatti, rispondi: []

Schema: [{{"entity": "...", "entity_type": "...", "attribute": "...", "value": "..."}}]

Campi:
  entity       : nome dell'entità ("utente" per l'utente; nome proprio per altri: "Jowel", "Marco", "Da Mario")
  entity_type  : utente | figlio | figlia | marito | moglie | padre | madre | fratello | sorella | gatto | cane | persona | ristorante | luogo | evento
  attribute    : nome | nome_utente | età | compleanno | anniversario | città | lavoro | figli | figli_nomi |
                 hobby | preferenza | antipatia | allergia | dieta | appuntamento | data | nota_speciale | note
  value        : il valore estratto (per date usare formato "gg mese" o "gg mese anno")

Regole:
- Usa entity_type="utente" SOLO quando la frase parla dell'utente stesso ("io", "mi chiamo", "ho anni", "mi piace").
- Per persone nominate, usa il loro nome come entity e entity_type="persona".
- Per ristoranti/locali usa entity_type="ristorante"; per appuntamenti/eventi usa entity_type="evento".
- Se vengono menzionati più figli con nomi, metti i nomi in attribute="figli_nomi", separati da virgola.
- Per richieste esplicite di memoria ("ricordati che", "non dimenticare", "tieni presente"), usa attribute="nota_speciale".
- Non inventare fatti. Se il fatto non è esplicito, omettilo.

Esempi:
"Mi chiamo Nicola" → [{{"entity": "utente", "entity_type": "utente", "attribute": "nome_utente", "value": "Nicola"}}]
"Ho una figlia di 7 anni" → [{{"entity": "figlia", "entity_type": "figlia", "attribute": "età", "value": "7"}}]
"Vivo a Milano" → [{{"entity": "utente", "entity_type": "utente", "attribute": "città", "value": "Milano"}}]
"Il mio compleanno è il 15 marzo" → [{{"entity": "utente", "entity_type": "utente", "attribute": "compleanno", "value": "15 marzo"}}]
"Il compleanno di Marco è l'8 aprile" → [{{"entity": "Marco", "entity_type": "persona", "attribute": "compleanno", "value": "8 aprile"}}]
"Il nostro anniversario è il 5 giugno" → [{{"entity": "utente", "entity_type": "utente", "attribute": "anniversario", "value": "5 giugno"}}]
"Jowel ha due figli che si chiamano Andrea e Anita" → [{{"entity": "Jowel", "entity_type": "persona", "attribute": "figli", "value": "due"}}, {{"entity": "Jowel", "entity_type": "persona", "attribute": "figli_nomi", "value": "Andrea, Anita"}}]
"Marco abita a Roma" → [{{"entity": "Marco", "entity_type": "persona", "attribute": "città", "value": "Roma"}}]
"Mi piace tantissimo il sushi" → [{{"entity": "utente", "entity_type": "utente", "attribute": "preferenza", "value": "sushi"}}]
"Non mi piace la verdura" → [{{"entity": "utente", "entity_type": "utente", "attribute": "antipatia", "value": "verdura"}}]
"Sono allergico al lattosio" → [{{"entity": "utente", "entity_type": "utente", "attribute": "allergia", "value": "lattosio"}}]
"Sono vegano" → [{{"entity": "utente", "entity_type": "utente", "attribute": "dieta", "value": "vegano"}}]
"Gioco a calcio ogni sabato" → [{{"entity": "utente", "entity_type": "utente", "attribute": "hobby", "value": "calcio"}}]
"Da Mario si mangia benissimo, è il mio preferito" → [{{"entity": "Da Mario", "entity_type": "ristorante", "attribute": "nota_speciale", "value": "ristorante preferito dell'utente"}}]
"Ho un appuntamento dal dentista il 20 maggio" → [{{"entity": "appuntamento dentista", "entity_type": "evento", "attribute": "data", "value": "20 maggio"}}]
"Ricordati che mia sorella si chiama Elena" → [{{"entity": "sorella", "entity_type": "sorella", "attribute": "nome", "value": "Elena"}}]
"Non dimenticare che il mio medico è il dottor Rossi" → [{{"entity": "utente", "entity_type": "utente", "attribute": "nota_speciale", "value": "medico: dottor Rossi"}}]
"Che bel tempo oggi" → []
"Come stai?" → []

Frase: {text}"""


class FactExtractor:
    """Extract personal facts from user messages and store them in ChromaDB.

    Parameters
    ----------
    db_connector : ChromaDBConnector
        Must support ``store_user_fact()``.
    llm_client : OllamaClient
        Used for a lightweight extraction call (temperature=0, max_tokens=200).
    """

    def __init__(self, db_connector: Any, llm_client: Any, entity_memory: Any = None):
        self._db = db_connector
        self._llm = llm_client
        self._entity_memory = entity_memory

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def extract_and_store(
        self,
        user_text: str,
        speaker_id: str = "owner",
    ) -> List[Dict[str, str]]:
        """Extract facts from *user_text* and persist them.

        Safe to call fire-and-forget (swallows all exceptions).
        Returns the list of facts that were stored (empty on failure).
        """
        if not user_text:
            return []
        if not self._has_fact_signals(user_text):
            return []

        prompt = _PROMPT.format(text=user_text.strip())
        try:
            raw = await self._llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=350,
            )
        except Exception as e:
            logger.debug(f"[FactExtractor] LLM call skipped: {e}")
            return []

        facts = self._parse_facts(raw)
        stored: List[Dict[str, str]] = []
        for fact in facts:
            # Support both old schema (subject) and new schema (entity/entity_type)
            entity_name = str(fact.get("entity") or fact.get("subject") or "").strip()
            entity_type = str(fact.get("entity_type") or entity_name).strip().lower()
            attribute = str(fact.get("attribute", "")).strip().lower()
            value = str(fact.get("value", "")).strip()
            if not entity_name or not attribute or not value:
                continue
            try:
                if self._entity_memory is not None:
                    # Structured entity profile (preferred)
                    await self._entity_memory.upsert_fact(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        attribute=attribute,
                        value=value,
                        speaker_id=speaker_id,
                    )
                else:
                    # Legacy flat store (fallback)
                    await self._db.store_user_fact(
                        speaker_id=speaker_id,
                        subject=entity_name,
                        attribute=attribute,
                        value=value,
                    )
                stored.append({"entity": entity_name, "attribute": attribute, "value": value})
                logger.info(f"[FactExtractor] 💾 saved: {entity_name}.{attribute} = {value!r}")
            except Exception as e:
                logger.warning(f"[FactExtractor] store failed for {entity_name}.{attribute}: {e}")

        return stored

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_fact_signals(self, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in _FACT_SIGNALS)

    @staticmethod
    def _parse_facts(raw: str) -> List[Dict]:
        """Extract JSON array from LLM output, tolerating surrounding text."""
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not m:
            return []
        try:
            result = json.loads(m.group())
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[FactExtractor] JSON parse failed: {raw[:120]!r}")
            return []
