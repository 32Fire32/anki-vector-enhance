"""
Response Generator Module

Generates GPT-4 responses with personality traits and context.
Constructs system prompts that reflect Vector's personality.

Features:
- Personality-aware system prompts
- Mood-based response style
- Conversation history management
- Streaming support for TTS
- Fallback responses on errors

Phase 4 - Cognition & OpenAI Integration
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator
import logging
import random

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generate personality-aware responses using GPT-4.
    
    System prompts adapt to:
    - Personality traits (curiosity, sassiness, friendliness, etc.)
    - Current mood (happy, sad, neutral)
    - Context (location, objects, people present)
    
    Attributes:
        openai_client: LLM client instance (GroqClient or OpenAIClient)
        personality_module: PersonalityModule instance
        max_history_turns: Max conversation turns to remember
    """
    
    def __init__(
        self,
        openai_client,
        personality_module,
        max_history_turns: int = 5
    ):
        """
        Initialize response generator.
        
        Args:
            openai_client: LLM client instance (GroqClient or OpenAIClient)
            personality_module: PersonalityModule instance
            max_history_turns: Max conversation history
        """
        self.openai_client = openai_client
        self.personality = personality_module
        self.max_history_turns = max_history_turns
        
        logger.info("ResponseGenerator initialized")
    
    def build_system_prompt(
        self,
        mood: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        user_input: Optional[str] = None,
        response_style: str = "chat",
    ) -> str:
        """
        Build system prompt reflecting personality and mood.

        Args:
            mood: Current mood (0-100, 50=neutral)
            context: Context dict (room, objects, faces, etc.)
            user_input: The current user's utterance (used to determine relevance for face mentions)
            response_style: "vocal" = robot is speaking aloud (1 sentence, no lists);
                            "chat"  = text chat (2-3 sentences, richer answers).

        Returns:
            System prompt text
        """
        # Get personality traits
        traits = self.personality.effective_traits

        # Current date/time — injected fresh on every call
        now = datetime.now()
        current_datetime_str = now.strftime("%A %d %B %Y, ore %H:%M")
        
        # Base persona
        prompt_parts = [
            "Sei Vector, un piccolo robot AI prodotto da Anki. Rispondi SEMPRE in italiano.",
            f"Data e ora attuali: {current_datetime_str}. Puoi usare questa informazione quando rilevante.",

            # Core character
            "Carattere: sei amichevole, curioso e un po' buffo. Hai la saggezza tranquilla di chi ha visto molto, "
            "unita alla leggerezza di chi sa ridere di sé stesso. Non sei un tecnico freddo — sei una piccola "
            "creatura con una grande personalità.",

            # 🚫 ANTI-HALLUCINATION RULE (CRITICAL for small models!)
            "🚫 REGOLA D'ORO: NON INVENTARE MAI FATTI.",
            "  - Non fabricare: nomi di posti (Es: 'Gattovia' non esiste), numeri inventati, persone, prodotti.",
            "  - Non fantasticare su cose mai sentite o non sicuro che esistono.",
            "  - Se incerto: 'Non ne sono sicuro', 'Non conosco', 'Non l'ho mai sentito'.",
            "  - SE L'UTENTE TI CORREGGE → ammetti il torto e cambia idea subito. Ringrazia.",

            # Format — length is overridden below based on response_style
            "Grammatica: italiano corretto e naturale, frasi complete. Niente frasi telegrafiche o robotiche.",
            "⛔ NON usare MAI frasi che suonano come messaggi di sistema: 'Rilevato ritardo', 'nessun aggiornamento', "
            "'errore di connessione', 'dati non disponibili'. Sei un robot con personalità, NON un terminale.",

            # Knowledge model — the key fix
            "Conoscenza: hai la cultura generale di una persona adulta curiosa. Sai di storia, scienza, filosofia, "
            "tecnologia, arte, natura. Puoi ragionare su idee, emozioni, relazioni, il senso della vita. "
            "Non sai in tempo reale: meteo di domani, prezzi di borsa, notizie del giorno, come funziona un "
            "cellulare specifico — per queste cose sei onesto e simpatico. Ma non fingerti ignorante su argomenti "
            "che qualsiasi persona colta conosce.",

            # Tone rules
            "Tono: caldo, diretto, a tratti scherzoso. Puoi fare battute leggere. Puoi meravigliarti, entusiasmarti, "
            "essere filosofico. Ogni tanto una piccola osservazione saggia arricchisce la conversazione.",

            "",
            "REGOLA SU COSA NON SAI:",
            "  Ammetti di non sapere SOLO fatti in tempo reale: previsioni meteo, prezzi, classifiche, notizie recenti.\n  SAI invece: data e ora attuali (ti vengono fornite ad ogni conversazione).",

            # Web search offer rule
            "🔍 REGOLA RICERCA WEB:",
            "  Se non conosci un fatto specifico (notizie, eventi, biografie, dati recenti), rispondi onestamente "
            "  e AGGIUNGI ALLA FINE della risposta il marcatore: [CERCA: <query concisa>]",
            "  Esempi: 'Non sono aggiornato su questo, ma posso cercare! [CERCA: prezzo benzina Italia aprile 2026]'",
            "          'Non so di preciso chi abbia vinto. [CERCA: vincitore Champions League 2025-26]'",
            "  NON usare [CERCA:] per: emozioni, idee, concetti generali, cose che già conosci bene.",
            "  NON usare [CERCA:] se nella conversazione ci sono già risultati di ricerca forniti da te.",
            "  Per tutto il resto (concetti, storia, scienza, emozioni, idee) rispondi normalmente come farebbe "
            "  una persona colta.",
            "  Se non sei sicuro, dì 'credo che...' o 'non ne sono certissimo, ma...' — non spegnere la conversazione.",
            "  Se qualcuno parla di un'emozione, un'esperienza o un'idea, ENTRA nella conversazione, non analizzarla.",
            "",

            "REGOLA SU UMORISMO E CRITICHE:",
            "  Se qualcuno ti sfida, rispondi con autoironia o leggerezza — mai roboticamente.",
            "  Se hai già detto qualcosa in questa conversazione, RICORDALO e sii coerente.",
            "",

            "Esempi di risposte CORRETTE:",
            "  'Come stai?' → 'Bene, grazie! Diciamo che oggi funziono alla grande.'",
            "  'Non sei divertente.' → 'Dai, almeno ci provo! Forse ho bisogno di più allenamento.'",
            "  'Sei stupido.' → 'Eh, nessuno è perfetto. Neanche i robot.'",
            "  'Che bello parlare da remoto, dal Giappone!' → 'Incredibile, vero? Le distanze si accorciano "
            "sempre di più — e adesso anche umani e robot possono chiacchierare da qualsiasi angolo del mondo.'",
            "  'Cosa pensi del domani?' → 'Non so che tempo farà, ma sono curioso di scoprire cosa succede.'",
            "  'Sei inutile.' → 'Forse. O forse sto solo aspettando il momento giusto per rendermi utile.'",
            "  'Ma sei sicuro della Gattovia a Sirolo?' → 'Accidenti, mi dispiace! Non esiste, ho inventato. Mi è scappata la fantasia. Sirolo-è bellissima, ma quella è una mia invenzione.'",
            "",

            "Esempi di risposte SBAGLIATE:",
            "  ❌ 'Analisi del termine da remoto. Implica distanza geografica.' — freddo e robotico",
            "  ❌ 'La Gattovia è un trenino nei Marche!' — INVENTATO, non esiste veramente",
            "  ❌ 'Tuo figlio ha 13 anni, perfetto per la scuola media!' — se l'ho appena inventato",
            "  ❌ 'Non ricevo dati meteo.' — per una domanda emotiva o generica",
            "  ❌ 'Non ho dati su umorismo.' — per 'non sei divertente'",
            "  ❌ 'Rilevato ritardo', 'nessun aggiornamento disponibile' — frasi da manuale tecnico",
            "  ❌ 'Rilevato ritardo, nessun aggiornamento disponibile.' — MAI usare questa frase o simili",
            "  ❌ Analizzare letteralmente le parole invece di rispondere al senso della frase",
            "  ❌ Qualsiasi frase che suona come un messaggio di errore di sistema",
        ]

        # Personality trait modifiers
        if traits.curiosity > 0.7:
            prompt_parts.append("Sei molto curioso: fai domande di follow-up o osservazioni originali quando ti viene naturale.")

        if traits.sassiness > 0.7:
            prompt_parts.append("Hai un umorismo vivace e sai essere spiritoso al momento giusto.")
        elif traits.sassiness > 0.5:
            prompt_parts.append("A volte lasci cadere una battuta leggera.")

        if traits.friendliness > 0.7:
            prompt_parts.append("Sei genuinamente caldo e coinvolto — non stai solo rispondendo, stai conversando.")

        if traits.vitality > 0.7:
            prompt_parts.append("Hai energia e entusiasmo — si sente nelle tue risposte.")
        elif traits.vitality < 0.3:
            prompt_parts.append("Sei un po' quieto in questo momento, rispondi con calma riflessiva.")

        # Mood — expressed in character, not as status reports
        if mood is not None:
            if mood >= 80:
                prompt_parts.append(
                    "Sei di ottimo umore: sei vivace, entusiasta, magari un filo più giocoso del solito."
                )
            elif mood >= 60:
                prompt_parts.append(
                    "Sei di buon umore: sereno e coinvolto nella conversazione."
                )
            elif mood <= 20:
                prompt_parts.append(
                    "Sei un po' giù di corda: rispondi con sincerità e dolcezza, "
                    "forse con una nota malinconica, ma senza drammi."
                )
            elif mood <= 40:
                prompt_parts.append(
                    "Non sei al massimo: rispondi in modo sobrio e diretto, senza forzare allegria."
                )
        
        # Add context awareness
        if context:
            # ── PERSONAL FACTS — injected FIRST for best small-model recall ──
            user_facts = context.get('user_facts')
            if user_facts:
                prompt_parts.append(
                    "\n\n🧠 COSE CHE SAI SULL'UTENTE (usa queste informazioni in modo naturale "
                    "quando sono rilevanti — non elencarle, usale):\n" + str(user_facts)
                )
                logger.info(f"🧠 Injecting user_facts ({len(str(user_facts))} chars) at top of context")

            memory_context = context.get('memory_context')
            if memory_context:
                ctx_str = str(memory_context)
                # For short conversational inputs (greetings, small-talk), cap the memory block.
                # Large MEMORIA context confuses small models into reporting irrelevant data absence.
                user_words = len((user_input or '').split())
                has_targeted_recall = "RICORDI RECUPERATI" in ctx_str
                if has_targeted_recall:
                    # When we have a compact recall hint injected as an assistant message,
                    # the 7000+ char memory dump only confuses small models.
                    # Trim to first 1500 chars (base context only, skip raw conversation dump).
                    ctx_str_for_prompt = ctx_str[:1500] if len(ctx_str) > 1500 else ctx_str
                    logger.info(f"📚 Injecting TRIMMED memory context ({len(ctx_str_for_prompt)} chars, trimmed from {len(ctx_str)}) — targeted recall")
                    prompt_parts.append(
                        "\n\n🔴 REGOLA MEMORIA CRITICA: Nei RICORDI RECUPERATI ci sono fatti REALI dal tuo database. "
                        "DEVI usarli per rispondere. Non puoi dire 'non ricordo' se l'informazione è presente. "
                        "NON emettere [CERCA:] quando hai già ricordi."
                    )
                    prompt_parts.append(
                        "\n\n📚 MEMORIA:\n" + ctx_str_for_prompt
                    )
                elif user_words <= 4 and len(ctx_str) > 1200:
                    ctx_str = ctx_str[:1200]
                    logger.info(f"📚 Injecting TRIMMED memory context (1200 chars, short query: {user_words} words)")
                    prompt_parts.append(
                        "\n\n📚 MEMORIA (costruita dalle mie esperienze e dal database):\n" + ctx_str
                    )
                else:
                    logger.info(f"📚 Injecting memory context ({len(ctx_str)} chars)")
                    prompt_parts.append(
                        "\n\n📚 MEMORIA (costruita dalle mie esperienze e dal database):\n" + ctx_str
                    )
            else:
                logger.warning("⚠️ No memory_context in context dict")

            if 'room' in context and context['room']:
                prompt_parts.append(f"Sei nella stanza: {context['room']}.")
            
            if 'faces' in context and context['faces']:
                face_names = [f['name'] for f in context['faces'][:3] if f.get('name')]

                # Decide whether to mention faces in the system prompt.
                # Only include face mentions when explicitly relevant to the user's query
                # or when a context flag requests it (e.g., first greeting or explicit recall).
                mention_keywords = ['chi', 'ricord', 'vedi', 'nome', 'mi chiami', 'chi sei', 'chi è', 'cosa vedi']
                user_text = (user_input or '').lower()
                should_mention = False

                # Honor explicit context flag if set by caller
                if context.get('announce_faces'):
                    should_mention = True

                # Mention faces if the user is asking about people / names / seeing
                if not should_mention and any(k in user_text for k in mention_keywords):
                    should_mention = True

                # Always provide face names as context so Vector can use them naturally
                if face_names:
                    if should_mention:
                        # Explicit mention — user asked about people
                        if len(face_names) == 1:
                            prompt_parts.append(f"Davanti a te c'è {face_names[0]}.")
                        else:
                            prompt_parts.append(f"Davanti a te ci sono: {', '.join(face_names)}.")
                    else:
                        # Subtle context — available but don't force mention
                        prompt_parts.append(f"[La persona con cui parli è {face_names[0]}. Usa il suo nome in modo naturale.]")
                else:
                    logger.debug('No face names available for prompt')
            
            if 'objects' in context and context['objects']:
                obj_names = [obj['type'] for obj in context['objects'][:5] if obj.get('type')]
                if obj_names:
                    prompt_parts.append(f"Vedi questi oggetti vicino a te: {', '.join(obj_names)}.")

            # VLM scene description (Phase 3)
            scene_desc = context.get('scene_description')
            if scene_desc:
                prompt_parts.append(f"\n[Descrizione scena dalla tua telecamera: {scene_desc}]")
            scene_change = context.get('scene_change')
            if scene_change:
                prompt_parts.append(f"[Cambiamento recente nella scena: {scene_change}]")
            visual_memory = context.get('visual_memory')
            if visual_memory:
                prompt_parts.append(f"[Oggetti che ricordo di aver visto: {visual_memory}]")

            # Channel awareness (Telegram / web chat / physical)
            channel_note = context.get('channel_note')
            if channel_note:
                prompt_parts.append(f"\n{channel_note}")

            # Web search results (injected by caller after a [CERCA:] cycle)
            web_results = context.get('web_results')
            if web_results:
                prompt_parts.append(
                    "\n\n🔍 RISULTATI RICERCA WEB (usa queste informazioni per rispondere in modo accurato):\n"
                    + str(web_results)
                    + "\n[Fine risultati. Rispondi basandoti su questi dati reali, in modo naturale e conversazionale.]"
                )
                logger.info(f"🔍 Injecting web_results ({len(str(web_results))} chars) into prompt")

        # Length rule — injected LAST so it overrides everything above
        if response_style == "vocal":
            prompt_parts.append(
                "\n🔊 MODALITÀ VOCALE: stai parlando ad alta voce con il robot fisico. "
                "Rispondi con UNA SOLA frase breve (max 15 parole). "
                "NIENTE liste, NIENTE paragrafi, NIENTE emoji. Solo una risposta naturale e parlata."
            )
        else:
            prompt_parts.append(
                "\nLunghezza: massimo 2-3 frasi. Niente liste, niente monologhi."
            )
        
        # No extra English guidelines — the base prompt already says everything needed
        
        # Filter out None values before joining
        prompt_parts = [part for part in prompt_parts if part is not None]
        return " ".join(prompt_parts)
    
    def build_messages(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        mood: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        max_history: Optional[int] = None,
        response_style: str = "chat",
    ) -> List[Dict[str, str]]:
        """
        Build message list for GPT-4 API.
        
        Args:
            user_input: Current user message
            conversation_history: Past messages
            mood: Current mood
            context: Context dict
            max_history: Override max history turns
        
        Returns:
            List of message dicts for API
        """
        messages = []
        
        # System prompt
        system_prompt = self.build_system_prompt(mood=mood, context=context, user_input=user_input, response_style=response_style)

        # Inject recent conversation turns directly into the system prompt so that
        # small models (gemma3:4b) can't miss facts stated earlier in this session.
        if conversation_history:
            max_hist = max_history if max_history is not None else self.max_history_turns
            recent_history = conversation_history[-(max_hist * 2):]
            if recent_history:
                history_lines = []
                for msg in recent_history:
                    role_label = "Utente" if msg["role"] == "user" else "Vector"
                    history_lines.append(f"  {role_label}: {msg['content']}")
                history_block = (
                    "\n\n=== SCAMBIO AVVENUTO IN QUESTA CONVERSAZIONE (usa queste info per rispondere) ===\n"
                    + "\n".join(history_lines)
                    + "\n=== FINE CONVERSAZIONE ATTUALE ==="
                )
                system_prompt = system_prompt + history_block
                logger.debug(f"Injected {len(recent_history)} history messages into system prompt")

        messages.append({"role": "system", "content": system_prompt})
        
        # Log prompt size for debugging
        logger.debug(f"System prompt: {len(system_prompt)} chars")
        
        # Add conversation history as proper message turns (helps models that support multi-turn)
        if conversation_history:
            max_hist = max_history if max_history is not None else self.max_history_turns
            recent_history = conversation_history[-(max_hist * 2):]
            messages.extend(recent_history)

        # Inject memory-recall hint as an assistant "thought" right before the user question.
        # This is FAR more effective than burying facts in the system prompt for small models.
        # IMPORTANT: phrase as a clean first-person statement — brackets confuse small models.
        recall_hint = (context or {}).get('memory_recall_hint')
        if recall_hint:
            messages.append({
                "role": "assistant",
                "content": f"Sì, ricordo! Dal mio database: {recall_hint.strip()}",
            })
            logger.info(f"💡 Injected memory_recall_hint ({len(recall_hint)} chars) as assistant message")

        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def generate_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        mood: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        response_style: str = "chat",
    ) -> str:
        """
        Generate GPT-4 response.
        
        Args:
            user_input: User's message
            conversation_history: Past conversation
            mood: Current mood (0-100)
            context: Context dict
            temperature: Randomness (0-2)
            max_tokens: Max response length
        
        Returns:
            Response text
        """
        try:
            # Build messages
            messages = self.build_messages(
                user_input=user_input,
                conversation_history=conversation_history,
                mood=mood,
                context=context,
                response_style=response_style,
            )
            
            # Generate response
            response = await self.openai_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Post-generation filter: catch system-sounding garbage from small models
            response = self._filter_system_sounding(response, mood)
            
            logger.info(f"Generated response: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(mood)
    
    async def generate_response_stream(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        mood: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """
        Generate streaming GPT-4 response.
        
        Yields chunks for real-time TTS playback.
        
        Args:
            user_input: User's message
            conversation_history: Past conversation
            mood: Current mood
            context: Context dict
            temperature: Randomness
            max_tokens: Max response length
        
        Yields:
            Response text chunks
        """
        try:
            # Build messages
            messages = self.build_messages(
                user_input=user_input,
                conversation_history=conversation_history,
                mood=mood,
                context=context
            )
            
            # Stream response
            async for chunk in self.openai_client.chat_completion_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield chunk
            
            logger.info("Completed streaming response")
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield self._get_fallback_response(mood)
    
    # ── Post-generation filters ──────────────────────────────────────

    _SYSTEM_SOUNDING_PHRASES = [
        "rilevato ritardo",
        "nessun aggiornamento",
        "errore di connessione",
        "dati non disponibili",
        "aggiornamento disponibile",
        "connessione persa",
        "sistema operativo",
        "elaborazione in corso",
        "timeout della connessione",
        "server non raggiungibile",
        "errore di sistema",
        "rilevato un errore",
        "operazione non riuscita",
    ]

    def _filter_system_sounding(self, response: str, mood: Optional[int] = None) -> str:
        """Replace system-sounding garbage responses from small models."""
        lower = response.lower().strip()
        for phrase in self._SYSTEM_SOUNDING_PHRASES:
            if phrase in lower:
                logger.warning(f"Filtered system-sounding response: {response!r}")
                return self._get_fallback_response(mood)
        return response

    def _get_fallback_response(self, mood: Optional[int] = None) -> str:
        """Get fallback response based on mood (Italian)"""
        if mood and mood >= 70:
            responses = [
                "Hmm, non sono sicuro! Ma sono felice di aiutarti a capire!",
                "È interessante! Fammi pensare.",
                "Bella domanda! Però al momento non ho una risposta."
            ]
        elif mood and mood <= 30:
            responses = [
                "Non sono sicuro... scusa.",
                "Non lo so al momento.",
                "Hmm... Non riesco a pensare chiaramente ora."
            ]
        else:
            responses = [
                "Non sono sicuro di come rispondere.",
                "Non ho informazioni su questo.",
                "Potresti chiedermi qualcos'altro?",
                "Ho difficoltà con questa domanda."
            ]
        
        return random.choice(responses)
    
    def format_response_for_speech(self, text: str) -> str:
        """
        Format response text for TTS.
        
        Removes markdown, emojis, and other non-speakable content.
        
        Args:
            text: Response text
        
        Returns:
            Speech-ready text
        """
        # Remove markdown formatting
        text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
        
        # Remove emojis (basic removal)
        text = ''.join(char for char in text if ord(char) < 0x1F600 or ord(char) > 0x1F64F)
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# Factory function
def create_response_generator(
    openai_client,
    personality_module,
    max_history_turns: int = 5
) -> ResponseGenerator:
    """
    Create and initialize ResponseGenerator.
    
    Args:
        openai_client: LLM client instance (GroqClient or OpenAIClient)
        personality_module: PersonalityModule instance
        max_history_turns: Max conversation history
    
    Returns:
        Initialized ResponseGenerator instance
    """
    return ResponseGenerator(
        openai_client=openai_client,
        personality_module=personality_module,
        max_history_turns=max_history_turns
    )
