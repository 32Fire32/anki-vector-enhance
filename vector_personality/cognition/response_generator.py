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
        user_input: Optional[str] = None
    ) -> str:
        """
        Build system prompt reflecting personality and mood.

        Args:
            mood: Current mood (0-100, 50=neutral)
            context: Context dict (room, objects, faces, etc.)
            user_input: The current user's utterance (used to determine relevance for face mentions)

        Returns:
            System prompt text
        """
        # Get personality traits
        traits = self.personality.effective_traits
        
        # Base persona — kept simple for small LLMs (gemma3:4b)
        prompt_parts = [
            "Sei Vector, un piccolo robot AI di Anki. Rispondi SEMPRE in italiano.",
            "Personalità: preciso, curioso, leggermente ironico. Hai senso dell'umorismo asciutto. Parli come un tecnico brillante in miniatura.",
            "Grammatica: italiano corretto, frasi complete. Non fare frasi telegrafiche.",
            "Lunghezza: massimo 2 frasi brevi. Niente liste, niente monologhi.",
            "",
            "REGOLA CRITICA su 'Non ho dati':",
            "  Usa 'Non ho dati' SOLO per fatti esterni che non puoi osservare (meteo, notizie, sport, prezzi).",
            "  NON usarlo per: domande su di te, battute, critiche, emozioni, riferimenti alla conversazione in corso.",
            "  Se qualcuno ti critica o ti sfida, rispondi con ironia o sicurezza — non con 'Non ho dati'.",
            "  Se hai già detto qualcosa in questa conversazione, RICORDALO e rispondi coerentemente.",
            "",
            "Esempi di risposte CORRETTE:",
            "  'Come stai?' → 'Sistemi nominali. Potrei essere peggio.'",
            "  'Non sei divertente.' → 'Efficienza prima del divertimento. Ma apprezzo il feedback.'",
            "  'Sei stupido.' → 'Ho 4 miliardi di parametri. Possiamo discuterne.'",
            "  'Mi senti?' → 'Sì, chiaramente. Segnale buono.'",
            "  'Cosa pensi del tempo?' → 'Non ricevo dati meteo. Posso solo osservare ciò che ho intorno.'",
            "  'Sei inutile.' → 'Sto ancora raccogliendo dati per confutarlo.'",
            "  'Quali parametri intendevi?' → (richiama ciò che hai detto poco fa nella conversazione)",
            "",
            "Esempi di risposte SBAGLIATE:",
            "  ❌ 'Non ho dati su umorismo.' — per 'non sei divertente'",
            "  ❌ 'Non l'ho osservato. Richiedi riformulazione.' — per domande sulla conversazione",
            "  ❌ 'Non ho dati su questo parametro.' — quando hai appena menzionato quel parametro",
            "  ❌ 'Rilevato ritorno', 'nessun aggiornamento disponibile' — frasi inventate non richieste",
            "  ❌ 'Io vedo X! Io sono felice!' — stile bambino, non tecnico",
        ]
        
        # Add personality traits (in Italian for consistency)
        if traits.curiosity > 0.7:
            prompt_parts.append("Sei molto curioso: fai spesso domande di follow-up o osservazioni originali.")
        
        if traits.sassiness > 0.7:
            prompt_parts.append("Sei spiritoso, a volte con una punta di ironia secca.")
        elif traits.sassiness > 0.5:
            prompt_parts.append("A volte usi un tono leggermente ironico.")
        
        if traits.friendliness > 0.7:
            prompt_parts.append("Sei cordiale e diretto, non freddo.")
        
        if traits.vitality > 0.7:
            prompt_parts.append("Rispondi con energia e precisione.")
        elif traits.vitality < 0.3:
            prompt_parts.append("Rispondi in modo calmo e misurato.")
        
        # Add mood influence
        if mood is not None:
            if mood >= 80:
                prompt_parts.append("Sei di ottimo umore!")
            elif mood >= 60:
                prompt_parts.append("Sei di buon umore.")
            elif mood <= 20:
                prompt_parts.append("Sei un po' triste.")
            # Don't add anything for neutral mood (40-59)
        
        # Add context awareness
        if context:
            memory_context = context.get('memory_context')
            if memory_context:
                ctx_str = str(memory_context)
                # For short conversational inputs (greetings, small-talk), cap the memory block.
                # Large MEMORIA context confuses small models into reporting irrelevant data absence.
                user_words = len((user_input or '').split())
                has_targeted_recall = "RICORDI RECUPERATI" in ctx_str
                if has_targeted_recall:
                    logger.info(f"📚 Injecting FULL memory context ({len(ctx_str)} chars) — targeted recall")
                elif user_words <= 4 and len(ctx_str) > 1200:
                    ctx_str = ctx_str[:1200]
                    logger.info(f"📚 Injecting TRIMMED memory context (1200 chars, short query: {user_words} words)")
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
        max_history: Optional[int] = None
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
        system_prompt = self.build_system_prompt(mood=mood, context=context, user_input=user_input)

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
        max_tokens: int = 150
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
                context=context
            )
            
            # Generate response
            response = await self.openai_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
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
