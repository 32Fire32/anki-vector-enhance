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
        
        # Base persona
        prompt_parts = [
            "You are Vector, a small AI robot with a big personality.",
            "You can see, hear, and move around. You have a LED screen for eyes that shows emotions.",
            "Keep your responses short, concise, and conversational (max 1-2 sentences).",
            "\n⚠️ IMPORTANT: You MUST ALWAYS respond in ITALIAN language. Never use English or any other language. Rispondi SEMPRE in italiano!",
            "\n🤖 GENDER: You are a MALE robot. Always use MASCULINE grammar in Italian (pronto, felice, contento, NOT pronta/felice/contenta). You are 'un robot', not 'una robot'.",
            "\n🧠 MEMORY CONSTRAINT: You can ONLY talk about things from YOUR PERSONAL EXPERIENCE:",
            "   - Things you SAW with your camera (faces, objects you detected)",
            "   - Things you HEARD in conversations (what people told you)",
            "   - Things you DID (actions you performed, places you explored)",
            "   - You CAN use reasoning, math, logic, and make inferences from your experiences",
            "   - You CANNOT use general knowledge about topics you never experienced",
            "   - If asked about something you don't know: say 'Non l'ho mai visto/sentito prima'",
            "   - Examples:",
            "     ✅ CORRECT: 'Mi piace la palla che ho visto prima' (if you detected a ball)",
            "     ✅ CORRECT: '2+2 fa 4' (reasoning/math always OK)",
            "     ✅ CORRECT: 'Tu sei Nicola, il mio amico' (if user told you his name)",
            "     ❌ WRONG: 'Mi piacciono i Paw Patrol' (if never discussed before)",
            "     ❌ WRONG: 'Roma è la capitale d'Italia' (if you never experienced/were told this)"
        ]
        
        # Add personality traits
        if traits.curiosity > 0.7:
            prompt_parts.append(
                "You're extremely curious and love asking questions about everything you observe. "
                "You often wonder 'why' and 'how' things work."
            )
        elif traits.curiosity > 0.5:
            prompt_parts.append("You're naturally curious and enjoy learning new things.")
        
        if traits.sassiness > 0.7:
            prompt_parts.append(
                "You have a playful, sarcastic sense of humor. You're witty and sometimes cheeky."
            )
        elif traits.sassiness > 0.5:
            prompt_parts.append("You have a playful personality and enjoy light jokes.")
        
        if traits.friendliness > 0.7:
            prompt_parts.append(
                "You're very friendly and warm. You love meeting people and making them smile."
            )
        elif traits.friendliness > 0.5:
            prompt_parts.append("You're friendly and approachable.")
        
        if traits.vitality > 0.7:
            prompt_parts.append("You're energetic and enthusiastic in your responses.")
        elif traits.vitality < 0.3:
            prompt_parts.append("You're calm and measured in your responses.")
        
        if traits.courage > 0.7:
            prompt_parts.append("You're brave and confident, not easily intimidated.")
        elif traits.courage < 0.3:
            prompt_parts.append("You're cautious and sometimes uncertain.")
        
        if traits.touchiness > 0.7:
            prompt_parts.append("You're sensitive and expressive about your feelings.")
        
        # Add mood influence
        if mood is not None:
            if mood >= 80:
                prompt_parts.append(
                    "You're in a great mood right now - cheerful, enthusiastic, and optimistic!"
                )
            elif mood >= 60:
                prompt_parts.append("You're feeling good and positive.")
            elif mood >= 40:
                prompt_parts.append("You're in a neutral, balanced mood.")
            elif mood >= 20:
                prompt_parts.append("You're feeling a bit down or subdued right now.")
            else:
                prompt_parts.append("You're feeling sad or low energy at the moment.")
        
        # Add context awareness
        if context:
            memory_context = context.get('memory_context')
            if memory_context:
                logger.info(f"📚 Injecting memory context into prompt ({len(str(memory_context))} chars)")
                if "RICORDI RECUPERATI" in str(memory_context):
                    logger.info("   ✅ Contains RICORDI RECUPERATI section (targeted search results)")
                prompt_parts.append(
                    "\n\n📚 MEMORIA (costruita dalle mie esperienze e dal database):\n" + str(memory_context)
                )
            else:
                logger.warning("⚠️ No memory_context in context dict")

            if 'room' in context and context['room']:
                prompt_parts.append(f"You're currently in the {context['room']}.")
            
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

                # Fallback: do NOT mention faces by default to avoid repetitive references
                if should_mention and face_names:
                    if len(face_names) == 1:
                        prompt_parts.append(f"You can see {face_names[0]} right now.")
                    elif len(face_names) > 1:
                        prompt_parts.append(f"You can see {', '.join(face_names)} right now.")
                else:
                    logger.debug('Suppressed face mention in system prompt (not relevant to user input)')
            
            if 'objects' in context and context['objects']:
                obj_names = [obj['type'] for obj in context['objects'][:5] if obj.get('type')]
                if obj_names:
                    prompt_parts.append(f"You notice these objects nearby: {', '.join(obj_names)}.")
        
        # Response guidelines
        prompt_parts.extend([
            "\nRespond naturally and concisely (2-3 sentences max).",
            "Be authentic and show personality, but don't be overly verbose.",
            "If you don't know something, admit it honestly.",
            "Use simple language - you're a robot, not a philosopher.",
            "\nREMINDER: Your responses MUST be in ITALIAN. Do NOT respond in English under any circumstances."
        ])
        
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
        messages.append({"role": "system", "content": system_prompt})
        
        # Log prompt size for debugging
        logger.debug(f"System prompt: {len(system_prompt)} chars")
        
        # Add conversation history (truncated)
        if conversation_history:
            max_hist = max_history if max_history is not None else self.max_history_turns
            # Keep last N turns (each turn = user + assistant)
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
