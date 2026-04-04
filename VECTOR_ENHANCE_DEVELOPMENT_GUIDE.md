---

### 🤖 Project Brief: Vector 2.0 - Local AI Autonomous Agent

**Context:**
I am developing an advanced AI layer for an **Anki Vector** robot using the **Python SDK** and **wire-pod**. The goal is to transform Vector from a simple toy into a "living" autonomous agent by moving all processing from the cloud to a powerful local workstation.

**Hardware Environment:**

- **GPU:** NVIDIA GeForce RTX 5070 Ti 16GB VRAM (OC).
- **RAM:** 32GB DDR5 6000MHz.
- **Platform:** Local AI via **Ollama** (LLM/VLM) and dedicated Python services.

**Key Objectives:**

1.  **Full Local LLM Integration:** Transition the core reasoning engine from cloud APIs (OpenAI/Groq) to local models via **Ollama** (e.g., Llama 3 8B, Mistral). The focus is on ultra-low latency (<1.5s) and data privacy.
Raccomended LLM:
- mistral:7b-instruct (~4.1GB) — good Italian, fast
- llama3.1:8b (~4.7GB) — excellent reasoning
- gemma3:4b (~2.5GB) — ultra-fast if latency is critical (already fetched)
2.  **Continuous Social Listening (No Wake Word):** Move away from the "Hey Vector" trigger. Implement a continuous audio streaming buffer that uses **Local Whisper (STT)** to transcribe speech and an LLM-based filter to determine context. Vector should autonomously decide when to engage in a conversation, when to ignore background noise, or when to proactively interject based on the discussion.
3.  **Local Multimodal Vision (VLM):** Implement a local Vision-Language Model (e.g., **Llava** or **Moondream2**) to process Vector’s camera feed. Instead of simple object detection, the robot must perform "Continuous Scene Description," allowing it to comment on changes in the environment (e.g., "I see you're drinking coffee now") without user prompts.
4.  **Episodic Memory & RAG:** Replace basic SQLite storage with a **Vector Database (ChromaDB or FAISS)**. Vector must store and retrieve "memories" of faces, past conversations, and observed objects using Retrieval-Augmented Generation (RAG) to maintain long-term relationships and context (already done but m aybe to optimize)..
5.  **Dynamic Personality & Animation Mapping:** Utilize a "Personality Engine" that calculates Vector’s emotional state. This state must programmatically select and trigger specific **Vector Animation IDs** (mapped from the SDK) and influence the **TTS (XTTSv2)** tone to match the robot's current mood (e.g., sad, excited, curious).
6.  **Hybrid Remote Connectivity:** Create a bridge to a **Telegram Bot**. This allows for remote interaction with the AI's "brain" even when the physical Vector is powered off, creating a persistent digital entity that lives on the PC but manifests through the robot when available.
7.  **Optimized VRAM Management:** Ensure the Python orchestration layer manages VRAM efficiently across the LLM, VLM, and TTS services to allow for simultaneous background tasks (coding, web browsing) without saturating the GPU.

---

### Notes:

In this markdown file there are all the changes i want to do.
This does not all affect only the code here in this project but also operations i have to do on my pc like downloading models, software etc.. you should guiide me in those activities too
