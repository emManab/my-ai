# Manab AI 🦋

### A Personal AI Texting Assistant Powered by RAG, BGE-M3 & Cloud LLMs

Manab AI is a personalized AI texting assistant designed to generate replies that match a user's natural texting style.

Instead of relying only on an LLM's general knowledge, Manab AI retrieves relevant examples from previous conversations using **Retrieval-Augmented Generation (RAG)** and provides those examples to the language model as context.

The goal is simple:

> **Generate replies that feel like something I would actually type.**

---

## ✨ Features

- 🧠 Personalized conversational AI
- 🔎 Retrieval-Augmented Generation (RAG)
- 🧩 BGE-M3 semantic embeddings
- 🗄️ ChromaDB vector database
- ☁️ Cloud-based LLM inference
- 🆓 OpenRouter free-model support
- 🔄 Automatic fallback between free models
- 💬 WhatsApp-style conversation generation
- 🇮🇳 Natural English + Hinglish support
- 😭 Emoji and texting-pattern awareness
- 👤 Personal profile/context support
- ⚡ CUDA GPU acceleration for embeddings
- 🌐 Streamlit web interface
- 💻 Terminal-based chat interface
- 🔐 Environment-variable based API key management

---

# 🧠 How Manab AI Works

The system uses a Retrieval-Augmented Generation pipeline.

```text
                     User Message
                          │
                          ▼
                ┌──────────────────┐
                │     BGE-M3       │
                │  Query Embedding │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     ChromaDB     │
                │ Semantic Search  │
                └────────┬─────────┘
                         │
                         ▼
                 Relevant Memories
                         │
                         ▼
                ┌──────────────────┐
                │   Prompt Builder │
                │                  │
                │ • Memories       │
                │ • Profile        │
                │ • Chat History   │
                │ • Style Rules    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    OpenRouter    │
                │   Cloud LLM API  │
                └────────┬─────────┘
                         │
                         ▼
                  Generated Reply