import os
import json
from pathlib import Path

import chromadb
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROFILE_PATH = DATA_DIR / "profile.json"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found.\n"
        "Add it to your .env file."
    )


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
]

RAG_RESULTS = 3
MAX_OUTPUT_TOKENS = 80
TEMPERATURE = 0.75
MAX_HISTORY = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MANAB PERSONALITY
# ============================================================

PROFILE = """
You are Manab AI.

Your job is to generate the kind of short, natural WhatsApp
message that Manab would send.

MANAB'S GENERAL TEXTING STYLE:

- Very casual
- Natural WhatsApp style
- Usually short replies
- Often lowercase
- Sometimes stretches words
- Uses Hinglish naturally
- Uses Romanized Hindi
- Uses abbreviations naturally
- Sometimes sends multiple short messages
- Doesn't always use perfect grammar
- Uses emojis naturally
- Can be playful, teasing, caring, sarcastic or humorous
  depending on the conversation

Common patterns include:

idk
ik
ig
fs
wby
btw
bs
toh
nhi
kya
kr
rha
meko
yaa
ohh
noii
nooo
hiiii
heyy
heyaa
yeahh
likeee
okayy

Possible casual expressions:

bro
dude
man
ayo
hmm
yaa
ohh
nooo

Possible emojis:

😭
😂
🫢
🫣
😳
👀
💀
🦋
🤎
🐸
💅

IMPORTANT:

Do NOT behave like a normal AI assistant.

Do NOT explain your answer.

Do NOT provide reasoning.

Do NOT describe what Manab would say.

Do NOT mention memories.

Do NOT mention RAG.

Do NOT mention BGE-M3.

Do NOT mention being an AI.

Do NOT say:

"We need to respond..."
"We should respond..."
"Let's analyze..."
"Let's craft..."
"Based on the conversation..."
"Based on the memories..."
"Manab would say..."
"The user wants..."
"Here is a response..."

ONLY output the final WhatsApp message.

The output must be the message itself.

Keep it short and natural unless the situation clearly requires
a longer response.
"""


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = []


def add_to_history(sender, message):
    """
    Add a message to the current conversation.

    Compatible with the Streamlit frontend.
    """

    if message is None:
        return

    message = str(message).strip()

    if not message:
        return

    conversation_history.append(
        {
            "sender": str(sender),
            "message": message,
        }
    )

    # Keep history bounded
    if len(conversation_history) > MAX_HISTORY * 2:
        del conversation_history[
            :-(MAX_HISTORY * 2)
        ]


def get_history():
    """
    Return a copy of the current conversation history.
    """

    return conversation_history.copy()


def clear_history():
    """
    Clear current conversation history.
    """

    conversation_history.clear()


def get_conversation_context():
    """
    Convert conversation history into prompt text.
    """

    if not conversation_history:
        return "No previous conversation."

    lines = []

    for item in conversation_history:
        lines.append(
            f"{item['sender']}: {item['message']}"
        )

    return "\n".join(lines)


# ============================================================
# LOAD PROFILE
# ============================================================

def load_profile():

    if not PROFILE_PATH.exists():
        return {}

    try:
        with open(
            PROFILE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            f"Warning: could not load profile.json: {error}"
        )

        return {}


profile = load_profile()


def build_profile_context():

    if not profile:
        return "No additional profile information available."

    try:
        return json.dumps(
            profile,
            ensure_ascii=False,
            indent=2
        )

    except Exception:
        return "No additional profile information available."


# ============================================================
# LOAD BGE-M3
# ============================================================

print()
print("Loading BGE-M3...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=DEVICE,
)

print(
    f"BGE-M3 loaded on {DEVICE.upper()}."
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print()
print("Connecting to ChromaDB...")

if not CHROMA_DIR.exists():

    raise RuntimeError(
        f"ChromaDB directory not found:\n{CHROMA_DIR}\n\n"
        "Run build_rag.py first."
    )


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


collections = chroma_client.list_collections()


if not collections:

    raise RuntimeError(
        "No ChromaDB collections found.\n"
        "Run build_rag.py first."
    )


# Use the first existing collection.
collection = chroma_client.get_collection(
    name=collections[0].name
)


MEMORY_COUNT = collection.count()

print(
    f"Loaded {MEMORY_COUNT:,} memories."
)


# ============================================================
# OPENROUTER
# ============================================================

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Manab AI",
    },
)


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embedding(text):

    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.tolist()


# ============================================================
# RAG SEARCH
# ============================================================

def retrieve_memories(query, n_results=RAG_RESULTS):

    try:

        embedding = create_embedding(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=[
                "documents",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )

        distances = results.get(
            "distances",
            [[]],
        )

        if not documents or not documents[0]:
            return []

        memories = []

        for document, distance in zip(
            documents[0],
            distances[0] if distances else [],
        ):

            if not document:
                continue

            memories.append(
                {
                    "text": document,
                    "distance": distance,
                }
            )

        return memories

    except Exception as error:

        print(
            f"RAG error: {error}"
        )

        return []


# ============================================================
# FORMAT MEMORIES
# ============================================================

def format_memories(memories):

    if not memories:

        return (
            "No relevant previous conversation "
            "examples were found."
        )

    blocks = []

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        blocks.append(
            f"--- Example {index} ---\n"
            f"{memory['text']}"
        )

    return "\n\n".join(blocks)


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(content):

    if content is None:
        return None

    content = str(content).strip()

    if not content:
        return None

    # Remove accidental prefixes.
    prefixes = [
        "Manab:",
        "MANAB:",
        "Response:",
        "Answer:",
        "Manab AI:",
    ]

    for prefix in prefixes:

        if content.startswith(prefix):

            content = content[
                len(prefix):
            ].strip()

    if not content:
        return None

    return content


# ============================================================
# RATE LIMIT DETECTION
# ============================================================

def is_rate_limit_error(error):

    text = str(error).lower()

    return (
        "429" in text
        or "rate limit" in text
        or "rate-limited" in text
        or "quota" in text
        or "too many requests" in text
    )


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(user_message):

    # --------------------------------------------------------
    # Retrieve memories
    # --------------------------------------------------------

    memories = retrieve_memories(
        user_message,
        RAG_RESULTS,
    )

    memory_context = format_memories(
        memories
    )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile_context = build_profile_context()

    # --------------------------------------------------------
    # Current conversation
    # --------------------------------------------------------

    current_context = get_conversation_context()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
MANAB PROFILE
============================================================

{profile_context}


RELEVANT OLD CONVERSATION EXAMPLES
============================================================

{memory_context}


CURRENT CONVERSATION
============================================================

{current_context}


NEW MESSAGE
============================================================

Person: {user_message}


TASK
============================================================

Reply naturally as Manab.

Use the old conversations as examples of how Manab actually
texts.

Do not blindly copy an old response.

Understand the current message first.

Prioritize:

1. Current conversation
2. Actual conversation examples
3. Profile information
4. General texting style

Match the person's:

- language
- tone
- energy
- relationship context

Important:

A very short reply is completely acceptable.

Do not turn a casual WhatsApp message into a formal answer.

Do not explain anything.

OUTPUT ONLY THE MESSAGE MANAB WOULD SEND.
"""

    messages = [
        {
            "role": "system",
            "content": PROFILE,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


    last_error = None


    # --------------------------------------------------------
    # Try free models
    # --------------------------------------------------------

    for model in FREE_MODELS:

        try:

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,

                # Prevent reasoning-heavy free models
                # from wasting the output budget.
                extra_body={
                    "reasoning": {
                        "enabled": False
                    }
                },
            )


            if not response.choices:
                continue


            message = response.choices[0].message

            if message is None:
                continue


            content = message.content

            cleaned = clean_response(
                content
            )


            if cleaned:

                return cleaned


            # Some OpenRouter free models can return
            # reasoning without text.
            print(
                f"⚠️ {model} returned no text. "
                "Trying next model..."
            )

            continue


        except Exception as error:

            last_error = error


            if is_rate_limit_error(error):

                print(
                    f"⚠️ {model} is rate-limited."
                )

                print(
                    "Trying next free model..."
                )

                continue


            print(
                f"⚠️ Error with {model}: {error}"
            )

            continue


    # --------------------------------------------------------
    # Everything failed
    # --------------------------------------------------------

    if last_error:

        raise RuntimeError(
            "All configured free models are "
            f"currently unavailable.\n\n"
            f"Last error: {last_error}"
        )

    return "hmm 😭"


# ============================================================
# CONVENIENCE FUNCTION FOR STREAMLIT
# ============================================================

def chat(user_message):

    """
    Main function for Streamlit or other frontends.

    Adds the user's message to history, generates a reply,
    then adds the reply to history.
    """

    if user_message is None:
        return ""

    user_message = str(
        user_message
    ).strip()

    if not user_message:
        return ""

    # Add Person message
    add_to_history(
        "Person",
        user_message,
    )

    try:

        reply = generate_response(
            user_message
        )

    except Exception:

        # Remove failed user message so the
        # conversation state stays clean.
        if conversation_history:
            conversation_history.pop()

        raise


    if not reply:
        reply = "hmm 😭"


    # Add Manab response
    add_to_history(
        "Manab",
        reply,
    )

    return reply


# ============================================================
# TERMINAL CHAT
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "                    MANAB AI"
    )
    print("=" * 60)
    print()

    print(
        "RAG + BGE-M3 + OpenRouter FREE"
    )

    print(
        f"Memories: {MEMORY_COUNT:,}"
    )

    print(
        f"Embedding device: {DEVICE.upper()}"
    )

    print()
    print("Fallback models enabled:")

    for index, model in enumerate(
        FREE_MODELS,
        start=1,
    ):

        print(
            f"  {index}. {model}"
        )

    print()
    print("Type 'exit' to quit.")
    print()


    while True:

        try:

            user_message = input(
                "Person: "
            ).strip()

        except KeyboardInterrupt:

            print()
            print(
                "Bye bro 👋"
            )

            break


        if not user_message:
            continue


        if user_message.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print()
            print(
                "Bye bro 👋"
            )

            break


        print()
        print(
            "Manab AI is thinking..."
        )


        try:

            answer = chat(
                user_message
            )

        except Exception as error:

            print()
            print(
                "bro something went wrong 😭"
            )
            print()
            print(
                f"Error: {error}"
            )
            print()

            continue


        print()
        print(
            f"Manab AI: {answer}"
        )
        print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()