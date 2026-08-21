import json

from whatsapp_parser import parse_whatsapp_chat
from conversation_builder import build_conversations


INPUT_FILE = "data/whatsapp.txt"
OUTPUT_FILE = "data/conversations.jsonl"


def main():

    print("Loading WhatsApp chat...")

    messages = parse_whatsapp_chat(INPUT_FILE)

    print(f"Loaded {len(messages):,} messages")

    print("Building conversation examples...")

    conversations = build_conversations(messages)

    print(
        f"Built {len(conversations):,} conversation examples"
    )

    print("Saving dataset...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for conversation in conversations:

            # Skip empty examples
            if not conversation["context"].strip():
                continue

            if not conversation["response"].strip():
                continue

            record = {
                "context": conversation["context"],
                "response": conversation["response"]
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"\nDataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()