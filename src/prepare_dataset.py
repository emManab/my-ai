import json
import re

from whatsapp_parser import parse_whatsapp_chat
from conversation_builder import build_conversations


INPUT_FILE = "data/whatsapp.txt"
OUTPUT_FILE = "data/rag_dataset.jsonl"


def has_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]"
    )

    return bool(emoji_pattern.search(text))


def prepare_dataset(conversations):

    prepared = []

    for conversation in conversations:

        context = conversation["context"].strip()
        response = conversation["response"].strip()

        # Don't keep completely empty examples
        if not context or not response:
            continue

        record = {
            "context": context,
            "response": response,

            "context_message_count": len(
                context.split("\n")
            ),

            "response_message_count": len(
                response.split("\n")
            ),

            "context_has_question": "?" in context,

            "response_has_question": "?" in response,

            "context_has_emoji": has_emoji(context),

            "response_has_emoji": has_emoji(response)
        }

        prepared.append(record)

    return prepared


def main():

    print("Loading WhatsApp chat...")

    messages = parse_whatsapp_chat(INPUT_FILE)

    print(f"Messages: {len(messages):,}")

    print("Building conversations...")

    conversations = build_conversations(messages)

    print(
        f"Conversation examples: {len(conversations):,}"
    )

    print("Preparing RAG dataset...")

    dataset = prepare_dataset(conversations)

    print(
        f"Final dataset: {len(dataset):,}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for record in dataset:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()