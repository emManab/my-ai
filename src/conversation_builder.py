from whatsapp_parser import parse_whatsapp_chat


MY_NAME = "Manab"


def build_conversations(messages):
    """
    Converts individual WhatsApp messages into
    conversation examples.

    Each example contains:
        context  -> what the other person said
        response -> what you replied
    """

    conversations = []

    i = 0

    while i < len(messages):

        current = messages[i]

        # We only care about messages sent by the other person
        if current["is_me"] or current["type"] != "text":
            i += 1
            continue

        # Start collecting the other person's messages
        context_messages = []

        while (
            i < len(messages)
            and not messages[i]["is_me"]
            and messages[i]["type"] == "text"
        ):
            context_messages.append(messages[i]["message"])
            i += 1

        # If there is no response from us, skip it
        if i >= len(messages):
            break

        # Collect our consecutive replies
        response_messages = []

        while (
            i < len(messages)
            and messages[i]["is_me"]
            and messages[i]["type"] == "text"
        ):
            response_messages.append(messages[i]["message"])
            i += 1

        if not response_messages:
            continue

        context = "\n".join(context_messages)
        response = "\n".join(response_messages)

        conversations.append({
            "context": context,
            "response": response
        })

    return conversations


if __name__ == "__main__":

    messages = parse_whatsapp_chat("data/whatsapp.txt")

    conversations = build_conversations(messages)

    print(f"Total messages: {len(messages)}")
    print(f"Conversation examples: {len(conversations)}")

    print("\nFirst 10 examples:\n")

    for example in conversations[:10]:

        print("PERSON:")
        print(example["context"])

        print("\nMANAB:")
        print(example["response"])

        print("\n" + "-" * 50 + "\n")