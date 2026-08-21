import json


FILE = "data/conversations.jsonl"


with open(FILE, "r", encoding="utf-8") as file:

    for i, line in enumerate(file):

        conversation = json.loads(line)

        print("\nCONVERSATION", i + 1)

        print("\nPERSON:")
        print(conversation["context"])

        print("\nMANAB:")
        print(conversation["response"])

        print("\n" + "-" * 60)

        if i >= 9:
            break