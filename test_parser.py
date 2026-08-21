from src.whatsapp_parser import parse_whatsapp_chat


messages = parse_whatsapp_chat("data/whatsapp.txt")

print(f"Total messages: {len(messages)}")

print("\nFirst 10 messages:\n")

for message in messages[:10]:
    print(message)