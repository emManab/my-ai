import re
from pathlib import Path


MY_NAME = "Manab"


def clean_text(text):
    """
    Cleans invisible/special whitespace characters
    commonly found in WhatsApp exports.
    """

    text = text.replace("\u202f", " ")
    text = text.replace("\xa0", " ")

    # Collapse repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def detect_message_type(message):
    """
    Detects whether a WhatsApp message is text or media.
    """

    if message == "<Media omitted>":
        return "media"

    return "text"


def parse_whatsapp_chat(file_path):
    """
    Reads a WhatsApp exported chat and converts it
    into structured messages.
    """

    messages = []

    text = Path(file_path).read_text(
        encoding="utf-8-sig"
    )

    pattern = re.compile(
        r"^(\d{2}/\d{2}/\d{2}),\s+"
        r"(\d{1,2}:\d{2}\s*[ap]m)\s+-\s+"
        r"([^:]+):\s?(.*)$",
        re.IGNORECASE
    )

    for line in text.splitlines():

        match = pattern.match(line)

        if not match:
            continue

        date, time, sender, message = match.groups()

        message = clean_text(message)
        sender = clean_text(sender)
        time = clean_text(time)

        messages.append({
            "date": date,
            "time": time,
            "sender": sender,
            "message": message,
            "type": detect_message_type(message),
            "is_me": sender == MY_NAME
        })

    return messages