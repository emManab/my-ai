from whatsapp_parser import parse_whatsapp_chat
from collections import Counter
import re


MY_NAME = "Manab"


def analyze_style(messages):

    my_messages = [
        msg for msg in messages
        if msg["is_me"] and msg["type"] == "text"
    ]

    if not my_messages:
        return {}

    total_messages = len(my_messages)

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    lengths = [
        len(msg["message"])
        for msg in my_messages
    ]

    average_length = sum(lengths) / total_messages

    # --------------------------------------------------
    # Emoji detection
    # --------------------------------------------------

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]"
    )

    emoji_messages = sum(
        bool(emoji_pattern.search(msg["message"]))
        for msg in my_messages
    )

    # --------------------------------------------------
    # Questions
    # --------------------------------------------------

    question_messages = sum(
        "?" in msg["message"]
        for msg in my_messages
    )

    # --------------------------------------------------
    # Word extraction
    #
    # \w supports Unicode, so Bengali/Hindi characters
    # aren't automatically thrown away.
    # --------------------------------------------------

    words = []

    for msg in my_messages:

        extracted = re.findall(
            r"\b[\w]+\b",
            msg["message"].lower(),
            flags=re.UNICODE
        )

        # Remove extremely short/noisy tokens
        extracted = [
            word for word in extracted
            if len(word) >= 2
        ]

        words.extend(extracted)

    common_words = Counter(words).most_common(40)

    # --------------------------------------------------
    # Script detection
    # --------------------------------------------------

    english_chars = 0
    devanagari_chars = 0
    bengali_chars = 0

    for msg in my_messages:

        text = msg["message"]

        english_chars += len(
            re.findall(r"[A-Za-z]", text)
        )

        devanagari_chars += len(
            re.findall(r"[\u0900-\u097F]", text)
        )

        bengali_chars += len(
            re.findall(r"[\u0980-\u09FF]", text)
        )

    total_script_chars = (
        english_chars
        + devanagari_chars
        + bengali_chars
    )

    if total_script_chars > 0:

        english_percentage = (
            english_chars / total_script_chars * 100
        )

        hindi_percentage = (
            devanagari_chars / total_script_chars * 100
        )

        bengali_percentage = (
            bengali_chars / total_script_chars * 100
        )

    else:
        english_percentage = 0
        hindi_percentage = 0
        bengali_percentage = 0

    return {
        "total_messages": total_messages,

        "average_message_length": round(
            average_length, 2
        ),

        "emoji_message_percentage": round(
            emoji_messages / total_messages * 100,
            2
        ),

        "question_percentage": round(
            question_messages / total_messages * 100,
            2
        ),

        "english_script_percentage": round(
            english_percentage, 2
        ),

        "hindi_script_percentage": round(
            hindi_percentage, 2
        ),

        "bengali_script_percentage": round(
            bengali_percentage, 2
        ),

        "common_words": common_words
    }


if __name__ == "__main__":

    messages = parse_whatsapp_chat(
        "data/whatsapp.txt"
    )

    stats = analyze_style(messages)

    print("\n===== YOUR TEXTING STYLE =====\n")

    for key, value in stats.items():

        print(f"{key}: {value}")