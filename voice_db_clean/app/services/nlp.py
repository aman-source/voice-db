import re

def extract_transaction_info(text: str):
    """
    Extract sender, receiver, and amount from transaction sentences.

    Examples:
    - 'sumanth paid 5000 to aman'
    - 'sumanth send 1000 to aman'
    - 'transfer 500 from sumanth to aman'
    - 'sumanth sent 500' (incomplete - no receiver)
    """
    text = text.lower().strip()
    words = text.split()

    if not words:
        return {"sender": None, "amount": None, "receiver": None}

    # Extract amount (any number in the text)
    amount_match = re.search(r"\b(\d+)\b", text)
    amount = int(amount_match.group(1)) if amount_match else None

    sender = None
    receiver = None

    action_verbs = ["paid", "pay", "send", "sent", "transfer", "transferred", "gave", "give"]

    # Pattern 1: "transfer X from [sender] to [receiver]"
    if "from" in words and "to" in words:
        from_idx = words.index("from")
        to_idx = words.index("to")
        if from_idx < to_idx and from_idx + 1 < len(words):
            sender = words[from_idx + 1]
        if to_idx + 1 < len(words):
            receiver = words[to_idx + 1]

    # Pattern 2: "[sender] paid/send/transfer X to [receiver]"
    elif "to" in words:
        to_idx = words.index("to")
        if to_idx + 1 < len(words):
            receiver = words[to_idx + 1]

        # Find sender (word before action verb)
        for verb in action_verbs:
            if verb in words:
                verb_idx = words.index(verb)
                if verb_idx > 0:
                    sender = words[verb_idx - 1]
                    break

    # Pattern 3: "[sender] sent/paid X" (no receiver mentioned)
    else:
        for verb in action_verbs:
            if verb in words:
                verb_idx = words.index(verb)
                if verb_idx > 0:
                    sender = words[verb_idx - 1]
                # Check if there's a word after the amount that could be receiver
                if amount_match:
                    amount_end = amount_match.end()
                    remaining = text[amount_end:].strip().split()
                    if remaining:
                        # Filter out common non-name words
                        skip_words = ["rupees", "rs", "dollars", "to", "for", "the", "a", "an"]
                        for word in remaining:
                            if word not in skip_words and not word.isdigit():
                                receiver = word
                                break
                break

    # Fallback: first word might be sender if it's a name-like word
    if not sender and len(words) > 0:
        first_word = words[0]
        if first_word not in action_verbs and not first_word.isdigit():
            # Check if it looks like a name (not a common word)
            common_words = ["i", "we", "the", "a", "an", "please", "can", "could", "want"]
            if first_word not in common_words:
                sender = first_word

    return {
        "sender": sender,
        "amount": amount,
        "receiver": receiver
    }
