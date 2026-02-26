import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

from google import genai

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_REGION", "us-central1"),
        )
    return _client


def extract_transaction_info(text: str):
    """
    Use Gemini 2.5 Flash via Vertex AI to extract sender, receiver, and amount
    from a transaction sentence.
    """
    if not text or not text.strip():
        return {"sender": None, "amount": None, "receiver": None}

    prompt = f"""Extract the sender name, receiver name, and amount from this transaction sentence.
Return ONLY a JSON object with keys "sender", "receiver", "amount".
Use null if a value is not mentioned.
Amount must be a number (integer), not a string.

Sentence: "{text}"

JSON:"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        sender   = data.get("sender")
        receiver = data.get("receiver")
        amount   = data.get("amount")

        if isinstance(sender, str):
            sender = sender.lower().strip() or None
        if isinstance(receiver, str):
            receiver = receiver.lower().strip() or None
        if isinstance(amount, str):
            amount = int(re.sub(r"\D", "", amount)) if re.search(r"\d", amount) else None

        print(f"[OK] Gemini NLP: sender={sender}, receiver={receiver}, amount={amount}")
        return {"sender": sender, "amount": amount, "receiver": receiver}

    except Exception as e:
        print(f"[ERROR] Gemini NLP failed: {e} — falling back to rule-based")
        return _rule_based_fallback(text)


def _rule_based_fallback(text: str):
    text = text.lower().strip()
    words = text.split()

    if not words:
        return {"sender": None, "amount": None, "receiver": None}

    amount_match = re.search(r"\b(\d+)\b", text)
    amount = int(amount_match.group(1)) if amount_match else None

    sender = None
    receiver = None
    action_verbs = ["paid", "pay", "send", "sent", "transfer", "transferred", "gave", "give"]
    common_words = {"i", "we", "the", "a", "an", "please", "can", "could", "want",
                    "to", "from", "for", "of", "in", "on", "at", "and", "or"}

    if "from" in words and "to" in words:
        from_idx = words.index("from")
        to_idx = words.index("to")
        if from_idx < to_idx and from_idx + 1 < len(words):
            sender = words[from_idx + 1]
        if to_idx + 1 < len(words):
            receiver = words[to_idx + 1]
    elif "to" in words:
        to_idx = words.index("to")
        if to_idx + 1 < len(words):
            receiver = words[to_idx + 1]
        for verb in action_verbs:
            if verb in words:
                verb_idx = words.index(verb)
                if verb_idx > 0:
                    sender = words[verb_idx - 1]
                    break
    else:
        for verb in action_verbs:
            if verb in words:
                verb_idx = words.index(verb)
                if verb_idx > 0:
                    sender = words[verb_idx - 1]
                break

    if not sender and words:
        first = words[0]
        if first not in action_verbs and not first.isdigit() and first not in common_words:
            sender = first

    if not receiver and amount_match:
        amount_str = str(amount)
        if amount_str in words:
            amt_idx = words.index(amount_str)
            candidates = [
                w for w in words[amt_idx + 1:]
                if w not in common_words and not w.isdigit() and w not in action_verbs and len(w) >= 2
            ]
            if candidates:
                receiver = candidates[-1]

    def clean(w):
        return re.sub(r"[^a-z]", "", w.lower()) if w else None

    return {"sender": clean(sender), "amount": amount, "receiver": clean(receiver)}
