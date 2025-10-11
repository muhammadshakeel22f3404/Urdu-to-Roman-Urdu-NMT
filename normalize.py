import regex as re

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_urdu(text: str) -> str:
    if text is None:
        return ""
    # Remove diacritics
    text = ARABIC_DIACRITICS.sub("", text)
    # Normalize alef and ya/heh variants
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ی", "ئ": "ی", "ي": "ی",
        "ؤ": "و",
        "ه": "ہ",  # unify heh
        "ۃ": "ہ",
        "ك": "ک", "ٮ": "ب", "ٯ": "ف"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove weird spacing and control chars
    text = re.sub(r"[\u200c\u200d]", "", text)  # ZWNJ/ZWJ
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_roman(text: str) -> str:
    if text is None:
        return ""
    # Lowercase and basic cleanup
    text = text.strip()
    # Avoid aggressive lowercasing due to names; keep as-is, but normalize spaces
    text = re.sub(r"\s+", " ", text)
    return text