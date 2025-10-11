# Simple rule-based Urdu → Roman Urdu transliteration (fallback when Roman column is absent).
# This is a heuristic and not perfect for poetry, but helps bootstrap targets.

import regex as re

# Base mapping for Urdu letters to Roman approximations
URDU2ROMAN = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t",
    "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d",
    "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "w", "ؤ": "o",
    "ہ": "h", "ۂ": "h", "ء": "", "ی": "y", "ئ": "i", "ے": "e"
}

# Handle digraphs and contextual rules in rough form
LONG_VOWEL_FIXES = [
    (r"aa([ie])", r"a\1"),  # reduce awkward 'aa' before front vowels
]

def urdu_to_roman_fallback(s: str) -> str:
    if not s:
        return ""
    out_chars = []
    for ch in s:
        if ch in URDU2ROMAN:
            out_chars.append(URDU2ROMAN[ch])
        else:
            out_chars.append(ch)
    roman = "".join(out_chars)
    # basic cleanup
    roman = re.sub(r"\s+", " ", roman).strip()
    for pat, rep in LONG_VOWEL_FIXES:
        roman = re.sub(pat, rep, roman)
    # collapse repeated letters a bit
    roman = re.sub(r"(.)\1{2,}", r"\1\1", roman)
    return roman