# Optional: simple WordPiece-like tokenizer from scratch.
# This is a minimal implementation for experimentation (bonus).
import json
from collections import Counter
import regex as re
from typing import List

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def whitespace_tokenize(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", s.strip())
    return s.split(" ") if s else []

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = []
        self.min_freq = 2

    def fit(self, texts: List[str], vocab_size: int = 8000, min_freq: int = 2):
        self.min_freq = min_freq
        # Start with chars + '▁' word boundary
        counts = Counter()
        for line in texts:
            for w in whitespace_tokenize(line):
                if not w:
                    continue
                word = "▁" + w
                for ch in word:
                    counts[ch] += 1
        # Initialize vocab
        self.inv_vocab = list(SPECIAL_TOKENS)
        self.vocab = {tok: i for i, tok in enumerate(self.inv_vocab)}
        for ch, c in counts.most_common():
            if c >= self.min_freq and ch not in self.vocab:
                self.vocab[ch] = len(self.inv_vocab)
                self.inv_vocab.append(ch)
            if len(self.inv_vocab) >= vocab_size:
                break

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(BOS_ID)
        for w in whitespace_tokenize(text):
            w = "▁" + w
            # greedy longest-match-first
            i = 0
            while i < len(w):
                j = len(w)
                found = None
                while j > i:
                    sub = w[i:j]
                    if sub in self.vocab:
                        found = sub
                        break
                    j -= 1
                if found is None:
                    ids.append(UNK_ID)
                    i += 1
                else:
                    ids.append(self.vocab[found])
                    i = j
        if add_special_tokens:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self.inv_vocab):
                tok = "<unk>"
            else:
                tok = self.inv_vocab[i]
            if tok in SPECIAL_TOKENS:
                continue
            toks.append(tok)
        out = ""
        for t in toks:
            if t.startswith("▁"):
                out += " " + t[1:]
            else:
                out += t
        return out.strip()

    def to_json(self) -> str:
        return json.dumps({
            "vocab": self.inv_vocab,
            "min_freq": self.min_freq
        }, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "WordPieceTokenizer":
        data = json.loads(s)
        tok = WordPieceTokenizer()
        tok.inv_vocab = data["vocab"]
        tok.vocab = {t: i for i, t in enumerate(tok.inv_vocab)}
        tok.min_freq = data.get("min_freq", 2)
        return tok

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "WordPieceTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return WordPieceTokenizer.from_json(f.read())