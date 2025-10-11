# From-scratch BPE tokenizer for multilingual text (Urdu-friendly).
# Implements training merges and encoding/decoding with special tokens.

import json
from collections import Counter, defaultdict
import regex as re
from typing import List, Dict, Tuple

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

def whitespace_tokenize(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", s.strip())
    return s.split(" ") if s else []

def word_to_symbols(word: str) -> List[str]:
    # SentencePiece-ish: prefix '▁' to indicate word boundary, then characters
    if not word:
        return []
    return ["▁"] + list(word)

def get_pair_stats(dataset: List[List[str]]) -> Counter:
    stats = Counter()
    for symbols in dataset:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            stats[pair] += 1
    return stats

def merge_pair(symbols: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
    i = 0
    out = []
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out

class BPETokenizer:
    def __init__(self):
        self.vocab = {}          # token -> id
        self.inv_vocab = []      # id -> token
        self.merges = []         # list of merged pairs
        self.rank = {}           # pair -> rank
        self.min_freq = 2

    def fit(self, texts: List[str], vocab_size: int = 8000, min_freq: int = 2):
        self.min_freq = min_freq
        # Build initial dataset: list of list of symbols
        words = []
        for line in texts:
            for w in whitespace_tokenize(line):
                if w:
                    words.append(w)
        # Count words to weight frequency
        word_counter = Counter(words)
        dataset = []
        for w, c in word_counter.items():
            symbols = word_to_symbols(w)
            dataset.extend([symbols] * c)

        # Initial symbols (characters + '▁')
        symbols_set = set()
        for seq in dataset:
            symbols_set.update(seq)

        # Prepare vocab with special tokens first
        self.vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.inv_vocab = list(SPECIAL_TOKENS)
        # Add initial chars
        for ch in sorted(symbols_set):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.inv_vocab)
                self.inv_vocab.append(ch)

        target_size = max(vocab_size, len(self.inv_vocab))
        merges = []

        while len(self.inv_vocab) < target_size:
            stats = get_pair_stats(dataset)
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            if freq < self.min_freq:
                break
            merged = pair[0] + pair[1]
            # Merge in dataset
            dataset = [merge_pair(seq, pair, merged) for seq in dataset]
            merges.append(pair)
            # Add merged token to vocab
            if merged not in self.vocab:
                self.vocab[merged] = len(self.inv_vocab)
                self.inv_vocab.append(merged)

        self.merges = merges
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

    def _encode_word_symbols(self, symbols: List[str]) -> List[str]:
        # Greedy merge application using self.rank
        if not symbols or not self.rank:
            return symbols
        pairs = self._get_pairs(symbols)
        while True:
            # pick best pair
            candidate = None
            best_rank = None
            for p in pairs:
                if p in self.rank:
                    r = self.rank[p]
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        candidate = p
            if candidate is None:
                break
            merged = candidate[0] + candidate[1]
            symbols = merge_pair(symbols, candidate, merged)
            pairs = self._get_pairs(symbols)
        return symbols

    @staticmethod
    def _get_pairs(symbols: List[str]) -> set:
        return {(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        toks = []
        for w in whitespace_tokenize(text):
            symbols = word_to_symbols(w)
            symbols = self._encode_word_symbols(symbols)
            toks.extend(symbols)
        ids = []
        if add_special_tokens:
            ids.append(BOS_ID)
        for t in toks:
            ids.append(self.vocab.get(t, UNK_ID))
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
        # Join subwords by removing '▁' markers
        out = ""
        for t in toks:
            if t.startswith("▁"):
                out += " " + t[1:]
            else:
                out += t
        return out.strip()

    def to_json(self) -> str:
        data = {
            "vocab": self.inv_vocab,
            "merges": self.merges,
            "min_freq": self.min_freq
        }
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "BPETokenizer":
        data = json.loads(s)
        tok = BPETokenizer()
        tok.inv_vocab = data["vocab"]
        tok.vocab = {t: i for i, t in enumerate(tok.inv_vocab)}
        tok.merges = [tuple(x) for x in data["merges"]]
        tok.rank = {tuple(p): i for i, p in enumerate(tok.merges)}
        tok.min_freq = data.get("min_freq", 2)
        return tok

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return BPETokenizer.from_json(f.read())