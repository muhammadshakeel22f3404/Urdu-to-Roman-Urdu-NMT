import torch
import os
import glob
import json
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import regex as re

from data_prep.normalize import normalize_urdu, clean_roman
from data_prep.transliteration import urdu_to_roman_fallback
from tokenizers.bpe import BPETokenizer, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from tokenizers.wordpiece import WordPieceTokenizer
from nmt.utils import ensure_dir

# Detect presence of Urdu/Arabic script in a line
URDU_RE = re.compile(r"[\u0600-\u06FF]")

def has_urdu(s: str) -> bool:
    return bool(URDU_RE.search(s))

def _read_all_text_lines(path: str) -> List[str]:
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    lines.append(s)
    except Exception:
        return []
    return lines

def _scan_poet_ur_en(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Dataset organized as:
      dataset_dir/
        PoetA/
          ur/<files>   # Urdu text
          en/<files>   # Roman Urdu (same filenames, aligned line by line)
          hi/<ignored>
        PoetB/...
    Reads matching ur/en files line by line and makes (urdu, roman) pairs.
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(dataset_dir):
        return pairs

    poet_dirs = [
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if not poet_dirs:
        return pairs

    total_lines = 0
    kept_pairs = 0

    for poet_dir in tqdm(poet_dirs, desc="Scanning poet folders"):
        ur_dir = os.path.join(poet_dir, "ur")
        en_dir = os.path.join(poet_dir, "en")
        if not os.path.isdir(ur_dir) or not os.path.isdir(en_dir):
            continue

        # Match files by filename
        ur_files = {fn: os.path.join(ur_dir, fn) for fn in os.listdir(ur_dir)}
        en_files = {fn: os.path.join(en_dir, fn) for fn in os.listdir(en_dir)}

        # Process only files present in both ur and en
        common_files = set(ur_files.keys()) & set(en_files.keys())

        for fn in common_files:
            ur_lines = _read_all_text_lines(ur_files[fn])
            en_lines = _read_all_text_lines(en_files[fn])

            # Align line by line
            for ur, en in zip(ur_lines, en_lines):
                total_lines += 1
                if not has_urdu(ur):
                    continue
                ur_norm = normalize_urdu(ur)
                en_norm = clean_roman(en)
                if ur_norm and en_norm:
                    pairs.append((ur_norm, en_norm))
                    kept_pairs += 1

    print(f"[loader] From UR+EN files: total_lines={total_lines}, kept_pairs={kept_pairs}, pairs={len(pairs)}")
    return pairs

def _scan_csvs(dataset_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    files = glob.glob(os.path.join(dataset_dir, "**", "*.*"), recursive=True)
    for f in files:
        if f.lower().endswith(".csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            urdu_col = None
            roman_col = None
            # guess urdu column
            for c in df.columns:
                cl = c.lower()
                if cl in ["urdu", "in_urdu", "text_urdu", "ghazal_urdu", "ur", "src"]:
                    urdu_col = c
                    break
            # guess roman/english transliteration column
            for c in df.columns:
                cl = c.lower()
                if cl in ["roman", "roman_urdu", "english", "transliteration", "in_roman", "en", "eng", "tgt"]:
                    roman_col = c
                    break
            if urdu_col is not None:
                if roman_col is not None:
                    for _, row in df.iterrows():
                        src = normalize_urdu(str(row[urdu_col])) if pd.notna(row[urdu_col]) else ""
                        tgt = clean_roman(str(row[roman_col])) if pd.notna(row[roman_col]) else ""
                        if src and tgt:
                            pairs.append((src, tgt))
                else:
                    for _, row in df.iterrows():
                        src = normalize_urdu(str(row[urdu_col])) if pd.notna(row[urdu_col]) else ""
                        if src:
                            tgt = urdu_to_roman_fallback(src)
                            pairs.append((src, tgt))
    print(f"[loader] From CSVs: pairs={len(pairs)}")
    return pairs

def find_dataset_pairs(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Primary: harvest from UR+EN files.
    Fallback: scan CSVs if present.
    """
    pairs = _scan_poet_ur_en(dataset_dir)
    if len(pairs) == 0:
        pairs = _scan_csvs(dataset_dir)
    return pairs

def split_pairs(pairs: List[Tuple[str, str]], train_ratio=0.5, val_ratio=0.25):
    n = len(pairs)
    idx = list(range(n))
    import random
    random.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [pairs[i] for i in idx[:n_train]]
    val = [pairs[i] for i in idx[n_train:n_train+n_val]]
    test = [pairs[i] for i in idx[n_train+n_val:]]
    return train, val, test

def save_pairs(pairs: List[Tuple[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s, t in pairs:
            f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")

def load_pairs(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append((obj["src"], obj["tgt"]))
    return out

def build_tokenizers(train_pairs: List[Tuple[str, str]], cfg_tok: Dict, work_dir: str):
    ttype = cfg_tok.get("type", "bpe")
    src_vocab = cfg_tok.get("src_vocab_size", 8000)
    tgt_vocab = cfg_tok.get("tgt_vocab_size", 8000)
    min_freq = cfg_tok.get("min_freq", 2)
    src_texts = [s for s, _ in train_pairs]
    tgt_texts = [t for _, t in train_pairs]
    if ttype == "bpe":
        src_tok = BPETokenizer()
        tgt_tok = BPETokenizer()
        src_tok.fit(src_texts, vocab_size=src_vocab, min_freq=min_freq)
        tgt_tok.fit(tgt_texts, vocab_size=tgt_vocab, min_freq=min_freq)
        src_tok.save(os.path.join(work_dir, "tokenizer_src.json"))
        tgt_tok.save(os.path.join(work_dir, "tokenizer_tgt.json"))
        return ("bpe", os.path.join(work_dir, "tokenizer_src.json"), os.path.join(work_dir, "tokenizer_tgt.json"))
    elif ttype == "wordpiece":
        src_tok = WordPieceTokenizer()
        tgt_tok = WordPieceTokenizer()
        src_tok.fit(src_texts, vocab_size=src_vocab, min_freq=min_freq)
        tgt_tok.fit(tgt_texts, vocab_size=tgt_vocab, min_freq=min_freq)
        src_tok.save(os.path.join(work_dir, "tokenizer_src.json"))
        tgt_tok.save(os.path.join(work_dir, "tokenizer_tgt.json"))
        return ("wordpiece", os.path.join(work_dir, "tokenizer_src.json"), os.path.join(work_dir, "tokenizer_tgt.json"))
    else:
        raise ValueError(f"Unknown tokenizer type: {ttype}")

class ParallelTextDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_tok, tgt_tok, max_len: int = 200):
        self.pairs = pairs
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.src_tok.encode(src)
        tgt_ids = self.tgt_tok.encode(tgt)
        # clip
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }

def pad_sequence(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    out = []
    lens = []
    for s in seqs:
        lens.append(len(s))
        out.append(s + [pad_id] * (max_len - len(s)))
    return out, lens

def collate_fn(batch):
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]
    src_pad, src_lens = pad_sequence(src_ids, pad_id=PAD_ID)
    tgt_pad, tgt_lens = pad_sequence(tgt_ids, pad_id=PAD_ID)
    import torch
    return {
        "src": torch.tensor(src_pad, dtype=torch.long),
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "tgt": torch.tensor(tgt_pad, dtype=torch.long),
        "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
    }

# Lazy import torch to avoid global dependency during file scan
import torch