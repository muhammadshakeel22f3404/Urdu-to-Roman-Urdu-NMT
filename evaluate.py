import math
from typing import List, Tuple
import torch
import torch.nn.functional as F

# BLEU-4 from scratch with smoothing (method 1)
def ngram_counts(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(hyps: List[List[str]], refs: List[List[str]], max_n: int = 4, smooth=True) -> float:
    # corpus BLEU
    import collections
    weights = [1.0/max_n]*max_n
    clipped_counts = [0]*max_n
    total_counts = [0]*max_n
    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n+1):
            hyp_ngrams = collections.Counter(ngram_counts(hyp, n))
            ref_ngrams = collections.Counter(ngram_counts(ref, n))
            for ng, c in hyp_ngrams.items():
                clipped_counts[n-1] += min(c, ref_ngrams.get(ng, 0))
            total_counts[n-1] += max(sum(hyp_ngrams.values()), 1)
    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            p = 0.0
        else:
            p = clipped_counts[i] / total_counts[i]
            if p == 0.0 and smooth:
                p = 1e-9
        precisions.append(p)
    # brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(hyp_len, 1))
    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return score * 100.0

def perplexity_from_loss(avg_loss: float) -> float:
    return math.exp(avg_loss)

# Levenshtein distance & CER
def levenshtein(a: str, b: str) -> int:
    # O(nm) DP
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

def cer(hyps: List[str], refs: List[str]) -> float:
    total = 0
    dist = 0
    for h, r in zip(hyps, refs):
        dist += levenshtein(h, r)
        total += max(len(r), 1)
    return dist / total if total > 0 else 0.0

def greedy_decode(model, src, src_lens, max_len, bos_id, eos_id, pad_id=0, device="cuda"):
    model.eval()
    with torch.no_grad():
        enc_out, (h, c) = model.encoder(src, src_lens)
        dec_init = model.init_decoder_state(h, c)
        enc_mask = (src == pad_id)
        B = src.size(0)
        y = torch.full((B,), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(B)]
        hidden = dec_init
        for t in range(max_len):
            logits, hidden, attn = model.decoder.forward_step(y, hidden, enc_out, enc_mask)
            next_y = torch.argmax(logits, dim=-1)
            for i in range(B):
                if not finished[i]:
                    outputs[i].append(int(next_y[i].item()))
                    if next_y[i].item() == eos_id:
                        finished[i] = True
            y = next_y
            if finished.all():
                break
        return outputs