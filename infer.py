import os
import json
import torch
from tokenizers.bpe import BPETokenizer, PAD_ID, BOS_ID, EOS_ID
from tokenizers.wordpiece import WordPieceTokenizer
from nmt.models import Seq2Seq
from nmt.evaluate import greedy_decode

def load_tokenizer(ttype: str, path: str):
    if ttype == "bpe":
        return BPETokenizer.load(path)
    elif ttype == "wordpiece":
        return WordPieceTokenizer.load(path)
    else:
        raise ValueError(f"Unknown tokenizer type: {ttype}")

def load_model(ckpt_path: str, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    tok_type = ckpt["tok_type"]
    src_tok = load_tokenizer(tok_type, os.path.join(cfg["work_dir"], "tokenizer_src.json"))
    tgt_tok = load_tokenizer(tok_type, os.path.join(cfg["work_dir"], "tokenizer_tgt.json"))
    model = Seq2Seq(
        src_vocab_size=len(src_tok.inv_vocab),
        tgt_vocab_size=len(tgt_tok.inv_vocab),
        emb_dim=cfg["model"]["embedding_dim"],
        enc_hidden=cfg["model"]["hidden_size"],
        enc_layers=cfg["model"]["enc_layers"],
        dec_hidden=cfg["model"]["hidden_size"],
        dec_layers=cfg["model"]["dec_layers"],
        dropout=cfg["model"]["dropout"],
        pad_idx=PAD_ID,
        use_attention=cfg["model"].get("use_attention", True)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_tok, tgt_tok, cfg

def translate_sentences(ckpt_path: str, sentences, device="cuda"):
    model, src_tok, tgt_tok, cfg = load_model(ckpt_path, device=device)
    import torch
    ids = [src_tok.encode(s) for s in sentences]
    max_len = max(len(x) for x in ids)
    pad_ids = [x + [PAD_ID]*(max_len-len(x)) for x in ids]
    lens = [len(x) for x in ids]
    src = torch.tensor(pad_ids, dtype=torch.long, device=device)
    lens = torch.tensor(lens, dtype=torch.long, device=device)
    out_ids = greedy_decode(model, src, lens, max_len=200, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, device=device)
    outs = [tgt_tok.decode(x) for x in out_ids]
    return outs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (best.pt).")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="One or more Urdu sentences.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs = translate_sentences(args.ckpt, args.input, device=device)
    for s, o in zip(args.input, outs):
        print(f"URDU: {s}")
        print(f"ROMAN: {o}")
        print("-"*40)