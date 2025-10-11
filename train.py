import os
import json
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nmt.datasets import (
    find_dataset_pairs,
    split_pairs,
    save_pairs,
    load_pairs,
    ParallelTextDataset,
    collate_fn,
    build_tokenizers,
)
from nmt.utils import ensure_dir, save_json, load_json, set_seed, count_parameters
from nmt.evaluate import bleu_score, perplexity_from_loss, cer, greedy_decode
from tokenizers.bpe import BPETokenizer, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from tokenizers.wordpiece import WordPieceTokenizer

from nmt.models import Seq2Seq


def load_tokenizer(ttype: str, path: str):
    if ttype == "bpe":
        return BPETokenizer.load(path)
    elif ttype == "wordpiece":
        return WordPieceTokenizer.load(path)
    else:
        raise ValueError(f"Unknown tokenizer type: {ttype}")


def evaluate_model(model, dataloader, src_tok, tgt_tok, device):
    model.eval()
    import torch
    import torch.nn as nn
    from tokenizers.bpe import BOS_ID, EOS_ID, PAD_ID

    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    hyps: List[List[str]] = []
    refs: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch["src_lens"].to(device)

            # Teacher forcing off for validation
            logits = model(
                src,
                src_lens,
                tgt,
                src_pad_idx=PAD_ID,
                teacher_forcing_ratio=0.0,
            )
            B, Tm1, V = logits.size()
            loss = ce(logits.reshape(B * Tm1, V), tgt[:, 1:].reshape(B * Tm1))
            total_loss += loss.item()
            total_tokens += (B * Tm1)

            # Greedy decode to compute BLEU/CER
            gen_ids = greedy_decode(
                model,
                src,
                src_lens,
                max_len=tgt.size(1),
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_id=PAD_ID,
                device=device,
            )
            for i in range(len(gen_ids)):
                hyp = tgt_tok.decode(gen_ids[i])
                ref = tgt_tok.decode([int(x) for x in tgt[i].tolist()])
                hyps.append(hyp.split())
                refs.append(ref.split())

    bleu = bleu_score(hyps, refs, max_n=4, smooth=True)
    ppl = perplexity_from_loss(total_loss / max(1, total_tokens))
    hyps_s = [" ".join(h) for h in hyps]
    refs_s = [" ".join(r) for r in refs]
    cer_v = cer(hyps_s, refs_s)
    return bleu, ppl, cer_v


def run_training(config: Dict):
    # Seed and device
    set_seed(config["training"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir = config["dataset_dir"]
    work_dir = config["work_dir"]
    ensure_dir(work_dir)

    # Save config copy
    save_json(config, os.path.join(work_dir, "config.json"))

    # Build or load data splits
    train_path = os.path.join(work_dir, "train.jsonl")
    val_path = os.path.join(work_dir, "val.jsonl")
    test_path = os.path.join(work_dir, "test.jsonl")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        pairs = find_dataset_pairs(dataset_dir)
        # Only fail if zero pairs; NO >100 rule
        if len(pairs) == 0:
            raise RuntimeError(f"No pairs found in dataset dir: {dataset_dir}. Please check the path and files.")
        print(f"[data] Total pairs harvested: {len(pairs)}")
        train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.5, 0.25)
        print(f"[data] Split sizes -> train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")
        save_pairs(train_pairs, train_path)
        save_pairs(val_pairs, val_path)
        save_pairs(test_pairs, test_path)
    else:
        train_pairs = load_pairs(train_path)
        val_pairs = load_pairs(val_path)
        test_pairs = load_pairs(test_path)
        print(f"[data] Loaded cached splits -> train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    # Tokenizers
    tok_type, src_tok_path, tgt_tok_path = None, None, None
    if not (os.path.exists(os.path.join(work_dir, "tokenizer_src.json")) and os.path.exists(os.path.join(work_dir, "tokenizer_tgt.json"))):
        tok_type, src_tok_path, tgt_tok_path = build_tokenizers(train_pairs, config["tokenizer"], work_dir)
        print(f"[tok] Built tokenizers ({tok_type}) at {src_tok_path} and {tgt_tok_path}")
    else:
        tok_type = config["tokenizer"].get("type", "bpe")
        src_tok_path = os.path.join(work_dir, "tokenizer_src.json")
        tgt_tok_path = os.path.join(work_dir, "tokenizer_tgt.json")
        print(f"[tok] Using cached tokenizers from {work_dir} (type={tok_type})")

    src_tok = load_tokenizer(tok_type, src_tok_path)
    tgt_tok = load_tokenizer(tok_type, tgt_tok_path)

    # Datasets and loaders
    train_ds = ParallelTextDataset(train_pairs, src_tok, tgt_tok)
    val_ds = ParallelTextDataset(val_pairs, src_tok, tgt_tok)
    test_ds = ParallelTextDataset(test_pairs, src_tok, tgt_tok)

    train_dl = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = Seq2Seq(
        src_vocab_size=len(src_tok.inv_vocab),
        tgt_vocab_size=len(tgt_tok.inv_vocab),
        emb_dim=config["model"]["embedding_dim"],
        enc_hidden=config["model"]["hidden_size"],
        enc_layers=config["model"]["enc_layers"],
        dec_hidden=config["model"]["hidden_size"],
        dec_layers=config["model"]["dec_layers"],
        dropout=config["model"]["dropout"],
        pad_idx=PAD_ID,
        use_attention=config["model"].get("use_attention", True),
    ).to(device)

    # IMPORTANT: make sure decoder init states match dec_layers (handles enc/dec layer mismatch)
    # Patch the model to ensure init_decoder_state repeats/truncates to dec_layers
    orig_init = model.init_decoder_state

    def init_decoder_state_fixed(h, c):
        # h,c from encoder: [enc_layers*2, B, H]
        # orig will return [enc_layers, B, dec_H] after projection
        h_, c_ = orig_init(h, c)
        dec_layers = model.decoder.lstm.num_layers
        if h_.size(0) == dec_layers:
            return (h_, c_)
        elif h_.size(0) > dec_layers:
            return (h_[-dec_layers:], c_[-dec_layers:])
        else:
            # repeat last layer to match dec_layers
            need = dec_layers - h_.size(0)
            h_rep = torch.cat([h_, h_[-1:].repeat(need, 1, 1)], dim=0)
            c_rep = torch.cat([c_, c_[-1:].repeat(need, 1, 1)], dim=0)
            return (h_rep, c_rep)

    model.init_decoder_state = init_decoder_state_fixed

    print(f"[model] Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    max_epochs = config["training"]["max_epochs"]
    clip = config["training"]["clip_grad"]
    tf_start = config["training"]["teacher_forcing_start"]
    tf_end = config["training"]["teacher_forcing_end"]
    tf_anneal = config["training"]["teacher_forcing_anneal_epochs"]
    save_every = config["training"]["save_every"]
    eval_every = config["training"]["eval_every"]

    best_val_bleu = -1.0
    best_path = os.path.join(work_dir, "best.pt")

    for epoch in range(1, max_epochs + 1):
        # Anneal teacher forcing
        if tf_anneal <= 1:
            tf_ratio = tf_end
        else:
            if epoch <= tf_anneal:
                tf_ratio = tf_start - (epoch - 1) * (tf_start - tf_end) / max(1, tf_anneal - 1)
            else:
                tf_ratio = tf_end

        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{max_epochs} (TF={tf_ratio:.2f})")
        total_loss = 0.0
        total_tokens = 0

        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch["src_lens"].to(device)

            optimizer.zero_grad()
            # logits for positions 1..T-1; targets are tgt[:,1:]
            logits = model(
                src,
                src_lens,
                tgt,
                src_pad_idx=PAD_ID,
                teacher_forcing_ratio=tf_ratio,
            )
            B, Tm1, V = logits.size()
            loss = criterion(logits.reshape(B * Tm1, V), tgt[:, 1:].reshape(B * Tm1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item() * (B * Tm1)
            total_tokens += (B * Tm1)
            pbar.set_postfix({"loss/token": f"{total_loss/max(1,total_tokens):.6f}"})

        train_avg_loss = total_loss / max(1, total_tokens)
        train_ppl = perplexity_from_loss(train_avg_loss)
        print(f"[train] epoch {epoch}: avg_loss/token={train_avg_loss:.6f}  ppl={train_ppl:.2f}")

        # Validation
        if epoch % eval_every == 0:
            val_bleu, val_ppl, val_cer = evaluate_model(model, val_dl, src_tok, tgt_tok, device)
            print(f"[val] epoch {epoch}: BLEU={val_bleu:.2f}  PPL={val_ppl:.2f}  CER={val_cer:.4f}")

            # Save best by BLEU
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "tok_type": tok_type,
                    },
                    best_path,
                )
                print(f"[ckpt] Saved new best to {best_path} (BLEU={best_val_bleu:.2f})")

        # Periodic checkpoints
        if epoch % save_every == 0:
            ckpt_path = os.path.join(work_dir, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "tok_type": tok_type,
                },
                ckpt_path,
            )
            print(f"[ckpt] Saved {ckpt_path}")

    # Final test evaluation using best checkpoint if available
    test_dl = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[ckpt] Loaded best model from {best_path}")

    test_bleu, test_ppl, test_cer = evaluate_model(model, test_dl, src_tok, tgt_tok, device)
    print(f"[test] BLEU={test_bleu:.2f}  PPL={test_ppl:.2f}  CER={test_cer:.4f}")

    # Save metrics
    with open(os.path.join(work_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_bleu": best_val_bleu,
                "test_bleu": test_bleu,
                "test_ppl": test_ppl,
                "test_cer": test_cer,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_training(cfg)