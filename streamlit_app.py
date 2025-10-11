import os
import streamlit as st
import torch
import tempfile
import requests
from tokenizers.bpe import BPETokenizer, PAD_ID, BOS_ID, EOS_ID
from tokenizers.wordpiece import WordPieceTokenizer
from nmt.models import Seq2Seq
from nmt.evaluate import greedy_decode

st.set_page_config(page_title="Urdu → Roman Urdu NMT", page_icon="📝", layout="centered")

@st.cache_resource
def load_from_local_or_url(model_dir: str, model_url: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_url:
        # download into temp dir
        tmp = tempfile.mkdtemp()
        local_path = os.path.join(tmp, "best.pt")
        r = requests.get(model_url)
        with open(local_path, "wb") as f:
            f.write(r.content)
        ckpt_path = local_path
    else:
        ckpt_path = os.path.join(model_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    tok_type = ckpt["tok_type"]
    src_tok = BPETokenizer.load(os.path.join(cfg["work_dir"], "tokenizer_src.json")) if tok_type=="bpe" else WordPieceTokenizer.load(os.path.join(cfg["work_dir"], "tokenizer_src.json"))
    tgt_tok = BPETokenizer.load(os.path.join(cfg["work_dir"], "tokenizer_tgt.json")) if tok_type=="bpe" else WordPieceTokenizer.load(os.path.join(cfg["work_dir"], "tokenizer_tgt.json"))

    model = Seq2Seq(
        src_vocab_size=len(src_tok.inv_vocab),
        tgt_vocab_size=len(tgt_tok.inv_vocab),
        emb_dim=cdf(cfg,"model","embedding_dim",256),
        enc_hidden=cdf(cfg,"model","hidden_size",512),
        enc_layers=cdf(cfg,"model","enc_layers",2),
        dec_hidden=cdf(cfg,"model","hidden_size",512),
        dec_layers=cdf(cfg,"model","dec_layers",4),
        dropout=cdf(cfg,"model","dropout",0.3),
        pad_idx=PAD_ID,
        use_attention=cdf(cfg,"model","use_attention",True)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_tok, tgt_tok, device

def cdf(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

st.title("Urdu → Roman Urdu Neural Machine Translation")
st.write("BiLSTM Encoder + LSTM Decoder with attention. Enter Urdu text below and get Roman Urdu transliteration.")

model_dir = st.text_input("Local model directory (contains best.pt and tokenizers):", value="/app/urdu_roman_nmt/outputs/exp1")
model_url = st.text_input("Optional: MODEL_URL to a downloadable best.pt", value=os.environ.get("MODEL_URL",""))

if st.button("Load/Reload Model"):
    try:
        model, src_tok, tgt_tok, device = load_from_local_or_url(model_dir, model_url if model_url else None)
        st.success("Model loaded!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if "model" not in st.session_state:
    try:
        model, src_tok, tgt_tok, device = load_from_local_or_url(model_dir, model_url if model_url else None)
        st.session_state.model = model
        st.session_state.src_tok = src_tok
        st.session_state.tgt_tok = tgt_tok
        st.session_state.device = device
    except Exception:
        pass

inp = st.text_area("Input Urdu text", value="عشق نے غالب نکما کر دیا")
max_len = st.slider("Max decode length", min_value=32, max_value=256, value=128, step=8)

if st.button("Translate") and "model" in st.session_state:
    model = st.session_state.model
    src_tok = st.session_state.src_tok
    tgt_tok = st.session_state.tgt_tok
    device = st.session_state.device
    import torch
    ids = src_tok.encode(inp)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
    out_ids = greedy_decode(model, src, lens, max_len=max_len, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, device=device)[0]
    out = tgt_tok.decode(out_ids)
    st.subheader("Roman Urdu")
    st.write(out)

st.caption("Tip: Deploy this app on Streamlit Community Cloud and point MODEL_URL to a hosted best.pt (e.g., from GitHub Releases or Hugging Face).")