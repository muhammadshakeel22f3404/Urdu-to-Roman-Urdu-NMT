# Urdu → Roman Urdu NMT (BiLSTM Encoder-Decoder with Attention)

This project trains a sequence-to-sequence model to translate Urdu text into its Roman Urdu transliteration using a BiLSTM encoder and an LSTM decoder in PyTorch. It includes:
- From-scratch BPE tokenizer (no external tokenization/tokenizer libraries)
- Training, evaluation (BLEU, perplexity, CER), and inference
- Streamlit app for public demo
- Rule-based transliteration fallback if Roman Urdu is not present in the dataset
- Optional simple "xLSTM" variant (bonus)

## Dataset
Use: https://github.com/amir9ume/urdu_ghazals_rekhta

The repo reportedly includes Urdu, English transliteration, and Hindi forms. We extract Urdu → Roman Urdu. If Roman is not available, we fallback to a rule-based transliteration from Urdu to Roman Urdu.

## Splits
- Train/Val/Test = 50% / 25% / 25%

## Experiments
At least three experiments varying hyperparameters are included in `configs/exp*.json`.

## Metrics
- BLEU-4 (from scratch, with smoothing)
- Perplexity (exp of average cross-entropy)
- CER (Levenshtein distance normalized by length)
- Qualitative examples logged

## Streamlit
`app/streamlit_app.py` provides a public UI. You can deploy via Streamlit Community Cloud.

Code Tree
urdu_roman_nmt/
├─ requirements.txt
├─ README.md
├─ configs/
│  ├─ exp1.json
│  ├─ exp2.json
│  └─ exp3.json
├─ data_prep/
│  ├─ normalize.py
│  └─ transliteration.py
├─ tokenizers/
│  ├─ bpe.py
│  └─ wordpiece.py
├─ nmt/
│  ├─ datasets.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ infer.py
│  └─ utils.py
├─ app/
│  └─ streamlit_app.py
└─ experiments/
   └─ run_experiments.py
