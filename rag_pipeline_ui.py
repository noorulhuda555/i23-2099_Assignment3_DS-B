import json
import math
import re
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

try:
    import streamlit as st
except Exception:
    st = None


def cache_resource(func):
    if st is None:
        return func
    return st.cache_resource(show_spinner=True)(func)


# ---------------------------
# Constants and label maps
# ---------------------------
MAX_LEN = 128
MAX_DEC_LEN = 112
MIN_FREQ = 2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SENT_MAP = {0: "<NEG>", 1: "<NEU>", 2: "<POS>"}
LEN_MAP = {0: "<SHORT>", 1: "<MEDIUM>", 2: "<LONG>"}

SENTIMENT_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LENGTH_LABEL = {0: "Short", 1: "Medium", 2: "Long"}

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "subset.json"
ENCODER_CKPT = BASE_DIR / "partA_checkpoints" / "best_encoder.pt"
TRAIN_EMB_PATH = BASE_DIR / "partA_checkpoints" / "train_embeddings.pt"
DECODER_CKPT = BASE_DIR / "best_decoder.pt"


def map_sentiment(rating):
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def length_bucket(text):
    n = len(text.split())
    if n < 30:
        return 0
    if n < 100:
        return 1
    return 2


def tokenize(text):
    return text.split()


def encode(text, vocab, max_len=MAX_LEN):
    tokens = [CLS_TOKEN] + tokenize(text)
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens][:max_len]
    ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        q = self.split_heads(self.W_Q(x))
        k = self.split_heads(self.W_K(x))
        v = self.split_heads(self.W_V(x))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous()
        bsz, seq_len, heads, d_k = out.size()
        out = out.view(bsz, seq_len, heads * d_k)
        return self.dropout(self.W_O(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_len=MAX_LEN,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.sentiment_head = nn.Linear(d_model, 3)
        self.length_head = nn.Linear(d_model, 3)

    def make_pad_mask(self, x):
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        mask = self.make_pad_mask(x)
        out = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm(out)
        cls_emb = out[:, 0, :]
        sent_logits = self.sentiment_head(cls_emb)
        len_logits = self.length_head(cls_emb)
        return sent_logits, len_logits, cls_emb


def causal_scaled_dot_product(q, k, v, pad_mask=None):
    _, _, seq_len, d_k = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    if pad_mask is not None:
        scores = scores.masked_fill(pad_mask == 1, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


class DecoderMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        bsz, seq_len, _ = x.shape

        def split(w):
            return w(x).view(bsz, seq_len, self.h, self.dk).transpose(1, 2)

        q, k, v = split(self.Wq), split(self.Wk), split(self.Wv)
        out = causal_scaled_dot_product(q, k, v, pad_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.drop(self.Wo(out))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = DecoderMHA(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.ff(self.norm2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=MAX_DEC_LEN,
        dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def make_pad_mask(self, x):
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        pad_mask = self.make_pad_mask(x)
        out = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, pad_mask)
        out = self.norm(out)
        return self.lm_head(out)

def build_decoder_input(review_text, sentiment_id, length_id, retrieved_texts, max_ctx_words=30):
    sent_tok = SENT_MAP[sentiment_id]
    len_tok  = LEN_MAP[length_id]
    # Use all retrieved texts, not just index 0
    ctx_snippets = " ".join(
        " ".join(t.split()[:max_ctx_words]) for t in retrieved_texts
    )
    review_snippet = " ".join(review_text.split()[:max_ctx_words])
    return f"{sent_tok} {len_tok} {ctx_snippets} {review_snippet}"

@cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    if not ENCODER_CKPT.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint at {ENCODER_CKPT}")
    if not TRAIN_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing train embeddings at {TRAIN_EMB_PATH}")
    if not DECODER_CKPT.exists():
        raise FileNotFoundError(f"Missing decoder checkpoint at {DECODER_CKPT}")

    records = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.dropna(subset=["reviewText"]).reset_index(drop=True)
    df["sentiment"] = df["overall"].apply(map_sentiment)
    df["cleaned_review"] = df["reviewText"].apply(clean_text)
    df["length_label"] = df["cleaned_review"].apply(length_bucket)

    train_df, _ = train_test_split(df, test_size=0.30, random_state=42, stratify=df["sentiment"])
    train_df_reset = train_df.reset_index(drop=True)

    counter = Counter()
    for text in train_df_reset["cleaned_review"]:
        counter.update(tokenize(text))

    encoder_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2}
    for word, freq in counter.items():
        if freq >= MIN_FREQ:
            encoder_vocab[word] = len(encoder_vocab)

    vocab = dict(encoder_vocab)
    for token in (BOS_TOKEN, EOS_TOKEN, "<NEG>", "<NEU>", "<POS>", "<SHORT>", "<MEDIUM>", "<LONG>"):
        if token not in vocab:
            vocab[token] = len(vocab)

    idx2word = {v: k for k, v in vocab.items()}
    pad_id = encoder_vocab[PAD_TOKEN]
    bos_id = vocab[BOS_TOKEN]
    eos_id = vocab[EOS_TOKEN]

    encoder = EncoderTransformer(vocab_size=len(encoder_vocab), pad_idx=pad_id).to(device)
    encoder.load_state_dict(torch.load(ENCODER_CKPT, map_location=device))
    encoder.eval()

    decoder = DecoderTransformer(vocab_size=len(vocab), pad_idx=pad_id).to(device)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=device))
    decoder.eval()

    train_embeddings = torch.load(TRAIN_EMB_PATH, map_location="cpu").float()
    train_emb_norm = F.normalize(train_embeddings, p=2, dim=1)

    if len(train_df_reset) != train_emb_norm.shape[0]:
        raise RuntimeError(
            f"Train rows and embedding rows mismatch ({len(train_df_reset)} vs {train_emb_norm.shape[0]})."
        )

    return {
        "device": device,
        "vocab": vocab,
        "idx2word": idx2word,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "encoder": encoder,
        "decoder": decoder,
        "train_df_reset": train_df_reset,
        "train_emb_norm": train_emb_norm,
    }


def generate(prefix_text, assets, max_new_tokens=25, temperature=1.0):
    vocab = assets["vocab"]
    idx2word = assets["idx2word"]
    bos_id = assets["bos_id"]
    eos_id = assets["eos_id"]
    device = assets["device"]
    decoder = assets["decoder"]

    prefix_tokens = tokenize(prefix_text)
    prefix_ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in prefix_tokens][:80]
    input_ids = prefix_ids + [bos_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits = decoder(x)
            next_logit = logits[0, -1, :] / max(temperature, 1e-6)
            next_id = next_logit.argmax().item()
            if next_id == eos_id:
                break
            input_ids.append(next_id)

    gen_ids = input_ids[len(prefix_ids) + 1 :]
    return " ".join(idx2word.get(i, UNK_TOKEN) for i in gen_ids).strip()


def rag_pipeline(raw_review_text, assets, k=5, max_gen_tokens=30, temperature=1.0):
    vocab = assets["vocab"]
    device = assets["device"]
    encoder = assets["encoder"]
    train_emb_norm = assets["train_emb_norm"]
    train_df_reset = assets["train_df_reset"]

    cleaned = clean_text(raw_review_text)
    ids = torch.tensor([encode(cleaned, vocab, MAX_LEN)], dtype=torch.long, device=device)

    with torch.no_grad():
        sent_logits, len_logits, cls_emb = encoder(ids)

    pred_sentiment = sent_logits.argmax(-1).item()
    pred_length = len_logits.argmax(-1).item()
    sentiment_conf = torch.softmax(sent_logits, dim=-1)[0][pred_sentiment].item()

    q_norm = F.normalize(cls_emb.cpu(), p=2, dim=1)
    scores = (train_emb_norm @ q_norm.T).squeeze(1)
    topk = torch.topk(scores, k=k)
    top_idxs = topk.indices.tolist()
    top_scores = topk.values.tolist()
    retrieved = [train_df_reset.loc[j, "cleaned_review"] for j in top_idxs]

    prefix = build_decoder_input(cleaned, pred_sentiment, pred_length, retrieved)
    generated = generate(prefix, assets, max_new_tokens=max_gen_tokens, temperature=temperature)

    return {
        "pred_sentiment_id": pred_sentiment,
        "pred_length_id": pred_length,
        "cleaned_input": cleaned,
        "pred_sentiment": SENTIMENT_LABEL[pred_sentiment],
        "pred_length": LENGTH_LABEL[pred_length],
        "sentiment_conf": sentiment_conf,
        "retrieved_reviews": list(zip(top_scores, retrieved)),
        "generated": generated,
    }


def run_tkinter_app():
    assets = load_pipeline()

    root = tk.Tk()
    root.title("RAG Pipeline UI")
    root.geometry("1100x760")

    top_frame = ttk.Frame(root, padding=10)
    top_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(top_frame, text="Input Review").grid(row=0, column=0, sticky="w")
    review_box = tk.Text(top_frame, height=6, wrap=tk.WORD)
    review_box.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(4, 8))
    review_box.insert(
        tk.END,
        "this blender works great and crushes ice quickly but the jar feels a bit fragile and the motor is noisy after a few uses",
    )

    ttk.Label(top_frame, text="Top-K").grid(row=2, column=0, sticky="w")
    k_var = tk.StringVar(value="5")
    ttk.Entry(top_frame, textvariable=k_var, width=8).grid(row=2, column=1, sticky="w")

    ttk.Label(top_frame, text="Max Tokens").grid(row=2, column=2, sticky="w")
    tok_var = tk.StringVar(value="25")
    ttk.Entry(top_frame, textvariable=tok_var, width=8).grid(row=2, column=3, sticky="w")

    ttk.Label(top_frame, text="Temperature").grid(row=3, column=0, sticky="w")
    temp_var = tk.StringVar(value="1.0")
    ttk.Entry(top_frame, textvariable=temp_var, width=8).grid(row=3, column=1, sticky="w")

    pred_label = ttk.Label(top_frame, text="Predictions: --")
    pred_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 4))

    gen_label = ttk.Label(top_frame, text="Generated Explanation")
    gen_label.grid(row=5, column=0, sticky="w")
    gen_text = tk.Text(top_frame, height=4, wrap=tk.WORD)
    gen_text.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(4, 8))

    ctx_label = ttk.Label(top_frame, text="Retrieved Context")
    ctx_label.grid(row=7, column=0, sticky="w")
    ctx_text = tk.Text(top_frame, height=14, wrap=tk.WORD)
    ctx_text.grid(row=8, column=0, columnspan=4, sticky="nsew", pady=(4, 8))

    prefix_label = ttk.Label(top_frame, text="Auto generated (by pipeline) Decoder Prefix")
    prefix_label.grid(row=9, column=0, sticky="w")
    prefix_text = tk.Text(top_frame, height=3, wrap=tk.WORD)
    prefix_text.grid(row=10, column=0, columnspan=4, sticky="nsew", pady=(4, 8))

    top_frame.columnconfigure(0, weight=1)
    top_frame.columnconfigure(1, weight=0)
    top_frame.columnconfigure(2, weight=0)
    top_frame.columnconfigure(3, weight=1)
    top_frame.rowconfigure(1, weight=1)
    top_frame.rowconfigure(8, weight=2)

    def on_run():
        review = review_box.get("1.0", tk.END).strip()
        if not review:
            return
        try:
            k = max(1, int(k_var.get()))
            max_toks = max(1, int(tok_var.get()))
            temp = float(temp_var.get())
        except ValueError:
            return

        result = rag_pipeline(review, assets, k=k, max_gen_tokens=max_toks, temperature=temp)

        pred_label.config(
            text=(
                f"Predictions: Sentiment={result['pred_sentiment']} ({result['sentiment_conf']:.2%}) | "
                f"Length={result['pred_length']}"
            )
        )
        gen_text.delete("1.0", tk.END)
        gen_text.insert(tk.END, result["generated"] or "(Empty generation)")

        ctx_text.delete("1.0", tk.END)
        for rank, (score, text) in enumerate(result["retrieved_reviews"], 1):
            ctx_text.insert(tk.END, f"[{rank}] cosine={score:.4f}\n{text}\n\n")

        prefix = build_decoder_input(
            result["cleaned_input"],
            result["pred_sentiment_id"],
            result["pred_length_id"],
            [result["retrieved_reviews"][0][1]] if result["retrieved_reviews"] else [],
        )
        prefix_text.delete("1.0", tk.END)
        prefix_text.insert(tk.END, prefix)

    ttk.Button(top_frame, text="Run Full RAG Pipeline", command=on_run).grid(row=3, column=2, columnspan=2, sticky="e")
    root.mainloop()


def main():
    if st is None:
        run_tkinter_app()
        return

    st.set_page_config(page_title="RAG Pipeline Demo", layout="wide")
    st.title("Interactive RAG Pipeline Demo")
    st.caption("Encoder prediction -> dense retrieval -> decoder generation")

    with st.spinner("Loading models, embeddings, and data split..."):
        assets = load_pipeline()

    with st.sidebar:
        st.header("Controls")
        k = st.slider("Top-K retrieval", min_value=1, max_value=10, value=5)
        max_gen_tokens = st.slider("Max generated tokens", min_value=5, max_value=60, value=25)
        temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=0.8, step=0.1)

    default_text = (
        "the headphones work as expected sound quality is decent and the build feels average not too sturdy not too flimsy the ear cushions are okay for short use but get uncomfortable after an hour battery life is about what they advertise nothing special but gets the job done"
    )

    user_review = st.text_area("Enter a review:", value=default_text, height=140)

    if st.button("Run Full RAG Pipeline", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a non-empty review.")
            return

        result = rag_pipeline(
            raw_review_text=user_review,
            assets=assets,
            k=k,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Step 1: Encoder Prediction")
            st.write(f"**Sentiment:** {result['pred_sentiment']} ({result['sentiment_conf']:.2%} confidence)")
            st.write(f"**Length bucket:** {result['pred_length']}")
            st.write(f"**Cleaned input:** {result['cleaned_input']}")

        with c2:
            st.subheader("Step 3: Generated Explanation")
            st.success(result["generated"] if result["generated"] else "(Empty generation)")

        st.subheader("Step 2: Retrieved Context")
        for rank, (score, text) in enumerate(result["retrieved_reviews"], 1):
            st.markdown(f"**[{rank}] cosine={score:.4f}**")
            st.write(text)

        st.subheader("What was sent to the decoder")
        first_ctx = result["retrieved_reviews"][0][1] if result["retrieved_reviews"] else ""
        st.code(
            build_decoder_input(
                result["cleaned_input"],
                result["pred_sentiment_id"],
                result["pred_length_id"],
                [first_ctx],
            )
        )


if __name__ == "__main__":
    main()
