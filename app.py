"""
app.py — Streamlit demo cho VietCultural VQA
Chạy: streamlit run app.py

Yêu cầu:
    pip install streamlit transformers timm torch pillow
"""

import json
import math
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel

# ── Cấu hình ──────────────────────────────────────────────────────────────────
VIT_MODEL        = "vit_base_patch16_224"
PHOBERT_MODEL    = "vinai/phobert-base-v2"
IMAGE_SIZE       = 224
MAX_QUESTION_LEN = 64
MAX_ANSWER_LEN   = 10
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
FUSION_DIM       = 512
CO_ATTN_DIM      = 512
LSTM_HIDDEN_DIM  = 512
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3
TRANSF_NHEAD     = 8
TRANSF_NUM_LAYERS = 4
TRANSF_FF_DIM    = 2048
TRANSF_DROPOUT   = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Model definitions (phải khớp với notebook) ────────────────────────────────
class CoAttentionFusion(nn.Module):
    def __init__(self, img_dim=768, text_dim=768, attn_dim=CO_ATTN_DIM,
                 fusion_dim=FUSION_DIM, dropout=0.2):
        super().__init__()
        self.proj_img  = nn.Linear(img_dim,  attn_dim)
        self.proj_text = nn.Linear(text_dim, attn_dim)
        self.img_attn_query = nn.Linear(attn_dim, attn_dim)
        self.img_attn_key   = nn.Linear(attn_dim, attn_dim)
        self.txt_attn_query = nn.Linear(attn_dim, attn_dim)
        self.txt_attn_key   = nn.Linear(attn_dim, attn_dim)
        self.fusion = nn.Sequential(
            nn.Linear(attn_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.scale = math.sqrt(attn_dim)

    def _attend(self, q, k, v):
        scores  = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, v).mean(dim=1)

    def forward(self, img_feats, text_feats):
        V, T  = self.proj_img(img_feats), self.proj_text(text_feats)
        v_att = self._attend(self.img_attn_query(T), self.img_attn_key(V), V)
        t_att = self._attend(self.txt_attn_query(V), self.txt_attn_key(T), T)
        return self.fusion(torch.cat([v_att, t_att], dim=-1))


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=FUSION_DIM,
                 hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS,
                 dropout=LSTM_DROPOUT, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.context_proj = nn.Linear(FUSION_DIM, hidden_dim)
        self.out_proj     = nn.Linear(hidden_dim, vocab_size)
        self.dropout      = nn.Dropout(dropout)

    def _init_hidden(self, ctx):
        h = self.context_proj(ctx).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, torch.zeros_like(h)

    @torch.no_grad()
    def generate(self, context, sos_idx, eos_idx, max_len=MAX_ANSWER_LEN):
        B = context.size(0)
        h, c = self._init_hidden(context)
        tok  = torch.full((B, 1), sos_idx, dtype=torch.long, device=context.device)
        outputs = []
        for _ in range(max_len):
            emb        = self.embed(tok)
            out, (h,c) = self.lstm(emb, (h, c))
            logit      = self.out_proj(out.squeeze(1))
            tok        = logit.argmax(dim=-1, keepdim=True)
            outputs.append(tok)
        return torch.cat(outputs, dim=1)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=FUSION_DIM, nhead=TRANSF_NHEAD,
                 num_layers=TRANSF_NUM_LAYERS, dim_ff=TRANSF_FF_DIM,
                 dropout=TRANSF_DROPOUT, pad_idx=0, max_len=MAX_ANSWER_LEN + 2):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    @torch.no_grad()
    def generate(self, context, sos_idx, eos_idx, max_len=MAX_ANSWER_LEN):
        B      = context.size(0)
        memory = context.unsqueeze(1)
        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=context.device)
        for _ in range(max_len):
            pos   = torch.arange(tokens.size(1), device=context.device).unsqueeze(0)
            emb   = self.embed(tokens) + self.pos_embed(pos)
            mask  = self._causal_mask(tokens.size(1), context.device)
            out   = self.decoder(tgt=emb, memory=memory, tgt_mask=mask)
            next_ = self.out_proj(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_], dim=1)
        return tokens[:, 1:]


class VQAModel(nn.Module):
    def __init__(self, variant, vocab_size):
        super().__init__()
        self.variant = variant
        self.vit  = timm.create_model(VIT_MODEL, pretrained=False, num_classes=0)
        self.bert = AutoModel.from_pretrained(PHOBERT_MODEL)
        self.co_attn = CoAttentionFusion()
        if variant == "A1":
            self.decoder = LSTMDecoder(vocab_size)
        else:
            self.decoder = TransformerDecoder(vocab_size)

    def encode(self, image, input_ids, attention_mask):
        img_feats  = self.vit.forward_features(image)
        text_feats = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        return self.co_attn(img_feats, text_feats)

    @torch.no_grad()
    def generate(self, image, input_ids, attention_mask, sos_idx, eos_idx):
        ctx = self.encode(image, input_ids, attention_mask)
        return self.decoder.generate(ctx, sos_idx, eos_idx)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VietCultural VQA Demo",
    page_icon="🏛️",
    layout="centered",
)
st.title("🏛️ VietCultural VQA")
st.caption("Upload ảnh → Nhập câu hỏi → Nhận câu trả lời")

# Sidebar: model config
st.sidebar.header("⚙️ Cấu hình")
model_variant = st.sidebar.selectbox("Model Variant", ["A1 (LSTM)", "A2 (Transformer)"])
variant_key   = "A1" if "A1" in model_variant else "A2"

checkpoint_path = st.sidebar.text_input(
    "Đường dẫn checkpoint (.pt)",
    value=f"working/checkpoints/vqa_best_{variant_key}.pt",
)
vocab_path = st.sidebar.text_input(
    "Đường dẫn ANSWER_VOCAB.json",
    value="working/ANSWER_VOCAB.json",
)

# ── Load resources (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(PHOBERT_MODEL)

@st.cache_resource
def load_model_and_vocab(ckpt_path: str, vocab_path: str, variant: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict = json.load(f)
    idx2word = {int(v): k for k, v in vocab.items()}

    sos_idx = vocab.get("<sos>", 1)
    eos_idx = vocab.get("<eos>", 2)
    pad_idx = vocab.get("<pad>", 0)

    model = VQAModel(variant=variant, vocab_size=len(vocab))

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    return model, vocab, idx2word, sos_idx, eos_idx

# ── Main form ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📷 Upload ảnh", type=["jpg", "jpeg", "png", "webp"])
question      = st.text_input("❓ Nhập câu hỏi (tiếng Việt)")

if uploaded_file and question.strip():
    col1, col2 = st.columns([1, 1])
    pil_image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(pil_image, caption="Ảnh tải lên", use_column_width=True)

    if st.button("🔍 Dự đoán câu trả lời"):
        # Validate file paths
        if not Path(checkpoint_path).exists():
            st.error(f"❌ Không tìm thấy checkpoint: `{checkpoint_path}`")
            st.stop()
        if not Path(vocab_path).exists():
            st.error(f"❌ Không tìm thấy vocab file: `{vocab_path}`")
            st.stop()

        with st.spinner("Đang tải model …"):
            tokenizer = load_tokenizer()
            model, vocab, idx2word, sos_idx, eos_idx = load_model_and_vocab(
                checkpoint_path, vocab_path, variant_key
            )

        with st.spinner("Đang suy luận …"):
            # Preprocess image
            image_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

            # Tokenize question
            enc = tokenizer(
                question,
                max_length=MAX_QUESTION_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            # Generate
            pred_ids = model.generate(image_tensor, input_ids, attention_mask,
                                      sos_idx, eos_idx)

            # Decode
            tokens = []
            for t in pred_ids[0].tolist():
                if t == eos_idx:
                    break
                word = idx2word.get(t, "")
                if word not in ("<pad>", "<sos>", "<eos>", "<unk>"):
                    tokens.append(word)
            answer = " ".join(tokens) if tokens else "(không dự đoán được)"

        with col2:
            st.markdown("### 💬 Câu trả lời")
            st.success(answer)
            st.caption(f"Model: {model_variant}  |  Device: {DEVICE}")

elif uploaded_file and not question.strip():
    st.info("Vui lòng nhập câu hỏi để tiếp tục.")
elif not uploaded_file:
    st.info("Vui lòng upload ảnh để bắt đầu.")
