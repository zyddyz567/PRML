import os
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备: {DEVICE}")

RESULT_DIR = "transformer_comparison_results"
os.makedirs(RESULT_DIR, exist_ok=True)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 20

SEQ_LEN = 10
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 300
TEST_SAMPLES = 300

D_MODEL = 64
N_HEADS = 4
D_FF = 128
N_LAYERS = 2
DROPOUT = 0.1
MAX_LEN = 30

BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

MODEL_TYPES = [
    "Transformer-None",
    "Transformer-Sinusoidal",
    "Transformer-Learned",
    "RNN-LSTM",
    "CNN-1D"
]

class ReverseSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        super().__init__()
        self.src = []
        self.tgt_in = []
        self.tgt_out = []

        for _ in range(num_samples):
            seq = np.random.randint(3, vocab_size, size=(seq_len,), dtype=np.int64)
            rev = seq[::-1].copy()

            src = seq
            tgt_in = np.concatenate([[BOS_ID], rev])
            tgt_out = np.concatenate([rev, [EOS_ID]])

            self.src.append(src)
            self.tgt_in.append(tgt_in)
            self.tgt_out.append(tgt_out)

        self.src = torch.tensor(np.array(self.src), dtype=torch.long)
        self.tgt_in = torch.tensor(np.array(self.tgt_in), dtype=torch.long)
        self.tgt_out = torch.tensor(np.array(self.tgt_out), dtype=torch.long)

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt_in[idx], self.tgt_out[idx]

def build_dataloaders():
    train_set = ReverseSequenceDataset(TRAIN_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    val_set = ReverseSequenceDataset(VAL_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    test_set = ReverseSequenceDataset(TEST_SAMPLES, SEQ_LEN, VOCAB_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, pe_type="sinusoidal", dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        nn.init.normal_(self.token_emb.weight, mean=0, std=d_model ** -0.5)

        if pe_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif pe_type == "learned":
            self.pos_emb = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.pos_emb.weight, mean=0, std=d_model ** -0.5)
        elif pe_type == "none":
            self.pos_emb = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        x = self.token_emb(tokens) * math.sqrt(self.d_model)

        if self.pe_type == "sinusoidal":
            x = x + self.pos_emb(x)
        elif self.pe_type == "learned":
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
            x = x + self.pos_emb(positions)

        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        norm_x = self.norm1(x)
        attn_out, attn = self.self_attn(norm_x, norm_x, norm_x, src_mask)
        x = x + self.dropout1(attn_out)

        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)
        return x, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        norm_x = self.norm1(x)
        self_attn_out, self_attn = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout1(self_attn_out)

        norm_x2 = self.norm2(x)
        cross_attn_out, cross_attn = self.cross_attn(norm_x2, memory, memory, memory_mask)
        x = x + self.dropout2(cross_attn_out)

        norm_x3 = self.norm3(x)
        ffn_out = self.ffn(norm_x3)
        x = x + self.dropout3(ffn_out)
        return x, self_attn, cross_attn

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, n_layers=2, dropout=0.1, max_len=64,
                 pe_type="sinusoidal"):
        super().__init__()
        self.pe_type = pe_type
        self.src_embedding = TokenPositionEmbedding(vocab_size, d_model, max_len, pe_type, dropout)
        self.tgt_embedding = TokenPositionEmbedding(vocab_size, d_model, max_len, pe_type, dropout)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        return (src == PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        pad_mask = (tgt == PAD_ID).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask | causal_mask

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        x = self.src_embedding(src)
        enc_attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x, src_mask)
            enc_attns.append(attn)

        x = self.encoder_norm(x)
        return x, src_mask, enc_attns

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_embedding(tgt)
        dec_self_attns, dec_cross_attns = [], []
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
            dec_self_attns.append(self_attn)
            dec_cross_attns.append(cross_attn)

        x = self.decoder_norm(x)
        return x, dec_self_attns, dec_cross_attns

    def forward(self, src, tgt, return_attn=False):
        memory, src_mask, enc_attns = self.encode(src)
        dec_out, dec_self_attns, dec_cross_attns = self.decode(tgt, memory, src_mask)
        logits = self.output_proj(dec_out)

        if return_attn:
            return logits, {"decoder_cross": dec_cross_attns}
        return logits

class RNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=dropout)

        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.fc_out = nn.Linear(d_model * 2, vocab_size)

    def forward(self, src, tgt, return_attn=False):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        enc_out, (hidden, cell) = self.encoder(src_emb)
        dec_out, _ = self.decoder(tgt_emb, (hidden, cell))

        attn_out, _ = self.attention(dec_out, enc_out, enc_out)

        out = torch.cat([dec_out, attn_out], dim=-1)
        logits = self.fc_out(out)

        if return_attn: return logits, None
        return logits

class CNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.encoder = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.decoder = nn.Conv1d(d_model, d_model, kernel_size=3)

        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, return_attn=False):
        src_emb = self.embedding(src).transpose(1, 2)
        tgt_emb = self.embedding(tgt).transpose(1, 2)

        enc_out = torch.relu(self.encoder(src_emb)).transpose(1, 2)

        dec_in_padded = F.pad(tgt_emb, pad=(2, 0))
        dec_out = torch.relu(self.decoder(dec_in_padded)).transpose(1, 2)

        attn_out, _ = self.attention(dec_out, enc_out, enc_out)

        logits = self.fc_out(dec_out + attn_out)

        if return_attn: return logits, None
        return logits

@torch.no_grad()
def greedy_decode(model, src, max_len):
    model.eval()
    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        logits = model(src, ys)
        if isinstance(logits, tuple): logits = logits[0]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

    return ys[:, 1:]

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_tokens, total_correct, total_seq, exact_match = 0.0, 0, 0, 0, 0

    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

        batch_tokens = (tgt_out != PAD_ID).sum().item()
        pred = logits.argmax(dim=-1)
        correct = ((pred == tgt_out) & (tgt_out != PAD_ID)).sum().item()

        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        total_correct += correct

        generated = greedy_decode(model, src, max_len=tgt_out.size(1))
        exact_match += (generated == tgt_out).all(dim=1).sum().item()
        total_seq += src.size(0)

    return total_loss / total_tokens, total_correct / total_tokens, exact_match / total_seq

def train_one_model(model_name, train_loader, val_loader, test_loader):
    print(f"\n" + "=" * 60)
    print(f" 开始训练模型: {model_name} ")
    print("=" * 60)

    if "Transformer" in model_name:
        pe_type = model_name.split("-")[1].lower()
        model = TransformerSeq2Seq(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
            d_ff=D_FF, n_layers=N_LAYERS, dropout=DROPOUT, max_len=MAX_LEN, pe_type=pe_type
        )
    elif "RNN" in model_name:
        model = RNNSeq2Seq(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT)
    elif "CNN" in model_name:
        model = CNNSeq2Seq(vocab_size=VOCAB_SIZE, d_model=D_MODEL)

    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"-> 模型参数量: {param_count:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val_loss = float("inf")
    best_val_seq_acc = -1.0
    best_state = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_tokens = 0.0, 0

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)

            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_tokens = (tgt_out != PAD_ID).sum().item()
            train_loss_sum += loss.item() * batch_tokens
            train_tokens += batch_tokens

        train_loss = train_loss_sum / train_tokens
        val_loss, val_tok_acc, val_seq_acc = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch, "model_name": model_name,
            "train_loss": train_loss, "val_loss": val_loss,
            "val_seq_acc": val_seq_acc
        })

        print(
            f"[{model_name}] Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Seq Acc: {val_seq_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_seq_acc = val_seq_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_tok_acc, test_seq_acc = evaluate(model, test_loader, criterion)
    elapsed = time.time() - start_time

    result = {
        "model_name": model_name, "test_seq_acc": test_seq_acc,
        "best_val_seq_acc": best_val_seq_acc, "time_sec": elapsed, "params": param_count
    }

    print(f"--> [{model_name}] 训练完毕! 耗时: {elapsed:.2f} 秒 | 最终 Test Seq Acc: {test_seq_acc:.4f}")
    return model, pd.DataFrame(history), result

def plot_comparison(all_histories, all_results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for model_name, hist_df in all_histories.items():
        ax1.plot(hist_df["epoch"], hist_df["train_loss"], marker='s', label=model_name, linewidth=2)
    ax1.set_title('Training Loss Decay Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('CrossEntropy Loss', fontsize=10)
    ax1.legend(frameon=True)

    names = [r["model_name"] for r in all_results]
    times = [r["time_sec"] for r in all_results]
    colors = ['#10b981' if 'Transformer' in n else ('#4f46e5' if 'RNN' in n else '#0ea5e9') for n in names]

    bars = ax2.bar(names, times, color=colors, width=0.5)
    ax2.set_title('Computational Efficiency (Total Training Time)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time in Seconds (Lower is Better)', fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{yval:.1f}s', va='bottom', ha='center',
                 fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "fig1_model_comparison.png")
    plt.savefig(path, dpi=300)
    plt.show()

def plot_test_accuracy(results_df):
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["model_name"], results_df["test_seq_acc"], color='#f59e0b', width=0.5)
    plt.xlabel("Model Architecture")
    plt.ylabel("Test Exact Sequence Accuracy")
    plt.title("Sequence Reversal Task: Test Accuracy Comparison")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "fig2_test_accuracy.png")
    plt.savefig(path, dpi=300)
    plt.show()

@torch.no_grad()
def plot_cross_attention_heatmap(model, loader, model_name):
    if "Transformer" not in model_name: return
    model.eval()
    src, tgt_in, tgt_out = next(iter(loader))
    src, tgt_in, tgt_out = src[:1].to(DEVICE), tgt_in[:1].to(DEVICE), tgt_out[:1].to(DEVICE)

    _, attns = model(src, tgt_in, return_attn=True)
    cross_attn = attns["decoder_cross"][-1][0].mean(dim=0).detach().cpu().numpy()

    src_tokens = src[0].detach().cpu().numpy().tolist()
    tgt_tokens = tgt_out[0].detach().cpu().numpy().tolist()

    plt.figure(figsize=(8, 6))
    plt.imshow(cross_attn, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.xticks(range(len(src_tokens)), src_tokens)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title(f"Decoder Cross-Attention Heatmap ({model_name})")
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"fig3_attention_{model_name}.png")
    plt.savefig(path, dpi=300)
    plt.show()

def main():
    print("=" * 80)
    print(" 序列生成模型对比实验 (Transformer vs RNN vs CNN) ")
    print(f" 全局保存目录: {RESULT_DIR}")
    print("=" * 80)

    train_loader, val_loader, test_loader = build_dataloaders()

    all_histories = {}
    all_results = []
    best_transformer_model = None
    best_transformer_name = ""
    highest_trans_acc = -1

    for m_type in MODEL_TYPES:
        model, history, result = train_one_model(m_type, train_loader, val_loader, test_loader)

        all_histories[m_type] = history
        all_results.append(result)

        if "Transformer" in m_type and result["test_seq_acc"] > highest_trans_acc:
            highest_trans_acc = result["test_seq_acc"]
            best_transformer_model = model
            best_transformer_name = m_type

    results_df = pd.DataFrame(all_results).sort_values(by="test_seq_acc", ascending=False)
    print("\n" + "=" * 80)
    print("最终评估排行榜:")
    print(results_df.to_string(index=False))
    print("=" * 80)
    results_df.to_csv(os.path.join(RESULT_DIR, "final_results.csv"), index=False)

    print("正在生成可视化分析图表...")
    plot_comparison(all_histories, all_results)
    plot_test_accuracy(results_df)

    if best_transformer_model is not None:
        plot_cross_attention_heatmap(best_transformer_model, test_loader, best_transformer_name)

    print(f"\n全部实验已圆满结束，结果均保存于 [{RESULT_DIR}] 文件夹中。")

if __name__ == "__main__":
    main()