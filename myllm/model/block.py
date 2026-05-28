import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .pos_emb import SinPostionalEncoding
class EncoderLayer(nn.Module):
    """
    ┌─────────────┐
    │   Input      │
    └──────┬──────┘
           ▼
    ┌──────────────────┐
    │  Multi-Head       │ ← self-attention
    │  Self-Attention   │
    └──────┬───────────┘
           ▼ + Residual + LayerNorm
    ┌──────────────────┐
    │  Feed-Forward     │
    │  Network          │
    └──────┬───────────┘
           ▼ + Residual + LayerNorm
    ┌─────────────┐
    │   Output     │
    └──────────────┘
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = SinPostionalEncoding(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))# post norm

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out)) # post norm
        return x
    
class DecoderLayer(nn.Module):
    """
    ┌─────────────┐
    │   Input      │
    └──────┬──────┘
           ▼
    ┌──────────────────┐
    │  Masked Multi-    │ ← masked self-attention (causal)
    │  Head Self-Attn   │
    └──────┬───────────┘
           ▼ + Residual + LayerNorm
    ┌──────────────────┐
    │  Multi-Head       │ ← cross-attention (Q from decoder,
    │  Cross-Attention  │   K/V from encoder)
    └──────┬───────────┘
           ▼ + Residual + LayerNorm
    ┌──────────────────┐
    │  Feed-Forward     │
    │  Network          │
    └──────┬───────────┘
           ▼ + Residual + LayerNorm
    ┌─────────────┐
    │   Output     │
    └──────────────┘
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = SinPostionalEncoding(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-Attention + Residual + LayerNorm
        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinPostionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # embedding * sqrt(d_model) — 论文中的 scaling
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinPostionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x
    
class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer

    输入:  src (batch, src_seq_len)  — 源语言 token ids
           tgt (batch, tgt_seq_len)  — 目标语言 token ids

    输出:  logits (batch, tgt_seq_len, tgt_vocab_size)
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_encoder_layers, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_decoder_layers, dropout, max_len
        )
        # 输出投影: d_model → tgt_vocab_size（论文中与 embedding 共享权重可选）
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    # ---------- Mask 构造工具 ----------

    @staticmethod
    def make_padding_mask(seq, pad_id=0):
        """
        seq: (batch, seq_len) — token ids
        返回: (batch, 1, 1, seq_len) — pad 位置为 0，非 pad 为 1
        """
        return (seq != pad_id).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size, device):
        """
        返回: (1, 1, size, size) 的下三角矩阵
        位置 i 只能 attend to 位置 <= i
        """
        mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)

    # ---------- Forward ----------

    def forward(self, src, tgt, src_pad_id=0, tgt_pad_id=0):
        # 源端 padding mask — 用于 encoder self-attn & decoder cross-attn
        src_mask = self.make_padding_mask(src, src_pad_id)        # (B, 1, 1, S)

        # 目标端: padding mask ∩ causal mask — 用于 decoder self-attn
        tgt_pad_mask = self.make_padding_mask(tgt, tgt_pad_id)    # (B, 1, 1, T)
        tgt_causal_mask = self.make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, T, T)
        tgt_mask = tgt_pad_mask.bool() & tgt_causal_mask.bool()                  # (B, 1, T, T)

        # Encoder → Decoder → Linear
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.output_proj(dec_output)
        return logits

if __name__ == "__main__":
    # ------- 超参数 -------
    VOCAB_SIZE    = 20   # 词表大小
    D_MODEL       = 64
    NUM_HEADS     = 4
    D_FF          = 128
    NUM_LAYERS    = 2
    DROPOUT       = 0.1
    PAD_ID        = 0
    SEQ_LEN       = 10
    BATCH_SIZE    = 32
    EPOCHS        = 50
    LR            = 3e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------- 模型 -------
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(model)
    print("=" * 60)

    # ------- 数据: 学习把序列倒过来 -------
    # src = [a, b, c, d, ...]  →  tgt = [..., d, c, b, a]
    def make_batch(batch_size, seq_len, vocab_size):
        src = torch.randint(2, vocab_size, (batch_size, seq_len))  # 从 2 开始，0=PAD, 1=BOS
        tgt_input = torch.flip(src, dims=[1])                      # 反转作为目标
        # decoder 输入加 BOS 前缀, decoder label 是右移一位
        bos = torch.ones(batch_size, 1, dtype=torch.long)
        tgt_input = torch.cat([bos, tgt_input[:, :-1]], dim=1)     # teacher forcing
        tgt_label = torch.flip(src, dims=[1])                      # 期望输出
        return src.to(device), tgt_input.to(device), tgt_label.to(device)

    # ------- 训练 -------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        src, tgt_in, tgt_label = make_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        logits = model(src, tgt_in)                       # (B, T, V)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            # 计算 token-level 准确率
            preds = logits.argmax(dim=-1)
            acc = (preds == tgt_label).float().mean().item() * 100
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Token Acc: {acc:.1f}%")

    # ------- 推理测试 -------
    print("\n" + "=" * 60)
    print("推理测试：序列反转")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        test_src = torch.tensor([[5, 8, 3, 7, 2]]).to(device)
        print(f"源序列:  {test_src[0].tolist()}")

        # 自回归生成
        bos = torch.ones(1, 1, dtype=torch.long, device=device)  # BOS
        generated = bos

        for _ in range(SEQ_LEN):
            logits = model(test_src, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        # 去掉 BOS
        pred = generated[0, 1:].tolist()
        expected = test_src[0].flip(0).tolist()
        print(f"期望输出: {expected}")
        print(f"模型输出: {pred}")
        print(f"{'成功!' if pred == expected else '未完全收敛，可增加训练轮次'}")