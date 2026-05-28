import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinPostionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 模型支持的最大seq len，也就是最大上下文
        # 每个token都对应一个长为 d_model的向量
        pe = torch.zeros(max_len, d_model)

        # (max_len, 1)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)
        )

        '''
        pe[:, 0::2]
        : 表示选中所有行（即所有位置）。

        0::2 表示从第 0 列开始，步长为 2，也就是选取索引 0, 2, 4, ... 的列。

        结果形状：(max_len, ceil(d_model/2))，即所有行的偶数列。
        '''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        '''
        pe.unsqueeze(0)
        在赋值完成后，执行 pe = pe.unsqueeze(0)，
        将形状从 (max_len, d_model) 变为 (1, max_len, d_model)，
        增加一个 batch 维度，便于后续与输入 x 相加。
        '''
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    
    def forward(self, x):
        '''
        self.pe 形状：(1, max_len, d_model)

        : (第一个维度)：取所有 batch（这里只有 1 个 batch）。

        :x.size(1) (第二个维度)：取序列维度的前 x.size(1) 个位置。x.size(1) 是当前输入 x 的序列长度（seq_len）。
        因为 self.pe 存储了最长 max_len 的编码，这里只切片出实际需要的长度。

        : (第三个维度)：取所有特征维度（d_model）。

        结果形状：(1, seq_len, d_model)。然后与 x（形状 (batch, seq_len, d_model)）相加，
        利用广播机制自动将 batch 维度从 1 扩展到 batch。
        '''
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed