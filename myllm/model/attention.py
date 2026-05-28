import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class ScaleDotProduction(nn.Module):
    def __init__(self, drop: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop)
    
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None
            ):
        '''
        Q, K, V: (batch, heads, seq_len, d_k)
        mask:    (batch, 1, 1, seq_len)  — padding mask
                 (batch, 1, seq_len, seq_len) — causal mask
        '''
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        return output, attn_weights
    

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, num_heads:int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "必须整除"

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProduction(dropout)

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None
            ):
        batch_size = Q.size(0)

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        attn_output, attn_weights = self.attention(Q,K,V,mask)

        attn_output = (
            attn_output.transpose(1,2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        output = self.W_O(attn_output)
        return output, attn_weights
    
class GroupedQueryAttention(nn.Module):
    """
MHA (Multi-Head Attention):     Q: 8头  K: 8头  V: 8头   ← 每个Q头独立KV
GQA (Grouped Query Attention):  Q: 8头  K: 2头  V: 2头   ← 每组Q共享一组KV
MQA (Multi-Query Attention):    Q: 8头  K: 1头  V: 1头   ← 所有Q共享一个KV

GQA 是 MHA 和 MQA 的中间态

Query Heads:   Q0 Q1 Q2 Q3 | Q4 Q5 Q6 Q7
                  ↓           ↓
KV Heads:         K0, V0       K1, V1
               (group 0)     (group 1)

    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_kv_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    
    def forward(
        self, 
        x: torch.Tensor, 
        position_embeddings, 
        kv_cache = None,
        use_cache=False, 
        attention_mask=None
    ):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads,    self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        from .pos_emb import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # 在 S 维拼接
            v = torch.cat([cached_v, v], dim=2)
        kv_cache_out = (k, v)if use_cache else None

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GQA 核心 —— 扩展 KV 头以匹配 Q 头数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #
        #  这是 GQA 的精髓:
        #
        #  before repeat:
        #    Q: (B, 8, S, D)     ← 8 个 Q 头
        #    K: (B, 2, S, D)     ← 2 个 KV 头
        #    V: (B, 2, S, D)
        #
        #  group_size = 8 // 2 = 4
        #
        #  after repeat_interleave:
        #    K: (B, 8, S, D)     ← 每个 KV 头复制 4 次
        #    V: (B, 8, S, D)
        #
        #  原始 KV 头 0 → 复制给 Q 头 0,1,2,3
        #  原始 KV 头 1 → 复制给 Q 头 4,5,6,7
        #
        k = k.repeat_interleave(self.group_size, dim=1)  # (B, num_heads, S_full, D)
        v = v.repeat_interleave(self.group_size, dim=1)  # (B, num_heads, S_full, D)

        # 使用 PyTorch 2.0+ 的高效实现（自动选择 Flash Attention）
        if attention_mask is not None:
            # attention_mask 形状: (B, 1, S, S_full)
            # 已经是加性 mask（0 表示可见，-inf 表示不可见）
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,  # mask 已经外部传入
            )
        else:
            # 无 mask 时用 is_causal 自动构造因果 mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(kv_cache is None),  # 有 cache 时不做 causal（只有新 token）
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 7: 合并多头 + 输出投影
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # attn_output: (B, num_heads, S, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)  # (B, S, H)

        output = self.o_proj(attn_output)  # (B, S, H)

        return output, kv_cache_out