import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import MOEFFN as FFN
from .ffn import MOE as MOE
from .pos_emb import precompute_freqs_cis
class MyBlock(nn.Module):
    def __init__(
        self, 
        layer_id: int,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        act_fun: str,
        eps: float,
        dropout: float,
        use_moe: bool,
        router_aux_loss_coef: float,
        router_z_loss_coef: float,
        norm_topk: bool,
        num_experts: int = 0,
    ):
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout
        )
        self.layer_norm = RMSNorm(dim=d_model, eps=eps)
        self.post_attn_layer_norm = RMSNorm(dim=d_model, eps=eps)
        self.mlp = FFN(d_model=d_model,d_ff=d_ff,activate_func=act_fun) if not use_moe else MOE(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
            norm_topk=norm_topk
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        position_embeddings, 
        past_key_value=None, 
        use_cache=False, 
        attention_mask=None
    ):
        residual = x
        x, present_k_v = self.attention(
            self.layer_norm(x),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        x += residual
        x = x + self.mlp(self.post_attn_layer_norm(x))
        return x, present_k_v
    
class MyLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        act_fun: str,
        eps: float,
        dropout: float,
        use_moe: bool,
        router_aux_loss_coef: float,
        router_z_loss_coef: float,
        norm_topk: bool,
        num_experts: int = 0,
    ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            MyBlock(
                layer_id=l,
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                act_fun=act_fun,
                eps=eps,
                dropout=dropout,
                use_moe=use_moe,
                num_experts=num_experts,
                router_aux_loss_coef=router_aux_loss_coef,
                router_z_loss_coef=router_z_loss_coef,
                norm_topk=norm_topk
            )
            for l in range(num_layers)
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=d_model, end=max_position_embeddings, rope_base=rope_theta, rope_scaling=rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None, 
        past_key_values=None, 
        use_cache=False, 
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_len], self.freqs_sin[start_pos:start_pos + seq_len])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOE)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MyLLMForCausalLLM:
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: float,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        act_fun: str,
        eps: float,
        dropout: float,
        use_moe: bool,
        router_aux_loss_coef: float,
        router_z_loss_coef: float,
        norm_topk: bool,
        num_experts: int = 0,
    ):
        
        super().__init__(self.config)
        self.model = MyLLM(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            act_fun=act_fun,
            eps=eps,
            dropout=dropout,
            use_moe=use_moe,
            router_aux_loss_coef=router_aux_loss_coef,
            router_z_loss_coef=router_z_loss_coef,
            norm_topk=norm_topk,
            num_experts=num_experts
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return (loss, aux_loss, logits, past_key_values, hidden_states)