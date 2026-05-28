import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

class BaseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.f2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        return x
    
class MOEFFN(nn.Module):
    def __init__(
            self, 
            d_model: int,
            d_ff: int,
            activate_func: str,
        ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff,bias=False)
        self.down_proj = nn.Linear(d_ff, d_model,bias=False)
        self.act_fn = ACT2FN[activate_func]()
    
    def forward(self, x: torch.Tensor):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x
    

class MOE(nn.Module):
    '''
好，我用一个**具体的数值例子**，逐步拆解这段代码里的每一步操作。

---

## 先设定一个最小可复现的场景

```python
# 假设参数
hidden_size    = 4    # H=4
num_experts    = 3    # E=3
num_experts_per_tok = 2  # K=2 (top-2)

# batch=1, seq_len=3 → N=3 个 token
# x_flat 形状: (3, 4)

# 假设 routing 结果:
topk_idx    = tensor([[0, 2],   # token0 选了 expert0 和 expert2
                      [1, 0],   # token1 选了 expert1 和 expert0
                      [2, 2]])  # token2 选了 expert2 和 expert2?

# 注意: token2 的 topk_idx = [2, 2]，同一个 expert 被选了两次
# (实际 top-k 不会这样，但代码做了防御)

topk_weight = tensor([[0.6, 0.4],   # token0 给 expert0 权重0.6, expert2 权重0.4
                      [0.7, 0.3],   # token1 给 expert1 权重0.7, expert0 权重0.3
                      [0.5, 0.5]])  # token2 两个 expert2 各 0.5
```

---

## 第 1 行：初始化输出缓冲区

```python
y = torch.zeros_like(x_flat)
# y 形状: (3, 4)  —— 全零
# 最终要累积每个 expert 的加权输出
```

```
y = [[0, 0, 0, 0],    ← token0
     [0, 0, 0, 0],    ← token1
     [0, 0, 0, 0]]    ← token2
```

**为什么要 `zeros_like`？** 后面用 `index_add_` 原地累加，需要一个干净的零初始化张量。

---

## 第 2 行：遍历每个 expert

```python
for i, expert in enumerate(self.experts):
```

循环 3 次（i=0, 1, 2），每次处理一个 expert。

---

## 第 3 步：构造 mask —— "谁选了我？"

```python
mask = (topk_idx == i)  # (N, K) bool
```

### 当 i=0 时：

```python
topk_idx = [[0, 2],
            [1, 0],
            [2, 2]]

mask = (topk_idx == 0)
#     [[ True, False],   ← token0 的 slot0 选了 expert0
#      [False,  True],   ← token1 的 slot1 选了 expert0
#      [False, False]]   ← token2 没选 expert0
```

形状 `(3, 2)` —— 每个 token 的 K 个 slot 逐个比较。

---

## 第 4 步：取 token 下标（去重）

```python
token_idx = mask.any(dim=-1).nonzero(as_tuple=False).flatten()
```

分两步理解：

### 4a. `mask.any(dim=-1)` —— 沿 K 维度做"或"运算

```python
mask = [[ True, False],   → any(dim=-1) = [ True],  ← token0 选了 expert0
        [False,  True],   →            = [ True],  ← token1 选了 expert0
        [False, False]]   →            = [False]]  ← token2 没选
```

**`any(dim=-1)` 的含义**：只要一个 token 的 K 个 slot 中**有任意一个**选了 expert i，结果就是 `True`。

结果形状：`(3,)` —— 一个布尔值对应一个 token。

### 4b. `.nonzero().flatten()` —— 把 True 的位置拿出来

```python
[True, True, False].nonzero()  →  tensor([[0], [1]])
                            .flatten()  →  tensor([0, 1])
```

**`token_idx = [0, 1]`** —— 说的是：token0 和 token1 被路由到了 expert0。

> 为什么需要 `.any(dim=-1)` 先去重？假设 `topk_idx = [[0, 0], ...]`，同一个 token 的两个 slot 都选了 expert0，mask 两个都是 True。但我们只需要这个 token 被 expert 处理**一次**，而不是处理两遍。

---

## 第 5 步：取出对应权重

```python
weight = topk_weight[mask].reshape(-1, 1)
```

### 关键：`topk_weight[mask]` 是 bool 索引

```python
topk_weight = [[0.6, 0.4],
               [0.7, 0.3],
               [0.5, 0.5]]

mask        = [[ True, False],   ← 取 (0,0)=0.6
               [False,  True],   ← 取 (1,1)=0.3
               [False, False]]

topk_weight[mask]  →  tensor([0.6, 0.3])  # 一维，按行优先顺序取出 True 位置的值
                   .reshape(-1, 1)
                   →  tensor([[0.6],
                              [0.3]])       # (2, 1)
```

**`weight = [[0.6], [0.3]]`** —— token0 给 expert0 的权重是 0.6，token1 给 expert0 的权重是 0.3。

### 注意 token_idx 和 weight 的对应关系

```
token_idx = [0, 1]     ← 位置 0 是 token0，位置 1 是 token1
weight    = [[0.6],    ← 位置 0 对应 token0 的权重 0.6
             [0.3]]    ← 位置 1 对应 token1 的权重 0.3
```

**两者严格一一对应**，因为 `nonzero` 和 `bool索引` 都按相同的行优先遍历顺序工作。

---

## 第 6 步：Expert 前向 + 加权

```python
expert_out = expert(x_flat[token_idx])  # (M, H)
# x_flat[token_idx]  → x_flat[[0,1]]  → 取出 token0 和 token1 的 hidden states
# expert(...)         → 送入 FFN      → 输出 (2, 4)

y.index_add_(0, token_idx, (expert_out * weight).to(y.dtype))
```

### 逐步拆解：

```python
# expert_out: (2, 4)
# weight:     (2, 1)

expert_out * weight
# broadcast: (2, 4) * (2, 1) → (2, 4)
# token0 的 4 维输出每维都乘 0.6
# token1 的 4 维输出每维都乘 0.3
```

```python
# 假设 expert_out = [[1.0, 2.0, 3.0, 4.0],    ← token0 的 expert 输出
#                    [5.0, 6.0, 7.0, 8.0]]     ← token1 的 expert 输出

# 乘权重后:
# [[0.6, 1.2, 1.8, 2.4],    ← 0.6 * token0
#  [1.5, 1.8, 2.1, 2.4]]    ← 0.3 * token1
```

---

## 第 7 步：`index_add_` —— 原地散射累加

```python
y.index_add_(0, token_idx, (expert_out * weight).to(y.dtype))
```

**这是最关键的操作。** 参数含义：

```
dim   = 0          ← 沿第 0 维（token 维）操作
index = [0, 1]     ← 源数据第 0 行加到 y 的第 0 行，第 1 行加到 y 的第 1 行
tensor = [...]     ← 要累加的值
```

```python
# 执行前 y:
# y = [[0, 0, 0, 0],    ← token0
#      [0, 0, 0, 0],    ← token1
#      [0, 0, 0, 0]]    ← token2

# expert0 的贡献:
# y[0] += [0.6, 1.2, 1.8, 2.4]
# y[1] += [1.5, 1.8, 2.1, 2.4]

# 执行后 y:
# y = [[0.6, 1.2, 1.8, 2.4],
#      [1.5, 1.8, 2.1, 2.4],
#      [0.0, 0.0, 0.0, 0.0]]
```

---

## 继续循环 i=1（expert1）

```python
mask = (topk_idx == 1)
#     [[False, False],   ← token0 没选 expert1
#      [ True, False],   ← token1 的 slot0 选了 expert1
#      [False, False]]   ← token2 没选 expert1

token_idx = [1]
weight = [[0.7]]   # token1 给 expert1 的权重

# expert1 处理 token1，乘权重 0.7，累加到 y[1]
```

```python
# y 变成:
# y = [[0.6,  1.2,  1.8,  2.4],           ← expert0 贡献
#      [1.5+expert1*0.7, ...],             ← expert0 + expert1 两路累加
#      [0.0,  0.0,  0.0,  0.0]]
```

---

## 继续循环 i=2（expert2）

```python
mask = (topk_idx == 2)
#     [[False,  True],   ← token0 的 slot1 选了 expert2
#      [False, False],
#      [ True,  True]]   ← token2 的 slot0 和 slot1 都选了 expert2 !!

# mask.any(dim=-1) = [True, False, True]
# token_idx = [0, 2]

# 但 weight 怎么办？
# topk_weight[mask] = [0.4, 0.5, 0.5]   ← 取出 3 个 True 位置的值
# .reshape(-1, 1) = [[0.4], [0.5], [0.5]]  ← (3, 1)
```

**问题来了：token_idx = [0, 2] 只有 2 个元素，但 weight 有 3 个！**

这就是代码注释说的"防御性处理"——实际上标准 top-k 不会选出重复 expert，所以这个情况在真实运行中不会发生。但如果你硬要处理这种 edge case，需要额外的 `groupby + sum` 逻辑。

正常情况下（top-k 不重复），token_idx 和 weight 的数量一定一致。

---

## 第 8 步：空 expert 的梯度保持

```python
elif self.training:
    y = y + 0.0 * expert(x_flat[:1]).sum()
```

### 为什么需要这个？

```
场景：8 个 expert，但某个 batch 中没有 token 被路由到 expert 5

问题：
  - expert 5 的参数在本次 forward 中完全没有参与计算
  - .backward() 时不会为 expert 5 的参数产生梯度
  - 如果用 DistributedDataParallel (DDP)，它要求所有参数都有梯度
  - DDP 报错：RuntimeError: Expected to have finished reduction...
```

### 解法的巧妙之处

```python
0.0 * expert(x_flat[:1]).sum()
#       ↑                        ← expert 真的执行了一次前向（计算图连通）
#                ↑               ← .sum() 把输出压成标量
#   ↑                            ← 乘以 0.0，对 y 的数值完全无影响
```

```python
# 计算图:
# expert.params → expert() → output → * 0.0 → + y
#                                        ↑
#                                  梯度通道连通！
# backward 时：d(0.0 * output)/d(params) = 0.0
# 数值上是零，但梯度图是完整的，DDP 满意
```

**`y = y + ...` 而不是 `y[0, 0] += ...`**：用 `+` 创建新张量再赋值，避免 `index_add_` 那种 in-place 操作可能的 autograd 问题。

---

## 完整的累加过程汇总

```
初始化:  y = zeros(3, 4)

expert0 处理 token0, token1 → 加权累加
expert1 处理 token1         → 加权累加
expert2 处理 token0, token2 → 加权累加

最终 y 的每一行 = Σ (该 token 选中的每个 expert 的输出 × 对应权重)
```

**数学上等价于：**

$$y_n = \sum_{k=1}^{K} w_{n,k} \cdot \text{Expert}_{\text{idx}_{n,k}}(x_n)$$

其中 $K$ 是 `num_experts_per_tok`，$w_{n,k}$ 是 top-k 权重，$\text{idx}_{n,k}$ 是选中的 expert 编号。    
    
    '''
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int,
            router_aux_loss_coef: float,
            router_z_loss_coef: float,
            norm_topk: bool = True
        ):
        super().__init__()

        self.num_experts = num_experts
        self.norm_topk = norm_topk
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            MOEFFN(d_model=d_model,d_ff=d_ff,activate_func="silu") for _ in range(num_experts)
        )
    # ── 辅助损失 ────────────────────────────
    @staticmethod
    def _load_balance_loss(
        router_probs: torch.Tensor,   # (N, E) softmax 后
        topk_idx: torch.Tensor,       # (N, K)
        num_experts: int,
    ) -> torch.Tensor:
        """
        Switch Transformer 风格的 load-balancing loss:
          L = E · Σ_e (f_e · P_e)
        f_e = fraction of tokens dispatched to expert e
        P_e = average router probability for expert e
        """
        N = router_probs.shape[0]
        # f_e: 每个 expert 被选中的 token 占比
        expert_mask = F.one_hot(topk_idx, num_experts).sum(dim=1).float()  # (N, E)
        f = expert_mask.mean(dim=0)            # (E,)
        P = router_probs.mean(dim=0)           # (E,)
        return num_experts * (f * P).sum()

    @staticmethod
    def _z_loss(router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss: 防止 logits 过大导致 softmax 过于尖锐"""
        return torch.logsumexp(router_logits, dim=-1).square().mean()
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)# (N, H), N = B*S

        router_logits = self.gate(x_flat)# (N, H)
        router_probs = F.softmax(router_logits, dim=-1)

        K = self.num_experts
        topk_weights, topk_idx = torch.topk(
            router_probs,
            k=K,
            dim=-1,
            sorted=False
        )# (N, K)

        if self.norm_topk:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-2)
        
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 哪些 (token, slot) 选了 expert i
            mask = (topk_idx == i)# (N, K) bool
            if mask.any():
                # 取出被路由到 expert i 的 token 下标 (去重)
                token_idx = mask.any(dim=-1).nonzero(as_tuple=False).flatten()
                # 对应的权重 (可能一个 token 在 K 个 slot 里选了同一个 expert 两次,
                # 理论上 top-k 不会重复，但防御性处理)
                weight = topk_weights[mask].reshape(-1,1)# (M, 1)
                expert_out = expert(x_flat[token_idx])       # (M, H)
                y.index_add_(0, token_idx, (expert_out * weight).to(y.dtype))
            elif self.training:
                # 空 expert: 让梯度图仍然连到参数，避免 DDP 报错
                y = y + 0.0 * expert(x_flat[:1]).sum()

        if self.training:
            lb_loss = self._load_balance_loss(router_probs, topk_idx, self.num_experts)
            z_loss = self._z_loss(router_logits)

            self.aux_loss = (
                self.router_aux_loss_coef * lb_loss
              + self.router_z_loss_coef   * z_loss
            )
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)
        
        return y.view(batch_size, seq_len, d_model)


class MOEFeedForwardBatched(nn.Module):
    """
    不用 Python for-loop，将所有分发给同一 expert 的 token
    一次性打包送入 expert，GPU 利用率更高。

    思路:
      1. top-k routing
      2. 按 expert 排序 token → 一次矩阵乘法
      3. scatter 回原位
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate    = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MOEFFN(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        N = B * S
        E = self.config.num_experts
        K = self.config.num_experts_per_tok

        x_flat = x.view(N, H)
        router_logits = self.gate(x_flat)
        router_probs  = F.softmax(router_logits, dim=-1)

        topk_weight, topk_idx = torch.topk(router_probs, K, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # ---- 将 (token_idx, expert_idx) 对展平并按 expert 排序 ----
        # flat_idx: [0,0,...,0, 1,1,...,1, ..., N-1,...] 共 N*K 个
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K)  # (N,K)
        flat_token_idx = flat_token_idx.reshape(-1)       # (N*K,)
        flat_expert_idx = topk_idx.reshape(-1)             # (N*K,)
        flat_weight     = topk_weight.reshape(-1)          # (N*K,)

        # 按 expert id 排序 → 同一 expert 的 token 连续
        sorted_expert_idx, perm = flat_expert_idx.sort(stable=True)
        sorted_token_idx = flat_token_idx[perm]
        sorted_weight    = flat_weight[perm]

        # 每个 expert 分到多少 token
        tokens_per_expert = torch.zeros(E, dtype=torch.long, device=x.device)
        tokens_per_expert.scatter_add_(
            0, sorted_expert_idx, torch.ones_like(sorted_expert_idx)
        )
        # 每个 expert 的起始偏移
        expert_offsets = tokens_per_expert.cumsum(0)

        # ---- 逐 expert 批量前向 (仍用 loop 但次数 = num_experts 且输入是 batched) ----
        output_flat = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            start = 0 if i == 0 else expert_offsets[i - 1]
            end   = expert_offsets[i]
            if start == end:
                if self.training:
                    output_flat = output_flat + 0.0 * expert(x_flat[:1]).sum()
                continue

            tok_ids = sorted_token_idx[start:end]       # (M,)
            w       = sorted_weight[start:end].unsqueeze(-1)  # (M, 1)
            expert_out = expert(x_flat[tok_ids])         # (M, H)
            # scatter_add 回原位（一个 token 可能被多个 slot 选中同一 expert 的情况已展开）
            output_flat.index_add_(0, tok_ids, (expert_out * w).to(output_flat.dtype))

        # ---- aux loss ----
        if self.training:
            expert_mask = F.one_hot(topk_idx, E).sum(1).float()
            f = expert_mask.mean(0)
            P = router_probs.mean(0)
            lb_loss = E * (f * P).sum()
            z_loss  = torch.logsumexp(router_logits, dim=-1).square().mean()
            self.aux_loss = (
                self.config.router_aux_loss_coef * lb_loss
              + self.config.router_z_loss_coef   * z_loss
            )
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        return output_flat.view(B, S, H)