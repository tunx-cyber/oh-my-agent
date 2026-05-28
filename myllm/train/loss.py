import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: (batch, seq_len, vocab_size) 模型输出
    labels: (batch, seq_len) 目标token ids
    ignore_index: 忽略的位置（padding、prompt部分等）

    标准做法：只计算 assistant 回复部分的 loss
    """
    # shift: 预测 token_{t+1} 基于 token_{t}
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 展平
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # CrossEntropyLoss 自带 softmax
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,  # prompt部分设为 -100
        reduction='mean'
    )
    return loss

# 构造 labels 时，prompt 部分设为 -100 只在 response 部分计算 loss
def create_sft_labels(input_ids, prompt_len):
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # 忽略prompt的loss
    return labels