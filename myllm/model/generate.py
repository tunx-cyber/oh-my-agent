import torch

def generate(
        forward_fun,
        inputs=None, 
        attention_mask=None, 
        max_new_tokens=8192, 
        temperature=0.85, 
        top_p=0.85, 
        top_k=50, 
        eos_token_id=2, 
        streamer=None, 
        use_cache=True, 
        num_return_sequences=1, 
        do_sample=True, 
        repetition_penalty=1.0, 
        **kwargs
    ):
    input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = kwargs.pop("past_key_values", None)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if streamer: streamer.put(input_ids.cpu())
    for _ in range(max_new_tokens):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = forward_fun(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
        logits = outputs.logits[:, -1, :] / temperature
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        if top_k > 0: 
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
        if streamer: streamer.put(next_token.cpu())
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all(): break
    if streamer: streamer.end()
    if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
    return input_ids