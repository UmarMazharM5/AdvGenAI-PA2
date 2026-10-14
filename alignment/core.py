"""Response-only causal scoring and original SFT/DPO objectives.

Prompts and responses are tokenized separately without automatic special tokens,
so their boundary is unambiguous. Labels are shifted exactly once when scoring.
SFT uses the mean over all scored response tokens, including EOS.
"""
import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses) or not prompts:
        raise ValueError('Expected equally sized nonempty prompt/response lists')
    if max_length < 1:
        raise ValueError('max_length must be positive')
    pad = getattr(tokenizer, 'pad_token_id', None)
    eos = getattr(tokenizer, 'eos_token_id', None)
    if pad is None:
        pad = eos if eos is not None else 0
    rows, labels = [], []
    for prompt, response in zip(prompts, responses):
        p = list(tokenizer.encode(prompt, add_special_tokens=False))
        r = list(tokenizer.encode(response, add_special_tokens=False))
        if eos is not None:
            r.append(eos)
        rows.append((p + r)[:max_length])
        labels.append(([-100] * len(p) + r)[:max_length])
    width = max(1, max(map(len, rows)))
    return {
        'input_ids': torch.tensor([r + [pad] * (width-len(r)) for r in rows], dtype=torch.long),
        'attention_mask': torch.tensor([[1]*len(r) + [0]*(width-len(r)) for r in rows], dtype=torch.long),
        'labels': torch.tensor([r + [-100]*(width-len(r)) for r in labels], dtype=torch.long),
    }


def score_batch(model, batch):
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    targets = batch['labels'][:, 1:]
    mask = targets.ne(-100) & batch['attention_mask'][:, 1:].bool()
    # Cross entropy avoids materializing an additional full log-softmax tensor.
    nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                          targets.reshape(-1), ignore_index=-100, reduction='none')
    logps = (-nll.reshape_as(targets) * mask).sum(-1)
    return logps, mask.sum(-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return score_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logps, counts = score_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    return -logps.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {'chosen': build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            'rejected': build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                              reference_chosen_logps, reference_rejected_logps, beta):
    gap = (policy_chosen_logps - policy_rejected_logps
           - reference_chosen_logps + reference_rejected_logps)
    return -F.logsigmoid(beta * gap).mean()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pc = score_batch(policy_model, batch['chosen'])[0]
    pr = score_batch(policy_model, batch['rejected'])[0]
    with torch.no_grad():
        rc = score_batch(reference_model, batch['chosen'])[0].to(pc.device)
        rr = score_batch(reference_model, batch['rejected'])[0].to(pc.device)
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
