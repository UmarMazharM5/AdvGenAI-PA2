"""Independent scalar checks catch causal shifts and preference-sign mistakes."""
from types import SimpleNamespace

import torch
from torch import nn

from alignment.core import (compute_response_logprobs, compute_sft_loss,
                            compute_dpo_loss_from_logps, compute_dpo_loss)
from tests.toy_lm import ToyTokenizer


class TransitionLM(nn.Module):
    def __init__(self, vocabulary, seed=7):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.transitions = nn.Parameter(torch.randn(vocabulary, vocabulary, generator=generator))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.transitions[input_ids])


def scalar_reference(model, tokenizer, prompt, response, maximum):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sequence = (prompt_ids + tokenizer.encode(response, add_special_tokens=False)
                + [tokenizer.eos_token_id])[:maximum]
    terms = [model.transitions[sequence[position-1]].log_softmax(-1)[sequence[position]]
             for position in range(max(1, len(prompt_ids)), len(sequence))]
    return torch.stack(terms).sum(), len(terms)


def test_nonuniform_scores_and_token_weighted_sft_match_scalar_reference():
    tokenizer = ToyTokenizer()
    model = TransitionLM(tokenizer.vocab_size)
    prompts, responses = ['ab', 'cdef', ''], ['x', 'yzabcdefgh', 'bc']
    reference = [scalar_reference(model, tokenizer, p, r, 8) for p, r in zip(prompts, responses)]
    scores = compute_response_logprobs(model, tokenizer, prompts, responses, 8)
    torch.testing.assert_close(scores, torch.stack([r[0] for r in reference]))
    expected = -sum(r[0] for r in reference)/sum(r[1] for r in reference)
    actual = compute_sft_loss(model, tokenizer, prompts, responses, 8)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    actual_gradient = model.transitions.grad.clone()
    model.zero_grad()
    expected.backward()
    torch.testing.assert_close(actual_gradient, model.transitions.grad)


def test_dpo_matches_scalar_reference_and_does_not_backpropagate_into_reference():
    tokenizer = ToyTokenizer()
    policy = TransitionLM(tokenizer.vocab_size, seed=1)
    reference = TransitionLM(tokenizer.vocab_size, seed=2)
    prompts, chosen, rejected = ['ab', 'cd'], ['xyz', 'f'], ['j', 'ghi']
    margins = []
    for prompt, good, bad in zip(prompts, chosen, rejected):
        pc = scalar_reference(policy, tokenizer, prompt, good, 8)[0]
        pr = scalar_reference(policy, tokenizer, prompt, bad, 8)[0]
        rc = scalar_reference(reference, tokenizer, prompt, good, 8)[0].detach()
        rr = scalar_reference(reference, tokenizer, prompt, bad, 8)[0].detach()
        margins.append(pc-pr-rc+rr)
    expected = torch.nn.functional.softplus(-0.1*torch.stack(margins)).mean()
    actual = compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, 0.1, 8)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert policy.transitions.grad.abs().sum() > 0
    assert reference.transitions.grad is None


def test_dpo_gradient_increases_chosen_and_decreases_rejected_log_probability():
    chosen = torch.tensor([-4., -9.], requires_grad=True)
    rejected = torch.tensor([-8., -2.], requires_grad=True)
    compute_dpo_loss_from_logps(chosen, rejected, chosen.detach(), rejected.detach(), 0.1).backward()
    assert chosen.grad.lt(0).all()
    assert rejected.grad.gt(0).all()
    torch.testing.assert_close(chosen.grad, torch.full_like(chosen, -0.025))


def test_one_token_sequence_has_no_scoreable_causal_target():
    tokenizer = ToyTokenizer()
    model = TransitionLM(tokenizer.vocab_size)
    loss = compute_sft_loss(model, tokenizer, ['a'], ['b'], 1)
    assert loss.item() == 0
    loss.backward()
    assert model.transitions.grad.eq(0).all()
