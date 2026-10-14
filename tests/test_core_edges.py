import math
import torch
from tests.toy_lm import ToyTokenizer, ToyCausalLM
from alignment.core import build_sft_batch, compute_response_logprobs, compute_sft_loss


def test_eos_removed_by_truncation_and_rows_stay_ordered():
    t = ToyTokenizer()
    b = build_sft_batch(t, ['aa', 'b'], ['cdef', 'd'], 4)
    assert b['labels'][0].tolist() == [-100, -100, t.vocab['c'], t.vocab['d']]
    assert b['labels'][1].tolist() == [-100, t.vocab['d'], t.eos_token_id, -100]


def test_uniform_scores_have_exact_response_length_scaling():
    t = ToyTokenizer()
    m = ToyCausalLM(t.vocab_size)
    scores = compute_response_logprobs(m, t, ['a', 'bb'], ['c', 'de'], 10)
    torch.testing.assert_close(scores, -math.log(t.vocab_size)*torch.tensor([2., 3.]))


def test_all_masked_loss_is_zero_and_finite():
    t = ToyTokenizer()
    m = ToyCausalLM(t.vocab_size)
    loss = compute_sft_loss(m, t, ['abcdef'], ['x'], 3)
    assert loss.item() == 0
    loss.backward()
    assert m.logit_bias.grad.eq(0).all()


def test_eos_can_also_be_padding_without_masking_real_eos():
    t = ToyTokenizer()
    t.pad_token_id = t.eos_token_id
    b = build_sft_batch(t, ['a','bb'], ['c','dd'], 10)
    assert b['labels'][0,2] == t.eos_token_id
    assert b['labels'][0,3] == -100
