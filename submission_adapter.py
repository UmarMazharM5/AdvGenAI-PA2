"""Stable grading interface; signatures are defined in alignment.core."""
from alignment.core import (
    build_sft_batch,
    compute_response_logprobs,
    compute_sft_loss,
    build_dpo_batch,
    compute_dpo_loss_from_logps,
    compute_dpo_loss,
)
