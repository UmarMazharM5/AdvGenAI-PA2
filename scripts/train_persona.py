"""The assignment's permitted TRL + PEFT experiment, kept separate from core loops.

API references: TRL v0.24.0 DPOTrainer; PEFT LoRAConfig documentation.
"""
import os
# TRL's implicit DataParallel replication conflicts with checkpointing/LoRA
# on this runtime. The 360M adapter experiment fits comfortably on one T4.
# Set before importing torch so the trainer sees exactly one CUDA device.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    output = Path('outputs/persona')
    output.mkdir(parents=True, exist_ok=True)
    name = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    rows = json.loads(Path('data/persona_lora/ramsay_style_preferences.json').read_text(encoding='utf-8'))
    conversational = [
        {'prompt': [{'role':'user', 'content':r['prompt']}],
         'chosen': [{'role':'assistant', 'content':r['chosen']}],
         'rejected': [{'role':'assistant', 'content':r['rejected']}]}
        for r in rows
    ]
    split = Dataset.from_list(conversational).train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(str(output/'split'))
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                      task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])
    config = DPOConfig(
        output_dir=str(output/'checkpoints'), num_train_epochs=3,
        learning_rate=5e-5, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, max_length=512, max_prompt_length=256,
        beta=0.1, fp16=True, bf16=False, gradient_checkpointing=True,
        optim='adamw_torch', lr_scheduler_type='cosine', warmup_ratio=0.03,
        weight_decay=0.0, max_grad_norm=1.0, eval_strategy='epoch',
        save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='eval_rewards/accuracies', greater_is_better=True,
        logging_steps=5, report_to='none', seed=42, data_seed=42,
        disable_dropout=True, precompute_ref_log_probs=True,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=config,
                         train_dataset=split['train'], eval_dataset=split['test'],
                         processing_class=tokenizer, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(output/'best_adapter'))
    tokenizer.save_pretrained(output/'best_adapter')
    (output/'metrics.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    (output/'config.json').write_text(json.dumps({
        'model':name, 'reference':'base SmolLM2 with adapter disabled',
        'train_count':len(split['train']), 'validation_count':len(split['test']),
        'training':config.to_dict(), 'lora':lora.to_dict(),
        'rationale':'Rank 16 gives moderate style capacity on a small dataset; alpha/r=2. '
                    'Attention projections cover query, key, value and output changes. '
                    'LR 5e-5 and three epochs are an initial conservative adapter recipe; '
                    'choose the best validation reward accuracy checkpoint.'},
        indent=2, default=str))
    # A merged checkpoint makes generation independent of adapter loading details.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output/'best', safe_serialization=True)
    tokenizer.save_pretrained(output/'best')
    (output/'completed.json').write_text(json.dumps({'best_checkpoint':trainer.state.best_model_checkpoint}))


if __name__ == '__main__':
    main()
