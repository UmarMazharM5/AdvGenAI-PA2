"""Original PyTorch SFT/DPO loop over normalized JSON rows.

Run as python -m scripts.train. SFT rows: prompt,response. DPO rows:
prompt,chosen,rejected. Prompts must already contain the documented template.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from alignment.core import build_sft_batch, build_dpo_batch, score_batch, compute_dpo_loss_from_logps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['sft', 'dpo'], required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--accumulation', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=0, help='Pilot only; 0 runs the full dataset')
    a = p.parse_args()
    dpo = a.task == 'dpo'
    a.batch_size = a.batch_size or (1 if dpo else 2)
    a.accumulation = a.accumulation or (32 if dpo else 16)
    a.learning_rate = a.learning_rate or (1e-6 if dpo else 2e-5)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Run training in the Kaggle GPU notebook')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to('cuda:0')
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    reference = None
    if dpo:
        if torch.cuda.device_count() < 2:
            raise RuntimeError('DPO requires policy on cuda:0 and reference on cuda:1')
        reference = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to('cuda:1')
        reference.requires_grad_(False).eval()
    def collate(rows):
        prompts = [r['prompt'] for r in rows]
        if dpo:
            return build_dpo_batch(tokenizer, prompts, [r['chosen'] for r in rows],
                                   [r['rejected'] for r in rows], a.max_length)
        return build_sft_batch(tokenizer, prompts, [r['response'] for r in rows], a.max_length)
    train_rows = json.loads(Path(a.train).read_text())
    val_rows = json.loads(Path(a.validation).read_text())
    # Refuse silent no-response training examples after truncation.
    for rows in (train_rows, val_rows):
        for row in rows:
            b = collate([row])
            sides = b.values() if dpo else [b]
            if any(not side['labels'][:, 1:].ne(-100).any() for side in sides):
                raise ValueError('An example has no scoreable response after truncation')
    train = DataLoader(train_rows, batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    validation = DataLoader(val_rows, batch_size=a.batch_size, collate_fn=collate)
    total = math.ceil(len(train) / a.accumulation) * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=a.learning_rate, weight_decay=0.1)
    scheduler = None if dpo else get_cosine_schedule_with_warmup(optimizer, round(total*0.03), total)
    scaler = torch.amp.GradScaler('cuda')

    def objective(batch):
        with torch.autocast('cuda', dtype=torch.float16):
            if not dpo:
                logps, counts = score_batch(policy, batch)
                return -logps.sum()/counts.sum().clamp_min(1), {'tokens': counts.sum().item()}
            pc = score_batch(policy, batch['chosen'])[0]
            pr = score_batch(policy, batch['rejected'])[0]
            with torch.no_grad():
                rc = score_batch(reference, batch['chosen'])[0].to(pc.device)
                rr = score_batch(reference, batch['rejected'])[0].to(pc.device)
            margin = a.beta * ((pc-rc) - (pr-rr))
            return compute_dpo_loss_from_logps(pc, pr, rc, rr, a.beta), {
                'reward_accuracy': margin.gt(0).float().mean().item(),
                'reward_margin': margin.mean().item(), 'pairs': len(pc)}

    def log(record):
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    def save(name):
        policy.save_pretrained(out/name, safe_serialization=True)
        tokenizer.save_pretrained(out/name)

    best = -float('inf')
    def evaluate(step):
        nonlocal best
        policy.eval()
        sums, weight = {}, 0
        with torch.no_grad():
            for batch in validation:
                loss, metrics = objective(batch)
                n = metrics.pop('pairs' if dpo else 'tokens')
                weight += n
                for key, value in {'loss': loss.item(), **metrics}.items():
                    sums[key] = sums.get(key, 0) + value*n
        values = {k: v/max(weight, 1) for k,v in sums.items()}
        log({'phase': 'validation', 'step': step, **values})
        score = values['reward_accuracy'] if dpo else -values['loss']
        if score > best:
            best = score
            save('best')
        policy.train()

    step = 0
    evaluate(step)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(a.epochs):
        window_loss = 0
        for i, batch in enumerate(train):
            # Normalize the last partial accumulation window by its actual size.
            start = (i//a.accumulation)*a.accumulation
            window_size = min(a.accumulation, len(train)-start)
            loss, metrics = objective(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss/window_size).backward()
            window_loss += loss.item()/window_size
            if (i+1)%a.accumulation == 0 or i+1 == len(train):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                if scheduler is not None and not skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log({'phase': 'train', 'step': step, 'epoch': epoch,
                     'loss': window_loss, 'lr': optimizer.param_groups[0]['lr'],
                     'gradient_norm': float(norm), 'skipped_update': skipped, **metrics})
                window_loss = 0
                if step%a.eval_steps == 0:
                    evaluate(step)
                if step >= total:
                    break
        if step >= total:
            break
    if step%a.eval_steps:
        evaluate(step)
    save('final')
    (out/'completed.json').write_text(json.dumps({'steps': step, 'best_score': best}))


if __name__ == '__main__':
    main()
