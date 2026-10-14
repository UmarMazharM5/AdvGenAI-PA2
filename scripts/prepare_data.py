"""Create disjoint, single-turn train/validation subsets and fixed eval prompts."""
import hashlib
import argparse
import json
import random
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def format_prompt(instruction):
    return f'### Instruction:\n{instruction.strip()}\n\n### Response:\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', nargs='+', choices=['sft', 'hh', 'uf'], default=['sft', 'hh', 'uf'])
    args = parser.parse_args()

    def retry_download(operation):
        for attempt in range(6):
            try:
                return operation()
            except Exception as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                transient = status in (429, 500, 502, 503, 504) or any(
                    text in str(exc).lower() for text in ('429', 'too many requests', 'timed out', 'connection error'))
                if not transient or attempt == 5:
                    raise
                delay = min(60, 15 * (2 ** attempt))
                print(f'Temporary download failure; retry {attempt+1}/5 in {delay}s', flush=True)
                time.sleep(delay)
    root = Path('outputs/data')
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = retry_download(lambda: AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'))
    def normalize(row):
        row = dict(row)
        row['prompt'] = format_prompt(row['prompt'])
        # Cap training prompts to 256 tokens to reserve a response budget.
        ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        if len(ids) > 256:
            return None
        return row

    def collect(dataset, convert, count, excluded=None):
        rows, seen = [], set(excluded or [])
        for raw in dataset.shuffle(seed=42):
            row = convert(raw)
            if row is None:
                continue
            row = normalize(row)
            if row is None:
                continue
            key = hashlib.sha256(row['prompt'].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == count:
                return rows, seen
        raise RuntimeError(f'Only {len(rows)} usable rows; expected {count}')

    def save(name, rows):
        (root/f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')

    def chat(row):
        m = row['messages']
        if len(m) < 2 or m[0]['role'] != 'user' or m[1]['role'] != 'assistant':
            return None
        return {'prompt': m[0]['content'], 'response': m[1]['content']}

    def uf(row):
        c, r = row['chosen'], row['rejected']
        if len(c) != 2 or len(r) != 2:
            return None
        if c[0]['role'] != 'user' or r[0]['role'] != 'user' or c[0]['content'] != r[0]['content']:
            return None
        if c[1]['role'] != 'assistant' or r[1]['role'] != 'assistant':
            return None
        return {'prompt': c[0]['content'], 'chosen': c[1]['content'], 'rejected': r[1]['content']}

    def hh(row):
        marker = '\n\nAssistant:'
        chosen, rejected = row['chosen'], row['rejected']
        if chosen.count('\n\nHuman:') != 1 or rejected.count('\n\nHuman:') != 1:
            return None
        if chosen.count(marker) != 1 or rejected.count(marker) != 1:
            return None
        cp, ca = chosen.split(marker)
        rp, ra = rejected.split(marker)
        if cp != rp:
            return None
        return {'prompt': cp.replace('\n\nHuman:', '', 1).strip(),
                'chosen': ca.strip(), 'rejected': ra.strip()}

    sources = [
        ('sft','HuggingFaceH4/ultrachat_200k','train_sft','test_sft',chat,30000),
        ('hh','Anthropic/hh-rlhf','train','test',hh,15000),
        ('uf','HuggingFaceH4/ultrafeedback_binarized','train_prefs','test_prefs',uf,15000),
    ]
    manifest = {'seed': 42, 'prompt_template': '### Instruction:\\n{instruction}\\n\\n### Response:\\n',
                'max_prompt_tokens': 256, 'filter': 'SFT first user/assistant turn; DPO single-turn only; unique prompts; disjoint validation', 'sources': []}
    for name, source, train_split, val_split, convert, n in sources:
        if name not in args.tracks:
            continue
        train_data = retry_download(lambda: load_dataset(source, split=train_split))
        val_data = retry_download(lambda: load_dataset(source, split=val_split))
        train, seen = collect(train_data, convert, n)
        val, _ = collect(val_data, convert, 1000, seen)
        save(f'{name}_train', train)
        save(f'{name}_validation', val)
        manifest['sources'].append({'name': source, 'train_split': train_split, 'validation_split': val_split,
                                    'train_count': len(train), 'validation_count': len(val),
                                    'fingerprints': {train_split: train_data._fingerprint, val_split: val_data._fingerprint}})
        print(f'{name}: saved {len(train)} training and {len(val)} validation rows', flush=True)
    eval_file = Path(retry_download(lambda: hf_hub_download('tatsu-lab/alpaca_eval', 'alpaca_eval.json', repo_type='dataset')))
    data = json.loads(eval_file.read_text(encoding='utf-8'))
    indices = random.Random(42).sample(range(len(data)), 300)
    save('alpaca_eval_300_seed42', [{'id': str(i), 'instruction': data[i]['instruction'],
                                   'dataset': data[i].get('dataset','alpaca_eval')} for i in indices])
    manifest['alpaca_eval_source_sha256'] = hashlib.sha256(eval_file.read_bytes()).hexdigest()
    for file in sorted(root.glob('*.json')):
        manifest.setdefault('sha256', {})[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (root/('manifest_'+'_'.join(args.tracks)+'.json')).write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
