"""Deterministic generation with per-prompt atomic checkpoints and resume."""
import argparse
import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.prepare_data import format_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--generator', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--no-repeat-ngram-size', type=int, default=4)
    p.add_argument('--chat-template', action='store_true', help='For SmolLM2 persona evaluation only')
    a = p.parse_args()
    torch.manual_seed(42)
    rows = json.loads(Path(a.prompts).read_text(encoding='utf-8'))
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding='utf-8')) if out.exists() else []
    settings = {**vars(a), 'seed':42, 'do_sample':False,
                'prompts_sha256':hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    settings_path = out.with_suffix('.settings.json')
    if results:
        if not settings_path.is_file():
            raise ValueError('Cannot safely resume outputs without saved generation settings')
        old_settings = json.loads(settings_path.read_text())
        compare_keys = set(settings)-{'output', 'prompts', 'prompts_sha256'}
        if any(old_settings.get(k) != settings[k] for k in compare_keys):
            raise ValueError('Generation settings changed; use a new output file')
        if 'prompts_sha256' in old_settings and old_settings['prompts_sha256'] != settings['prompts_sha256']:
            raise ValueError('Evaluation prompts changed; use a new output file')
    by_id = {str(r.get('id', i)):r for i,r in enumerate(rows)}
    if len(by_id) != len(rows) or len({r['id'] for r in results}) != len(results):
        raise ValueError('Duplicate evaluation or saved generation IDs')
    if any(r['id'] not in by_id or r['instruction'] != by_id[r['id']].get('instruction', by_id[r['id']].get('prompt'))
           or r['generator'] != a.generator for r in results):
        raise ValueError('Saved generations do not match the current evaluation inputs')
    settings_path.write_text(json.dumps(settings, indent=2))
    done = {r['id'] for r in results}
    if len(done) == len(rows):
        print(f'{a.generator}: all {len(rows)} saved generations already complete')
        return
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to('cuda:0').eval()
    for i, row in enumerate(rows):
        identifier = str(row.get('id', i))
        if identifier in done:
            continue
        instruction = row.get('instruction', row.get('prompt'))
        if not isinstance(instruction, str):
            raise ValueError('Evaluation row is missing its instruction')
        prompt = tokenizer.apply_chat_template([{'role':'user','content':instruction}], tokenize=False,
                                               add_generation_prompt=True) if a.chat_template else format_prompt(instruction)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                                    repetition_penalty=a.repetition_penalty,
                                    no_repeat_ngram_size=a.no_repeat_ngram_size)
        text = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append({'id': identifier, 'instruction': instruction, 'output': text,
                        'generator': a.generator, 'dataset': row.get('dataset', Path(a.prompts).stem)})
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(out)
        print(f'{a.generator} {len(results)}/{len(rows)}: {text[:160]!r}', flush=True)


if __name__ == '__main__':
    main()
