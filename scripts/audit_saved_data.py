"""Audit exact saved training subsets in a CPU Kaggle notebook."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value).casefold()).strip()


def main():
    inputs = Path('/kaggle/input')
    output = Path('/kaggle/working/data_audit.json')
    tokenizer_paths = [p for p in inputs.rglob('sft/best/tokenizer_config.json')
                       if 'pa2-sft-training' in str(p)]
    assert len(tokenizer_paths) == 1
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_paths[0].parent, local_files_only=True)
    alpaca_paths = [p for p in inputs.rglob('alpaca_eval_300_seed42.json') if 'pa2-sft-training' in str(p)]
    assert len(alpaca_paths) == 1
    alpaca = json.loads(alpaca_paths[0].read_text())
    hh_eval = json.loads(Path('eval_sets/hh_rlhf_eval_300_seed42.json').read_text())
    eval_sets = {'alpaca': alpaca, 'hh_fixed': hh_eval}
    report = {'checks': {}, 'overlap_definition': 'Exact formatted prompt; also NFKC/casefold/whitespace normalized'}
    for track, notebook, train_count in [('sft', 'pa2-sft-training', 30000), ('hh', 'pa2-dpo-hh', 15000), ('uf', 'pa2-dpo-uf', 15000)]:
        paths = [p for p in inputs.rglob(f'{track}_train.json') if notebook in str(p)]
        assert len(paths) == 1, paths
        directory = paths[0].parent
        manifest = json.loads((directory/f'manifest_{track}.json').read_text())
        rows = {}
        checks = {}
        for split, count in [('train', train_count), ('validation', 1000)]:
            path = directory/f'{track}_{split}.json'
            rows[split] = json.loads(path.read_text())
            assert len(rows[split]) == count
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == manifest['sha256'][path.name]
            prompts = [r['prompt'] for r in rows[split]]
            assert len(set(prompts)) == count
            lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
            assert max(lengths) <= 256
            assert all(p.startswith('### Instruction:\n') and p.endswith('\n\n### Response:\n') for p in prompts)
            checks[split] = {'count': count, 'sha256': digest, 'unique_prompts': count,
                             'maximum_prompt_tokens': max(lengths)}
        train_prompts = {r['prompt'] for r in rows['train']}
        validation_prompts = {r['prompt'] for r in rows['validation']}
        assert not train_prompts & validation_prompts
        checks['train_validation_exact_overlap'] = 0
        checks['evaluation_overlap'] = {}
        for name, evaluation in eval_sets.items():
            formatted = {'### Instruction:\n'+r['instruction'].strip()+'\n\n### Response:\n' for r in evaluation}
            checks['evaluation_overlap'][name] = {
                split: {'exact': len({r['prompt'] for r in rows[split]} & formatted),
                        'normalized': len({normalized(r['prompt']) for r in rows[split]} & {normalized(p) for p in formatted})}
                for split in ('train', 'validation')}
        report['checks'][track] = checks
        print(track, json.dumps(checks), flush=True)
    output.write_text(json.dumps(report, indent=2))
    print('All saved subset count, hash, uniqueness, prompt-format and token-length checks passed')


if __name__ == '__main__':
    main()
