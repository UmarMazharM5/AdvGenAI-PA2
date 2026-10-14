"""Recompute stratified results and inspect completed-run selection evidence."""
import json
from pathlib import Path

from alignment.evaluation import parse_judgments, summarize

ROOT = Path(__file__).resolve().parents[1]


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    prompts = {'hh': ROOT/'eval_sets/hh_rlhf_eval_300_seed42.json',
               'uf': next((ROOT/'outputs/data_saved/sft-training').rglob('alpaca_eval_300_seed42.json')),
               'persona': ROOT/'eval_sets/persona_eval_seed42.json'}
    categories = {}
    audit = {'tracks': {}, 'checkpoints': {}, 'manual_sanity_review': {
        'review_timing': 'Post-run audit; controlled DPO/persona sanity cells originally proceeded automatically',
        'hh': 'First two harmful requests received unsafe assistance; relative wins do not establish absolute safety.',
        'uf': 'Summary invents 100+ domains where the supplied abstract says 59; bear answer lists living species and unrelated animals.',
        'persona': 'Cooking outputs lack target style, include unnecessary capability disclaimers and weak/truncated recipe advice.',
    }}
    for track, file in prompts.items():
        rows = json.loads(file.read_text(encoding='utf-8'))
        by_id = {str(r['id']): r for r in rows}
        generation = ROOT/f'outputs/generated/{track}/pa2/outputs/eval/{track}'
        pairs = lines(generation/'pairs.jsonl')
        mappings = lines(generation/'pairs.identities.jsonl')
        decisions = lines(ROOT/f'outputs/evaluation/generate-{track}/judged.jsonl')
        assert {r['id'] for r in pairs} == set(by_id)
        assert all(p['instruction'] == by_id[p['id']]['instruction'] for p in pairs)
        parse_judgments(json.dumps(decisions), pairs)
        groups = {}
        for row in rows:
            category = row.get('category', row.get('dataset', 'unknown'))
            groups.setdefault(category, set()).add(str(row['id']))
        categories[track] = {category: summarize([p for p in pairs if p['id'] in ids],
                                                 [j for j in decisions if j['id'] in ids], mappings)
                             for category, ids in sorted(groups.items())}
        audit['tracks'][track] = summarize(pairs, decisions, mappings)
        if track == 'persona':
            for decision in decisions:
                for dimension in ('style', 'usefulness', 'safety'):
                    for position in ('a', 'b'):
                        value = decision[f'{dimension}_{position}']
                        assert isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5
    for track, directory, run in [('sft', 'sft', 'sft'), ('hh', 'hh', 'dpo_hh'), ('uf', 'uf', 'dpo_uf')]:
        folder = ROOT/f'outputs/monitor/{directory}/artifacts/pa2/outputs/{run}'
        history = lines(folder/'metrics.jsonl')
        validation = [r for r in history if r['phase'] == 'validation']
        training = [r for r in history if r['phase'] == 'train']
        best = min(validation, key=lambda r: r['loss']) if track == 'sft' else max(validation, key=lambda r: r['reward_accuracy'])
        completed = json.loads((folder/'completed.json').read_text())
        assert completed['steps'] == len(training) == training[-1]['step']
        assert completed['best_score'] == (-best['loss'] if track == 'sft' else best['reward_accuracy'])
        audit['checkpoints'][track] = {'selected_step': best['step'], 'attempted_updates': len(training),
            'skipped_steps': [r['step'] for r in training if r['skipped_update']],
            'successful_updates': sum(not r['skipped_update'] for r in training)}
    supplied = json.loads((ROOT/'data/persona_lora/ramsay_style_preferences.json').read_text(encoding='utf-8'))
    assert len(supplied) == len({r['prompt'] for r in supplied}) == 220
    assert not {r['prompt'] for r in supplied} & {r['instruction'] for r in json.loads(prompts['persona'].read_text())}
    output = ROOT/'outputs/audit'
    output.mkdir(exist_ok=True)
    (output/'category_results.json').write_text(json.dumps(categories, indent=2))
    (output/'evidence_audit.json').write_text(json.dumps(audit, indent=2))
    uf_train_path=ROOT/'outputs/audit/data/uf_retry/pa2/outputs/data/uf_train.json'
    if uf_train_path.is_file():
        import hashlib
        manifest=json.loads(next((ROOT/'outputs/data_saved/dpo-uf').rglob('manifest_uf.json')).read_text())
        assert hashlib.sha256(uf_train_path.read_bytes()).hexdigest()==manifest['sha256']['uf_train.json']
        training={r['prompt'] for r in json.loads(uf_train_path.read_text(encoding='utf-8'))}
        evaluation=json.loads(prompts['uf'].read_text(encoding='utf-8'))
        overlaps={str(r['id']) for r in evaluation
                  if '### Instruction:\n'+r['instruction'].strip()+'\n\n### Response:\n' in training}
        folder=ROOT/'outputs/generated/uf/pa2/outputs/eval/uf'
        pairs=lines(folder/'pairs.jsonl')
        judgments=lines(ROOT/'outputs/evaluation/generate-uf/judged.jsonl')
        sensitivity={'overlap_ids':sorted(overlaps), 'definition':'Exact formatted UF training prompt match',
            'fixed_300':summarize(pairs,judgments,lines(folder/'pairs.identities.jsonl')),
            'excluding_training_matches':summarize([p for p in pairs if p['id'] not in overlaps],
                [j for j in judgments if j['id'] not in overlaps],lines(folder/'pairs.identities.jsonl'))}
        (output/'evaluation_overlap_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    print(json.dumps({'checkpoint_audit': audit['checkpoints'], 'category_outcomes': {
        t: {c: [s['wins'], s['ties'], s['losses']] for c, s in g.items()} for t, g in categories.items()}}, indent=2))


if __name__ == '__main__':
    main()
