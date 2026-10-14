"""Blind randomized A/B pairs, with a separate identity mapping."""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    baseline = json.loads(Path(a.baseline).read_text(encoding='utf-8'))
    candidate = json.loads(Path(a.candidate).read_text(encoding='utf-8'))
    def index(rows):
        result = {r['id']:r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate generation IDs')
        return result
    b, c = index(baseline), index(candidate)
    if b.keys() != c.keys():
        raise ValueError('Models must have exactly the same prompt IDs')
    rng = random.Random(42)
    pairs, identities = [], []
    for identifier, left in b.items():
        right = c[identifier]
        if left['instruction'] != right['instruction']:
            raise ValueError('Prompt mismatch')
        first, second = (left,right) if rng.random()<0.5 else (right,left)
        pairs.append({'id':identifier, 'instruction':left['instruction'],
                      'response_a':first['output'], 'response_b':second['output']})
        identities.append({'id':identifier, 'A':first['generator'], 'B':second['generator'],
                           'candidate':right['generator'], 'baseline':left['generator']})
    out = Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf-8')
    out.with_suffix('.identities.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in identities))


if __name__ == '__main__':
    main()
