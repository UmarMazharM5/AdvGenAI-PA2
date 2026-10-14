"""Recover unresolved batch judgments as individual saved-pair requests.

Waits for the main serial judge queue to finish before making requests. Existing
valid judgments remain unchanged; every recovered decision keeps its provenance.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alignment.evaluation import summarize


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s] if path.exists() else []


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hours',type=float,default=2)
    a=p.parse_args()
    end=time.monotonic()+a.hours*3600
    while time.monotonic()<end:
        state=json.loads((ROOT/'outputs/pipeline/status.json').read_text())
        jobs=[j for j in state['jobs'] if j['name'].startswith('generate-')]
        if all('judge_exit_code' in j for j in jobs):
            break
        time.sleep(60)
    else:
        raise TimeoutError('Main judge queue has not finished')
    for track in ('uf','hh','persona'):
        original=ROOT/f'outputs/evaluation/generate-{track}/judged.jsonl'
        pair_path=next((ROOT/f'outputs/generated/{track}').rglob('pairs.jsonl'))
        pairs=read(pair_path)
        identities=read(pair_path.with_suffix('.identities.jsonl'))
        mappings={r['id']:r for r in identities}
        judgments=read(original)
        done={r['id'] for r in judgments}
        for pair in pairs:
            if pair['id'] in done:
                continue
            digest=hashlib.sha256(pair['id'].encode()).hexdigest()[:16]
            folder=ROOT/f'outputs/recovered_judgments/{track}/{digest}'
            folder.mkdir(parents=True,exist_ok=True)
            mini=folder/'pairs.jsonl'
            mini.write_text(json.dumps(pair,ensure_ascii=False)+'\n',encoding='utf-8')
            mini.with_suffix('.identities.jsonl').write_text(json.dumps(mappings[pair['id']])+'\n')
            target=folder/'judged.jsonl'
            subprocess.run([sys.executable,'-X','utf8','scripts/judge_pairwise_kaggle.py',
                '--pairs-path',str(mini),'--output-path',str(target),'--batch-size','1',
                '--rubric','safety' if track=='hh' else ('persona' if track=='persona' else 'general')],cwd=ROOT,check=True)
            # Each child has its own request timer; retain spacing across children.
            time.sleep(5)
            recovered=read(target)
            if len(recovered)==1 and recovered[0]['id']==pair['id']:
                record={**recovered[0],'recovered_single_pair':True,'request_batch_size':1}
                with original.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                judgments.append(record)
                done.add(pair['id'])
            result=summarize(pairs,judgments,identities)
            result.update(judge_model='google/gemini-3.6-flash',
                          rubric='safety' if track=='hh' else ('persona' if track=='persona' else 'general'),
                          retried_successful_pairs=sum(j.get('attempts',1)>1 for j in judgments),
                          recovered_single_pairs=sum(j.get('recovered_single_pair',False) for j in judgments))
            original.with_suffix('.summary.json').write_text(json.dumps(result,indent=2))
        print(f'{track}: {len(done)}/{len(pairs)} judgments resolved',flush=True)


if __name__=='__main__':
    main()
