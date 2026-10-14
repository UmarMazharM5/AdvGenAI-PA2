"""Resumable batch judging through the supported Kaggle model proxy.

The HTTP route is the same OpenAI-compatible route used by Kaggle Benchmarks
ModelProxy (kaggle_benchmarks/kaggle/model_proxy.py), not an external paid API.
In a Kaggle utility benchmark notebook, MODEL_PROXY_* are provided by Kaggle.
For local execution, setup_judge_auth.py provisions equivalent supported access.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT/'.tools').is_dir():
    sys.path.insert(0, str(ROOT/'.tools'))
from dotenv import load_dotenv
import requests
from alignment.evaluation import judge_prompt, parse_judgments, summarize


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def proxy_environment():
    load_dotenv(ROOT/'.env.benchmarks', override=False)
    url, key = os.getenv('MODEL_PROXY_URL'), os.getenv('MODEL_PROXY_API_KEY')
    if not url or not key:
        raise RuntimeError('Configure Kaggle model-proxy access first')
    for suffix in ('/openapi','/genai'):
        if url.endswith(suffix):
            url = url[:-len(suffix)]
    return url.rstrip('/')+'/openapi', key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs-path', required=True)
    p.add_argument('--output-path', required=True)
    p.add_argument('--judge-model', default='google/gemini-3.6-flash')
    p.add_argument('--batch-size', type=int, default=5)
    p.add_argument('--sleep-seconds', type=float, default=5)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--rubric', choices=['general','safety','persona'], default='general')
    a = p.parse_args()
    if not 1<=a.batch_size<=5 or a.sleep_seconds<5 or a.max_retries<1:
        raise ValueError('Use batches of 1-5, at least 5s spacing and at least one attempt')
    pairs = load_jsonl(a.pairs_path)
    if len({r['id'] for r in pairs})!=len(pairs):
        raise ValueError('Duplicate pair IDs')
    identities = load_jsonl(Path(a.pairs_path).with_suffix('.identities.jsonl'))
    out = Path(a.output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    config = {**vars(a), 'pairs_sha256':hashlib.sha256(Path(a.pairs_path).read_bytes()).hexdigest()}
    config_path = out.with_suffix('.config.json')
    def comparable(value):
        return {k:v for k,v in value.items() if k not in ('pairs_path', 'output_path')}
    if config_path.exists() and comparable(json.loads(config_path.read_text())) != comparable(config):
        raise ValueError('Resume config changed; choose a new output path')
    config_path.write_text(json.dumps(config,indent=2))
    judgments = load_jsonl(out) if out.exists() else []
    if judgments:
        parse_judgments(json.dumps(judgments), [r for r in pairs if r['id'] in {j['id'] for j in judgments}], a.rubric)
    done = {j['id'] for j in judgments}
    pending = [r for r in pairs if r['id'] not in done]
    if not pending:
        summary = summarize(pairs, judgments, identities)
        summary.update(judge_model=a.judge_model, rubric=a.rubric,
                       retried_successful_pairs=sum(j.get('attempts',1)>1 for j in judgments))
        out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
        return
    base, key = proxy_environment()
    session = requests.Session()
    last_attempt = 0
    last_auth_refresh = 0
    for start in range(0,len(pending),a.batch_size):
        batch = pending[start:start+a.batch_size]
        for attempt in range(1,a.max_retries+1):
            time.sleep(max(0,a.sleep_seconds-(time.monotonic()-last_attempt)))
            last_attempt = time.monotonic()
            try:
                response = session.post(base+'/chat/completions',
                    headers={'Authorization':'Bearer '+key},
                    json={'model':a.judge_model, 'messages':[{'role':'user','content':judge_prompt(batch,a.rubric)}]},
                    timeout=(20,180))
                if response.status_code == 401:
                    if not (ROOT/'.env').is_file() or time.monotonic()-last_auth_refresh<30:
                        raise PermissionError('Kaggle judge authentication denied after refresh; stopping requests')
                    from scripts.setup_judge_auth import refresh_credentials
                    os.environ.update(refresh_credentials())
                    base, key = proxy_environment()
                    last_auth_refresh = time.monotonic()
                    raise RuntimeError('Kaggle judge token renewed after HTTP 401; retrying saved batch')
                if not response.ok:
                    raise RuntimeError(f'Kaggle model proxy HTTP {response.status_code}')
                text = response.json()['choices'][0]['message']['content']
                with out.with_suffix('.raw_batches.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'ids':[r['id'] for r in batch], 'attempt':attempt,
                                        'response':text},ensure_ascii=False)+'\n')
                rows = parse_judgments(text,batch,a.rubric)
                with out.open('a',encoding='utf-8') as f:
                    for row in rows:
                        record = {**row,'attempts':attempt,'judge_model':a.judge_model}
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')
                        judgments.append(record)
                    f.flush()
                print(f'Judged {len(judgments)}/{len(pairs)}',flush=True)
                break
            except Exception as exc:
                failure = {'ids':[r['id'] for r in batch], 'attempt':attempt,
                           'error_type':type(exc).__name__, 'error':str(exc)[:300],
                           'unresolved':attempt==a.max_retries}
                with out.with_suffix('.failures.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(failure)+'\n')
                print(f"Judge attempt failed: {failure['error']}",flush=True)
                if isinstance(exc, PermissionError):
                    out.with_suffix('.summary.json').write_text(json.dumps(summarize(pairs,judgments,identities),indent=2))
                    raise
        summary = summarize(pairs,judgments,identities)
        summary.update(judge_model=a.judge_model, rubric=a.rubric,
                       retried_successful_pairs=sum(j.get('attempts',1)>1 for j in judgments))
        out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    if summarize(pairs, judgments, identities)['failures']:
        raise SystemExit('Unresolved judge failures; resume this output to retry missing pairs')


if __name__=='__main__':
    main()
