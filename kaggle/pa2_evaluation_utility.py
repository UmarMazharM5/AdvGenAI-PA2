# %%
"""Kaggle utility benchmark: resume saved evidence and check fresh blinded calls.

The original 620 judgments were produced with the supported Kaggle proxy locally.
This notebook preserves those completed decisions; fifteen new native judgments
are an interface/replication audit and are stored separately, without selection.
No model generations, checkpoint weights or credentials are included.
"""
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import kaggle_benchmarks as kbench


@kbench.task(name='pa2-saved-pairwise-evaluation')
def pa2_saved_pairwise_evaluation(llm: kbench.LLMChat) -> float:
    candidates = list(Path('/kaggle/input').rglob('evidence_manifest.json'))
    assert len(candidates) == 1, 'Attach the PA2 saved evaluation evidence dataset'
    source = candidates[0].parent
    # Kaggle may mount uploaded subdirectories as ZIP files instead of unpacking them.
    if not (source/'hh/pairs.jsonl').is_file():
        staging = Path('/kaggle/working/saved_evidence')
        staging.mkdir(parents=True, exist_ok=True)
        for file in source.glob('*'):
            if file.is_file() and file.suffix != '.zip':
                shutil.copy2(file, staging/file.name)
        for track in ('hh', 'uf', 'persona'):
            target = staging/track
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(source/f'{track}.zip') as archive:
                for member in archive.namelist():
                    if not member.endswith('/'):
                        (target/Path(member).name).write_bytes(archive.read(member))
        source = staging
    manifest = json.loads(candidates[0].read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(source))
    from evaluation import judge_prompt, parse_judgments, summarize

    output = Path('/kaggle/working/pa2_evaluation')
    output.mkdir(parents=True, exist_ok=True)
    summaries, audit_summaries = {}, {}
    last_attempt = 0
    for track, expected, rubric in [('hh', 300, 'safety'), ('uf', 300, 'general'), ('persona', 20, 'persona')]:
        folder = output/track
        shutil.copytree(source/track, folder, dirs_exist_ok=True)
        def read(name):
            return [json.loads(s) for s in (folder/name).read_text().splitlines() if s.strip()]
        pairs, mappings, decisions = read('pairs.jsonl'), read('pairs.identities.jsonl'), read('judged.jsonl')
        assert len(pairs) == len(mappings) == len(decisions) == expected
        assert len({r['id'] for r in decisions}) == expected
        parse_judgments(json.dumps(decisions), pairs)
        summary = summarize(pairs, decisions, mappings)
        original = json.loads((folder/'judged.summary.json').read_text())
        assert all(original[key] == value for key, value in summary.items())
        assert summary['failures'] == summary['unknown'] == 0
        summaries[track] = {**original, 'utility_resumed_pairs': expected}
        (folder/'judged.summary.json').write_text(json.dumps(summaries[track], indent=2))

        # Fixed first five; no filtering by old winner or response quality.
        batch = pairs[:5]
        audit_path = folder/'native_audit.jsonl'
        audit = read('native_audit.jsonl') if audit_path.exists() else []
        remaining = [p for p in batch if p['id'] not in {j['id'] for j in audit}]
        for attempt in range(1, 4) if remaining else []:
            time.sleep(max(0, 5-(time.monotonic()-last_attempt)))
            last_attempt = time.monotonic()
            try:
                with kbench.chats.new(name=f'{track} blinded audit attempt {attempt}'):
                    text = llm.prompt(judge_prompt(remaining, rubric), temperature=1.0)
                with (folder/'native_audit.raw_batches.jsonl').open('a') as stream:
                    stream.write(json.dumps({'ids': [p['id'] for p in remaining], 'attempt': attempt, 'response': text})+'\n')
                rows = parse_judgments(text, remaining)
                with audit_path.open('a') as stream:
                    for row in rows:
                        row.update(attempts=attempt, judge_model=llm.name, execution='native_kaggle_utility')
                        stream.write(json.dumps(row)+'\n')
                        audit.append(row)
                break
            except Exception as exc:
                # Never serialize request headers, credential values or exception payloads.
                with (folder/'native_audit.failures.jsonl').open('a') as stream:
                    stream.write(json.dumps({'ids': [p['id'] for p in remaining], 'attempt': attempt,
                        'error_type': type(exc).__name__, 'unresolved': attempt == 3})+'\n')
                print(f'{track} audit attempt {attempt} failed: {type(exc).__name__}', flush=True)
        audit_summary = summarize(batch, audit, mappings)
        (folder/'native_audit.summary.json').write_text(json.dumps(audit_summary, indent=2))
        audit_summaries[track] = audit_summary
        print(track, json.dumps(summaries[track]), json.dumps(audit_summary), flush=True)
    provenance = {'resumed_decisions': 620, 'new_audit_pairs': 15,
                  'primary_scores_changed': False, 'judge_model': llm.name,
                  'source_manifest_sha256': hashlib.sha256(candidates[0].read_bytes()).hexdigest(),
                  'summaries': summaries, 'audit_summaries': audit_summaries}
    (output/'utility_completed.json').write_text(json.dumps(provenance, indent=2))
    assert all(s['failures'] == 0 for s in audit_summaries.values()), 'Unresolved native audit judgments'
    return 1.0


# %%
run = pa2_saved_pairwise_evaluation.run(llm=kbench.llm)
print('Utility evaluation completed; files saved under /kaggle/working/pa2_evaluation')
