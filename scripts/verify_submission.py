"""Check the packaged evidence, generation consistency and credential exclusion."""
import json
import sys
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.tools'))
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values
from alignment.evaluation import summarize, parse_judgments


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]


def main():
    bundle=ROOT if (ROOT/'results/generations').is_dir() else ROOT/'submission/PA2_project'
    secrets=[]
    for env in (ROOT/'.env',ROOT/'.env.benchmarks'):
        if env.exists():
            secrets.extend(str(v).encode() for k,v in dotenv_values(env).items()
                           if v and k in ('KAGGLE_API_TOKEN','KAGGLE_KEY','MODEL_PROXY_API_KEY','GH_TOKEN','GITHUB_TOKEN'))
    manifest=json.loads((bundle/'artifact_manifest.json').read_text())
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
            and '.git' not in p.parts}
    expected=set(manifest)|{'artifact_manifest.json'}
    assert actual==expected, f'Unmanifested or missing files: {actual.symmetric_difference(expected)}'
    for name,expected_hash in manifest.items():
        path=bundle/name
        assert path.is_file(), f'Missing packaged artifact: {name}'
        data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected_hash, f'Changed artifact: {name}'
        assert not any(value in data for value in secrets), f'Credential detected in {name}'
        assert path.name!='.env' and (not path.name.startswith('.env.') or path.name=='.env.example')
    archive=bundle.parent/'PA2_submission.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==len(set(z.namelist())), 'Duplicate archive filenames'
            assert set(z.namelist())=={bundle.name+'/'+name for name in expected}
            for name in expected:
                data=z.read(bundle.name+'/'+name)
                assert data==(bundle/name).read_bytes(), f'Archive differs from bundle: {name}'
                assert not any(value in data for value in secrets), f'Credential detected in archive: {name}'
    for track,expected in [('hh',300),('uf',300),('persona',20)]:
        root=bundle/f'results/generations/{track}'
        pairs=lines(root/'pairs.jsonl')
        identities=lines(root/'pairs.identities.jsonl')
        judgments=lines(bundle/f'results/judgments/{track}/judged.jsonl')
        assert len(pairs)==len(identities)==len(judgments)==expected
        assert len({r['id'] for r in judgments})==expected
        assert {r['id'] for r in pairs}=={r['id'] for r in identities}=={r['id'] for r in judgments}
        mapping={r['id']:r for r in identities}
        generated={}
        settings=[]
        for file in root.glob('*.json'):
            if file.name.endswith('.settings.json'):
                settings.append(json.loads(file.read_text()))
            else:
                rows=json.loads(file.read_text(encoding='utf-8'))
                if isinstance(rows,list) and rows:
                    generated[rows[0]['generator']]={r['id']:r for r in rows}
        for pair in pairs:
            for position,field in [('A','response_a'),('B','response_b')]:
                source=generated[mapping[pair['id']][position]][pair['id']]
                assert source['instruction']==pair['instruction']
                assert source['output']==pair[field]
        comparable=[(s['max_new_tokens'],s['do_sample'],s['repetition_penalty'],s['no_repeat_ngram_size'],s['chat_template']) for s in settings]
        assert len(set(comparable))==1, f'Different decoding settings in {track}'
        recomputed=summarize(pairs,judgments,identities)
        stored=json.loads((bundle/f'results/judgments/{track}/judged.summary.json').read_text())
        assert all(stored[k]==v for k,v in recomputed.items()), f'Summary mismatch in {track}'
    utility_files=[p for p in (bundle/'results/utility').rglob('utility_completed.json')
                   if json.loads(p.read_text()).get('judge_model')=='google/gemini-3.6-flash']
    assert len(utility_files)==1, 'Missing or ambiguous native Gemini 3.6 utility evidence'
    utility=json.loads(utility_files[0].read_text())
    assert utility['resumed_decisions']==620 and utility['new_audit_pairs']==15
    assert utility['primary_scores_changed'] is False
    for track in ('hh','uf','persona'):
        native=utility_files[0].parent/track
        primary=bundle/f'results/generations/{track}'
        assert (native/'pairs.jsonl').read_bytes()==(primary/'pairs.jsonl').read_bytes()
        assert (native/'judged.jsonl').read_bytes()==(bundle/f'results/judgments/{track}/judged.jsonl').read_bytes()
        audit=lines(native/'native_audit.jsonl')
        pairs=lines(native/'pairs.jsonl')[:5]
        parse_judgments(json.dumps(audit),pairs,'persona' if track=='persona' else 'general')
        assert all(j['execution']=='native_kaggle_utility' and j['judge_model']=='google/gemini-3.6-flash' for j in audit)
        native_summary=summarize(pairs,audit,lines(native/'pairs.identities.jsonl'))
        assert native_summary==utility['audit_summaries'][track]
    data=json.loads((bundle/'results/audit/data_audit.json').read_text())
    assert set(data['checks'])=={'sft','hh','uf'}
    for track,checks in data['checks'].items():
        assert checks['train_validation_exact_overlap']==0
        assert checks['train']['count']==(30000 if track=='sft' else 15000)
        assert checks['validation']['count']==1000
        assert all(checks[split]['maximum_prompt_tokens']<=256 for split in ('train','validation'))
    sensitivity=json.loads((bundle/'results/audit/evaluation_overlap_sensitivity.json').read_text())
    pairs=lines(bundle/'results/generations/uf/pairs.jsonl')
    judgments=lines(bundle/'results/judgments/uf/judged.jsonl')
    overlap=set(sensitivity['overlap_ids'])
    assert len(overlap)==data['checks']['uf']['evaluation_overlap']['alpaca']['train']['exact']==2
    assert summarize([p for p in pairs if p['id'] not in overlap],
                     [j for j in judgments if j['id'] not in overlap],
                     lines(bundle/'results/generations/uf/pairs.identities.jsonl'))==sensitivity['excluding_training_matches']
    print(f'Verified {len(manifest)} credential-free artifacts and all 620 blinded judgments')


if __name__=='__main__':
    main()
