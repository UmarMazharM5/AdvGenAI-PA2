"""Build an explicit allowlist of code and audited artifacts; exclude all secrets."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='submission/PA2_project')
    a=parser.parse_args()
    dest=ROOT/a.output
    dest.mkdir(parents=True,exist_ok=True)
    allowed=[]
    for name in ('README.md','RUN_STATUS.md','SUBMISSION_CHECKLIST.md','requirements.txt','requirements-persona.txt','requirements-local.txt','pyproject.toml',
                 'submission_adapter.py','.gitignore','.env.example'):
        allowed.append((ROOT/name,Path(name)))
    for name in ('alignment','scripts','tests','notebooks','kaggle','data','eval_sets','report'):
        for file in (ROOT/name).rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix in ('.py','.json','.jsonl','.md','.txt','.toml','.ipynb','.pdf'):
                if name=='report' and file.name=='PA2_report_DRAFT.pdf' and (ROOT/'report/PA2_report.pdf').is_file():
                    continue
                allowed.append((file,file.relative_to(ROOT)))
    for file in (ROOT/'outputs/plots').glob('*'):
        if file.suffix in ('.png','.pdf'):
            allowed.append((file,Path('results/plots')/file.name))
    for file in (ROOT/'outputs/data_saved').rglob('*.json'):
        allowed.append((file,Path('results/data')/file.relative_to(ROOT/'outputs/data_saved')))
    for track,folder,run in [('sft','sft','sft'),('hh','hh','dpo_hh'),('uf','uf','dpo_uf'),('persona','persona_v2','persona')]:
        source=ROOT/f'outputs/monitor/{folder}/artifacts/pa2/outputs/{run}'
        for file in source.glob('*'):
            if file.is_file() and file.name in ('config.json','metrics.json','metrics.jsonl','completed.json'):
                allowed.append((file,Path('results/training')/track/file.name))
    for track in ('hh','uf','persona'):
        sanity=ROOT/f'outputs/generated/{track}/pa2/outputs'
        for directory in sanity.glob('sanity*'):
            for file in directory.glob('*.json'):
                allowed.append((file,Path('results/sanity')/track/file.name))
        source=ROOT/f'outputs/generated/{track}/pa2/outputs/eval/{track}'
        for file in source.glob('*'):
            if file.suffix in ('.json','.jsonl'):
                allowed.append((file,Path('results/generations')/track/file.name))
        for file in (ROOT/f'outputs/evaluation/generate-{track}').glob('*'):
            if file.suffix in ('.json','.jsonl','.log'):
                allowed.append((file,Path('results/judgments')/track/file.name))
    for file in (ROOT/'outputs/recovered_judgments').rglob('*.jsonl'):
        allowed.append((file,Path('results/judgment_recovery')/file.relative_to(ROOT/'outputs/recovered_judgments')))
    for file in (ROOT/'outputs/audit').glob('*.json'):
        allowed.append((file,Path('results/audit')/file.name))
    for file in (ROOT/'outputs/utility_saved').rglob('*'):
        if file.is_file() and file.suffix in ('.json', '.jsonl', '.ipynb', '.py', '.log'):
            allowed.append((file,Path('results/utility')/file.relative_to(ROOT/'outputs/utility_saved')))
    manifest={}
    for source,relative in allowed:
        if not source.is_file():
            continue
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        manifest[relative.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    (dest/'RESULTS.md').write_text('Training/config evidence: results/training/. Generations and blind mappings: '
                               'results/generations/. Judgments, summaries and historical failures: results/judgments/. '
                               'Plots: results/plots/. Report: report/. Full checkpoint weights and exact train/val '
                               'subsets remain in the linked saved Kaggle notebook versions.\n')
    manifest['RESULTS.md']=hashlib.sha256((dest/'RESULTS.md').read_bytes()).hexdigest()
    stale_draft=dest/'report/PA2_report_DRAFT.pdf'
    if 'report/PA2_report.pdf' in manifest and stale_draft.exists():
        assert stale_draft.resolve().is_relative_to(dest.resolve())
        archive_directory=ROOT/'outputs/archived_drafts'
        archive_directory.mkdir(parents=True,exist_ok=True)
        archived=archive_directory/'packaged_PA2_report_DRAFT.pdf'
        if archived.exists():
            archived=archive_directory/('packaged_PA2_report_DRAFT_'+hashlib.sha256(stale_draft.read_bytes()).hexdigest()[:12]+'.pdf')
        stale_draft.replace(archived)
    (dest/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=dest.parent/'PA2_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for relative in sorted([*manifest, 'artifact_manifest.json']):
            z.write(dest/relative,Path(dest.name)/relative)
    print(f'Packaged {len(manifest)} files into {archive}')


if __name__=='__main__':
    main()
