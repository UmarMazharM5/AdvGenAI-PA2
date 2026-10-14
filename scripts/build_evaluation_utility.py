"""Prepare credential-free saved outputs for a private Kaggle utility benchmark."""
import hashlib
import json
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'outputs/utility_dataset'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    files = [(ROOT/'alignment/evaluation.py', Path('evaluation.py'))]
    for track in ('hh', 'uf', 'persona'):
        generated = ROOT/f'outputs/generated/{track}/pa2/outputs/eval/{track}'
        judged = ROOT/f'outputs/evaluation/generate-{track}'
        for directory in (generated, judged):
            files.extend((p, Path(track)/p.name) for p in directory.glob('*')
                         if p.suffix in ('.json', '.jsonl'))
    for source, relative in files:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'title': 'PA2 saved outputs and blind judgments',
        'id': 'umarmazhar2025/pa2-saved-evaluation-evidence',
        'licenses': [{'name': 'CC0-1.0'}],
    }, indent=2))
    source = ROOT/'kaggle/pa2_evaluation_utility.py'
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# PA2 saved-output evaluation utility\n'
            'Resume and validate all 620 completed judgments, then independently rejudge the '
            'first five saved pairs per track as an interface audit. Audit results are separate '
            'from the original experiment and do not replace its scores.'),
        nbformat.v4.new_code_cell(source.read_text(encoding='utf-8')),
    ])
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbformat.write(notebook, ROOT/'notebooks/05_evaluation_utility.ipynb')
    print(f'Prepared {len(manifest)} credential-free evidence files for private dataset upload')


if __name__ == '__main__':
    main()
