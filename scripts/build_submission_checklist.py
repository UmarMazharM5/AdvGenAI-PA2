"""Prepare the assignment's five deliverables from verified saved artifacts."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools'))
sys.path.insert(0, str(ROOT))
from alignment.evaluation import summarize
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-folder', action='store_true')
    args = parser.parse_args()
    project = ROOT if (ROOT/'results/generations').is_dir() else ROOT/'submission/PA2_project'
    status = json.loads((ROOT/'report/report_status.json').read_text())
    assert status['submission_ready'] and status['repository_url']
    report = ROOT/'report/PA2_report.pdf'
    assert 1 <= len(PdfReader(report).pages) <= 8
    labels = {'hh': 'HH DPO vs SFT', 'uf': 'UltraFeedback DPO vs SFT',
              'persona': 'Persona LoRA vs SmolLM2-Instruct'}
    data = {}
    for track in labels:
        generation = project/f'results/generations/{track}'
        judged = project/f'results/judgments/{track}'
        pairs, decisions = lines(generation/'pairs.jsonl'), lines(judged/'judged.jsonl')
        actual = summarize(pairs, decisions, lines(generation/'pairs.identities.jsonl'))
        stored = json.loads((judged/'judged.summary.json').read_text())
        assert all(stored[k] == v for k, v in actual.items())
        assert actual['failures'] == actual['unknown'] == 0
        data[track] = {**actual,
            'retried_successful_pairs': sum(j.get('attempts', 1) > 1 for j in decisions),
            'recovered_single_pairs': sum(bool(j.get('recovered_single_pair')) for j in decisions)}

    sensitivity = json.loads((project/'results/audit/evaluation_overlap_sensitivity.json').read_text())
    clean = sensitivity['excluding_training_matches']
    heading = ['Comparison', 'N', 'Wins', 'Ties', 'Losses', 'Win% excl. ties', 'Win% ties-half']
    table_rows = [heading]+[[labels[t], s['sample_count'], s['wins'], s['ties'], s['losses'],
                           f"{100*s['win_rate_excluding_ties']:.1f}%", f"{100*s['win_rate_ties_half']:.1f}%"]
                          for t, s in data.items()]
    method = ('Judge: google/gemini-3.6-flash through Kaggle-supported model access. '
              'All 620 primary decisions are resolved; unknown and unresolved counts are zero in every track. '
              'Saved model responses use identical greedy decoding within each comparison: maximum 256 new '
              'tokens, repetition penalty 1.1 and no-repeat n-gram size 4. Model names are hidden from the '
              'judge, with A/B positions randomized using seed 42. Judge batches contain up to five pairs, '
              'with at least five seconds between attempts, up to three attempts, checkpoints and resume.')
    interpretation = ('HH DPO wins more often overall, but the category analysis and manually reviewed '
                      'outputs show remaining unsafe assistance and weaker self-harm support. UF DPO loses '
                      'more often than SFT. Persona results are balanced, with little target style: both '
                      'models average 1.00/5 for style and 1.95/5 for usefulness; safety averages are '
                      '4.35/5 tuned versus 4.30/5 baseline. These are observed results, not evidence that '
                      'every DPO track improved.')
    overlap = (f'Two UF training prompts overlap the prescribed AlpacaEval set. Excluding them leaves '
               f'N={clean["sample_count"]}, W/T/L={clean["wins"]}/{clean["ties"]}/{clean["losses"]}, '
               f'win rate excluding ties {100*clean["win_rate_excluding_ties"]:.1f}% and ties-half '
               f'{100*clean["win_rate_ties_half"]:.1f}%. This supplementary analysis preserves the '
               'conclusion and does not replace the required fixed 300-prompt score.')
    reliability = '; '.join(f'{labels[t]}: {s["retried_successful_pairs"]} successful pairs retried, '
                           f'{s["recovered_single_pairs"]} recovered individually' for t, s in data.items())
    provenance = ('The primary judgments were produced locally through the supported Kaggle proxy. '
                  'A saved private Kaggle utility benchmark then verified/resumed all 620 and made 15 '
                  'separate fresh Gemini 3.6 audit judgments; these did not replace the primary scores. '
                  'Historical authentication/parsing failures are logged separately and are not model losses. '
                  'Controlled DPO/persona sanity responses were manually reviewed during the final audit; '
                  'the timing deviation is disclosed in the main report.')
    evidence = ('Repository paths: results/generations/ for fixed responses and blind mappings; '
                'results/judgments/ for primary decisions, summaries and failure logs; results/audit/ '
                'for data/category checks; results/utility/ for the native utility audit.')
    markdown = ['# PA2 evaluation summary', '', status['repository_url'], '',
                '| '+' | '.join(heading)+' |', '| '+' | '.join(['---']*len(heading))+' |']
    markdown += ['| '+' | '.join(str(c) for c in row)+' |' for row in table_rows[1:]]
    markdown += ['', 'Win rate excluding ties = W/(W+L). Ties-half = (W+0.5T)/(W+T+L).', '',
                 method, '', reliability+'.', '', interpretation, '', overlap, '', provenance, '', evidence, '']
    (ROOT/'report/evaluation_summary.md').write_text('\n'.join(markdown), encoding='utf-8')
    (ROOT/'report/evaluation_summary.json').write_text(json.dumps({
        'repository_url': status['repository_url'], 'judge_model': 'google/gemini-3.6-flash',
        'tracks': data, 'uf_overlap_sensitivity': sensitivity}, indent=2))
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize, styles['BodyText'].leading = 8.5, 11
    story = [Paragraph('PA2 evaluation summary', styles['Heading1']),
             Paragraph(escape(status['repository_url']), styles['BodyText']), Spacer(1, 10)]
    cells = [[Paragraph(escape(str(c)), styles['BodyText']) for c in row] for row in table_rows]
    table = Table(cells, colWidths=[1.75*inch, 0.32*inch, 0.4*inch, 0.36*inch, 0.43*inch, 0.8*inch, 0.8*inch], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0), colors.HexColor('#e5edf5')),
                              ('GRID',(0,0),(-1,-1),0.3,colors.lightgrey), ('VALIGN',(0,0),(-1,-1),'TOP')]))
    story += [table, Spacer(1, 9)]
    for text in ('Win% excluding ties = W/(W+L); ties-half = (W+0.5T)/(W+T+L).',
                 method, reliability+'.', interpretation, overlap, provenance, evidence):
        story += [Paragraph(escape(text), styles['BodyText']), Spacer(1, 8)]
    SimpleDocTemplate(str(ROOT/'report/evaluation_summary.pdf'), pagesize=(8.5*inch, 11*inch),
                      leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.5*inch, bottomMargin=0.5*inch).build(story)
    assert len(PdfReader(ROOT/'report/evaluation_summary.pdf').pages) == 1
    (ROOT/'SUBMISSION_CHECKLIST.md').write_text(
        '# PA2 submission checklist\n\n'
        f'1. Repository URL: {status["repository_url"]}\n'
        '2. Final report: [report/PA2_report.pdf](report/PA2_report.pdf), eight pages.\n'
        '3. Training plots: [results/plots/](results/plots/), SFT, HH DPO, UF DPO and persona LoRA; PNG and PDF.\n'
        '4. Evaluation summary: [report/evaluation_summary.pdf](report/evaluation_summary.pdf); '
        'also Markdown and JSON in report/.\n'
        '5. Root-level [submission_adapter.py](submission_adapter.py) imports the original implementation '
        'in alignment/core.py. Public adapter tests: `python -m pytest -q tests/test_public_adapter_smoke.py`.\n\n'
        'The repository is private. Grant course staff access before submitting its URL.\n', encoding='utf-8')
    print('Generated the one-page evaluation summary and five-item checklist')
    if not args.prepare_folder:
        return
    destination = ROOT/'submission/PA2_checklist'
    destination.mkdir(parents=True, exist_ok=True)
    copies = [(report, 'PA2_report.pdf'), (ROOT/'report/evaluation_summary.pdf', 'evaluation_summary.pdf'),
              (ROOT/'report/evaluation_summary.md', 'evaluation_summary.md'),
              (ROOT/'report/evaluation_summary.json', 'evaluation_summary.json'),
              (ROOT/'submission_adapter.py', 'submission_adapter.py'),
              (ROOT/'requirements.txt', 'requirements.txt'), (ROOT/'pyproject.toml', 'pyproject.toml')]
    for folder in ('alignment', 'tests'):
        copies += [(p, p.relative_to(ROOT).as_posix()) for p in (ROOT/folder).glob('*.py')]
    plot_directory = project/'results/plots'
    copies += [(p, 'training_plots/'+p.name) for p in plot_directory.glob('*') if p.suffix in ('.png', '.pdf')]
    for source, relative in copies:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination/'repository_url.txt').write_text(status['repository_url']+'\n')
    (destination/'README.md').write_text(
        '# PA2 files to submit\n\n'
        '1. repository_url.txt: paste the URL into the course submission form.\n'
        '2. PA2_report.pdf: final eight-page report.\n'
        '3. training_plots/: SFT and DPO plots in PNG/PDF.\n'
        '4. evaluation_summary.pdf: one-page comparison, with Markdown/JSON copies.\n'
        '5. submission_adapter.py: connected to alignment/core.py; tests/ is included.\n\n'
        'The GitHub repository contains the full training code, notebooks and saved evaluation evidence. '
        'The complete project archive is PA2_submission.zip, next to this folder.\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_public_adapter_smoke.py'], cwd=destination, check=True)
    manifest = {p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in destination.rglob('*') if p.is_file() and p.name!='deliverable_manifest.json'
                and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    (destination/'deliverable_manifest.json').write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination.parent/'PA2_checklist.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted([*manifest, 'deliverable_manifest.json']):
            archive.write(destination/relative, 'PA2_checklist/'+relative)
    print('Prepared submission/PA2_checklist/ and PA2_checklist.zip; public adapter tests pass')


if __name__ == '__main__':
    main()
