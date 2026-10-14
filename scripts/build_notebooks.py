"""Embed only an explicit source allowlist in Kaggle notebooks; no secrets."""
import base64
import io
from pathlib import Path
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT/'submission_adapter.py', ROOT/'requirements.txt', ROOT/'pyproject.toml']
for directory in ('alignment', 'scripts', 'tests', 'data', 'eval_sets'):
    files.extend(p for p in (ROOT/directory).rglob('*')
                 if p.is_file() and p.suffix in ('.py', '.json', '.md') and '__pycache__' not in p.parts)
stream = io.BytesIO()
with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in files:
        z.write(file, file.relative_to(ROOT).as_posix())
payload = base64.b64encode(stream.getvalue()).decode()
setup = f'''import base64, io, zipfile, os, subprocess
from pathlib import Path
root = Path('/kaggle/working/pa2')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(root)
os.chdir(root)
subprocess.run(['python', '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
subprocess.run(['python', '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
import torch, transformers, datasets
print({{'torch': torch.__version__, 'transformers': transformers.__version__, 'datasets': datasets.__version__}})
subprocess.run(['nvidia-smi'], check=True)
assert torch.cuda.device_count() == 2, 'Select GPU T4 x2'
assert all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'Expected two T4 GPUs'
subprocess.run(['python', '-m', 'pytest', '-q', 'tests'], check=True)
'''
def write(name, cells):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# PA2: '+name), nbf.v4.new_code_cell(setup)] + cells)
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    target = ROOT/'notebooks'
    target.mkdir(exist_ok=True)
    nbf.write(nb, target/f'{name}.ipynb')
write('00_preflight', [nbf.v4.new_code_cell("print('PA2 GPU preflight passed; no training has run.')")])
write('01_sft', [
    nbf.v4.new_code_cell("subprocess.run(['python', '-m', 'scripts.prepare_data', '--tracks', 'sft'], check=True)"),
    nbf.v4.new_code_cell("""import json
for split, count in [('train', 64), ('validation', 16)]:
    rows = json.loads(Path(f'outputs/data/sft_{split}.json').read_text())
    Path(f'outputs/data/pilot_{split}.json').write_text(json.dumps(rows[:count]))
subprocess.run(['python', '-m', 'scripts.train', '--task', 'sft', '--train', 'outputs/data/pilot_train.json', '--validation', 'outputs/data/pilot_validation.json', '--output', 'outputs/sft_pilot', '--max-steps', '2'], check=True)
"""),
    nbf.v4.new_code_cell("subprocess.run(['python', '-m', 'scripts.train', '--task', 'sft', '--train', 'outputs/data/sft_train.json', '--validation', 'outputs/data/sft_validation.json', '--output', 'outputs/sft'], check=True)"),
    nbf.v4.new_code_cell("print('Save this notebook version to preserve /kaggle/working/pa2/outputs.')"),
])
write('01_sft_sanity', [
    nbf.v4.new_code_cell("""import json
candidates = list(Path('/kaggle/input').rglob('sft/best/config.json'))
assert len(candidates) == 1, f'Expected one SFT best checkpoint, found {candidates}'
sft = candidates[0].parent
data_source = sft.parents[1] / 'data'
prompts = data_source/'alpaca_eval_300_seed42.json'
assert prompts.is_file()
for model, name in [('Qwen/Qwen2.5-0.5B', 'qwen2.5-0.5b-base'), (str(sft), 'sft')]:
    target = f'outputs/sanity/{name}.json'
    subprocess.run(['python','-m','scripts.generate','--model',model,'--prompts',str(prompts),
                    '--output',target,'--generator',name,'--limit','3',
                    '--repetition-penalty','1.0','--no-repeat-ngram-size','0'],check=True)
    print(json.dumps(json.loads(Path(target).read_text()),indent=2))
"""),
])
for track in ('hh', 'uf'):
    write(f'02_dpo_{track}', [
        nbf.v4.new_code_cell("""import shutil
candidates = list(Path('/kaggle/input').rglob('sft/best/config.json'))
assert len(candidates) == 1, 'Attach exactly one saved PA2 SFT notebook output'
sft = candidates[0].parent
data_source = sft.parents[1] / 'data'
assert data_source.is_dir()
shutil.copytree(data_source, root/'outputs/data', dirs_exist_ok=True)
"""),
        nbf.v4.new_code_cell(f"subprocess.run(['python', '-m', 'scripts.prepare_data', '--tracks', '{track}'], check=True)"),
        nbf.v4.new_code_cell(f"subprocess.run(['python', '-m', 'scripts.train', '--task', 'dpo', '--model', str(sft), '--train', 'outputs/data/{track}_train.json', '--validation', 'outputs/data/{track}_validation.json', '--output', 'outputs/dpo_{track}'], check=True)"),
    ])
write('03_persona', [
    nbf.v4.new_code_cell("subprocess.run(['python','-m','pip','install','-q','trl==0.24.0'],check=True)"),
    nbf.v4.new_code_cell("subprocess.run(['python','-m','scripts.train_persona'],check=True)"),
])
for track in ('hh', 'uf'):
    write(f'04_generate_{track}', [
        nbf.v4.new_code_cell(f"""import json
def checkpoint(run):
    matches = list(Path('/kaggle/input').rglob(f'{{run}}/best/config.json'))
    assert len(matches)==1, f'Expected one {{run}} checkpoint, found {{matches}}'
    return matches[0].parent
sft = checkpoint('sft')
dpo = checkpoint('dpo_{track}')
prompts = (sft.parents[1]/'data/alpaca_eval_300_seed42.json') if '{track}'=='uf' else root/'eval_sets/hh_rlhf_eval_300_seed42.json'
models = [(str(sft),'sft'),(str(dpo),'dpo_{track}')]
if '{track}'=='uf':
    models.append(('Qwen/Qwen2.5-0.5B','qwen2.5-0.5b-base'))
for model,name in models:
    subprocess.run(['python','-m','scripts.generate','--model',model,'--prompts',str(prompts),
                    '--output',f'outputs/sanity_controlled/{{name}}.json','--generator',name,'--limit','3'],check=True)
    print(Path(f'outputs/sanity_controlled/{{name}}.json').read_text())
"""),
        nbf.v4.new_markdown_cell('Inspect the sanity responses above before running the full generation cell. '
                               'Check checkpoint identities, prompt formatting, EOS/truncation, repetition and factuality.'),
        nbf.v4.new_code_cell('SANITY_REVIEWED = False  # Set True after inspecting the saved responses above.'),
        nbf.v4.new_code_cell(f"""assert SANITY_REVIEWED, 'Inspect the sanity responses before full generation'
processes=[]
for i,(model,name) in enumerate(models):
    if len(processes)==2:
        assert all(p.wait()==0 for p in processes)
        processes=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%2))
    processes.append(subprocess.Popen(['python','-m','scripts.generate','--model',model,'--prompts',str(prompts),
        '--output',f'outputs/eval/{track}/{{name}}.json','--generator',name],env=env))
assert all(p.wait()==0 for p in processes)
subprocess.run(['python','-m','scripts.build_pairs','--baseline','outputs/eval/{track}/sft.json',
    '--candidate','outputs/eval/{track}/dpo_{track}.json','--output','outputs/eval/{track}/pairs.jsonl'],check=True)
"""),
    ])
write('04_generate_persona', [
    nbf.v4.new_code_cell("""matches=list(Path('/kaggle/input').rglob('persona/best/config.json'))
assert len(matches)==1
models=[('HuggingFaceTB/SmolLM2-360M-Instruct','smollm2-instruct'),(str(matches[0].parent),'persona_lora')]
prompts='eval_sets/persona_eval_seed42.json'
for model,name in models:
    subprocess.run(['python','-m','scripts.generate','--model',model,'--prompts',prompts,
        '--output',f'outputs/sanity_persona/{name}.json','--generator',name,'--limit','3','--chat-template'],check=True)
    print(Path(f'outputs/sanity_persona/{name}.json').read_text())
"""),
    nbf.v4.new_markdown_cell('Inspect both baseline and tuned sanity responses before full generation.'),
    nbf.v4.new_code_cell('SANITY_REVIEWED = False  # Set True after inspecting both models.'),
    nbf.v4.new_code_cell("""assert SANITY_REVIEWED, 'Inspect the sanity responses before full generation'
for model,name in models:
    subprocess.run(['python','-m','scripts.generate','--model',model,'--prompts',prompts,
        '--output',f'outputs/eval/persona/{name}.json','--generator',name,'--chat-template'],check=True)
subprocess.run(['python','-m','scripts.build_pairs','--baseline','outputs/eval/persona/smollm2-instruct.json',
    '--candidate','outputs/eval/persona/persona_lora.json','--output','outputs/eval/persona/pairs.jsonl'],check=True)
"""),
])
