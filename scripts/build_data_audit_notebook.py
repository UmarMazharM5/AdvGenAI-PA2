import base64
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
script = (ROOT/'scripts/audit_saved_data.py').read_text()
prompts = base64.b64encode((ROOT/'eval_sets/hh_rlhf_eval_300_seed42.json').read_bytes()).decode()
setup = f'''import base64
from pathlib import Path
Path('eval_sets').mkdir(exist_ok=True)
Path('eval_sets/hh_rlhf_eval_300_seed42.json').write_bytes(base64.b64decode({prompts!r}))
'''
nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# PA2 audit of exact saved data subsets'),
                                   nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(script)])
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
directory = ROOT/'kaggle/data-audit'
directory.mkdir(exist_ok=True)
nbformat.write(nb, directory/'data_audit.ipynb')
(directory/'kernel-metadata.json').write_text(json.dumps({
    'id': 'umarmazhar2025/pa2-saved-data-audit', 'title': 'PA2 saved data audit',
    'code_file': 'data_audit.ipynb', 'language': 'python', 'kernel_type': 'notebook',
    'is_private': True, 'enable_gpu': False, 'enable_internet': False,
    'dataset_sources': [], 'competition_sources': [],
    'kernel_sources': ['umarmazhar2025/pa2-sft-training', 'umarmazhar2025/pa2-dpo-hh', 'umarmazhar2025/pa2-dpo-uf'],
}, indent=2))
print('Prepared CPU audit notebook; no training or model generation')
