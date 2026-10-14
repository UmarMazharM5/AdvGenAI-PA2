"""Run the prepared PA2 stages as Kaggle GPU slots become available.

Only advances after dependencies complete successfully; never retries failed
training automatically. Judging uses saved outputs and one local judge process
at a time. Review outputs/pipeline/status.json for failures requiring inspection.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.kaggle_cli
import kaggle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=8)
    a = parser.parse_args()
    out = ROOT/'outputs/pipeline'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        {'name':'hh','kernel':'pa2-dpo-hh','submitted':True,'deps':[]},
        {'name':'uf','kernel':'pa2-dpo-uf','submitted':True,'deps':[]},
        {'name':'persona','kernel':'pa2-persona-lora','folder':'persona','deps':[]},
        {'name':'generate-uf','kernel':'pa2-generate-uf','folder':'generate-uf','deps':['uf']},
        {'name':'generate-hh','kernel':'pa2-generate-hh','folder':'generate-hh','deps':['hh']},
        {'name':'generate-persona','kernel':'pa2-generate-persona','folder':'generate-persona','deps':['persona']},
    ]
    state_file = out/'status.json'
    if state_file.exists():
        old = {j['name']:j for j in json.loads(state_file.read_text())['jobs']}
        for job in jobs:
            job.update(old.get(job['name'], {}))
    active_judge = None
    deadline = time.monotonic()+a.hours*3600

    def persist():
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(), 'jobs':jobs},indent=2))
        tmp.replace(state_file)

    def monitor(job):
        handle = (out/f"{job['name']}_monitor.log").open('a',encoding='utf-8')
        process = subprocess.Popen([sys.executable,'-X','utf8','scripts/monitor_kaggle.py',
            '--kernel','umarmazhar2025/'+job['kernel'],'--output',f"outputs/monitor/{job['name']}",
            '--run-name',job['name'],'--hours','8'],cwd=ROOT,stdout=handle,stderr=handle)
        handle.close()
        job['monitor_pid'] = process.pid

    while time.monotonic()<deadline:
        for job in jobs:
            if not job.get('submitted') or job.get('status') in ('COMPLETE','ERROR','UPSTREAM_ERROR'):
                continue
            try:
                response = kaggle.api.kernels_status('umarmazhar2025/'+job['kernel'])
                job['status'] = str(response.status).split('.')[-1]
                if response.failure_message:
                    job['failure_message'] = response.failure_message
            except Exception as exc:
                job['poll_error_type'] = type(exc).__name__
        by_name = {j['name']:j for j in jobs}
        active = sum(1 for j in jobs if j.get('submitted') and
                     j.get('status') not in ('COMPLETE','ERROR','CANCELLED'))
        for job in jobs:
            if job.get('submitted') or job.get('status')=='UPSTREAM_ERROR':
                continue
            if any(by_name[d].get('status') in ('ERROR','UPSTREAM_ERROR') for d in job['deps']):
                job['status']='UPSTREAM_ERROR'
                continue
            if active>=2 or not all(by_name[d].get('status')=='COMPLETE' for d in job['deps']):
                continue
            try:
                result = kaggle.api.kernels_push(str(ROOT/'kaggle'/job['folder']), acc='NvidiaTeslaT4')
                if result is None or result.error:
                    job['submission_error'] = result.error if result is not None else 'No result'
                    # Account session state can lag. Try again on the next minute.
                    continue
                job.update(submitted=True,status='SUBMITTED',version=result.versionNumber,url=result.url)
                active+=1
                monitor(job)
                persist()
            except Exception as exc:
                job['submission_error_type'] = type(exc).__name__
        if active_judge is not None:
            process, job = active_judge
            code = process.poll()
            if code is not None:
                job['judge_exit_code']=code
                summary = ROOT/f"outputs/evaluation/{job['name']}/judged.summary.json"
                if summary.exists():
                    job['judge_summary']=json.loads(summary.read_text())
                active_judge=None
        for job in jobs:
            if not job['name'].startswith('generate-') or job.get('status')!='COMPLETE':
                continue
            if 'judge_exit_code' in job or job.get('judge_pid') or active_judge is not None:
                continue
            track = job['name'].removeprefix('generate-')
            target = ROOT/f"outputs/generated/{track}"
            try:
                kaggle.api.kernels_output('umarmazhar2025/'+job['kernel'],path=str(target),
                    file_pattern=r'outputs/(eval|sanity_controlled|sanity_persona)/.*\.(json|jsonl)$',quiet=True)
                pairs = list(target.rglob('pairs.jsonl'))
                if len(pairs)!=1:
                    raise ValueError('Expected one saved comparison file')
                result_path = ROOT/f"outputs/evaluation/{job['name']}/judged.jsonl"
                result_path.parent.mkdir(parents=True,exist_ok=True)
                handle = (result_path.parent/'judge.log').open('a',encoding='utf-8')
                process = subprocess.Popen([sys.executable,'-X','utf8','scripts/judge_pairwise_kaggle.py',
                    '--pairs-path',str(pairs[0]),'--output-path',str(result_path),
                    '--rubric','safety' if track=='hh' else ('persona' if track=='persona' else 'general')],
                    cwd=ROOT,stdout=handle,stderr=handle)
                handle.close()
                job['judge_pid']=process.pid
                active_judge=(process,job)
            except Exception as exc:
                job['artifact_error_type']=type(exc).__name__
        persist()
        if all(j.get('status') in ('COMPLETE','ERROR','UPSTREAM_ERROR') for j in jobs) and active_judge is None:
            completed_generation = [j for j in jobs if j['name'].startswith('generate-') and j.get('status')=='COMPLETE']
            if all('judge_exit_code' in j for j in completed_generation):
                return
        time.sleep(60)
    persist()


if __name__=='__main__':
    main()
