"""Finalize the report and submission bundle after all judge outputs are resolved."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hours',type=float,default=3)
    p.add_argument('--repo-url')
    a=p.parse_args()
    end=time.monotonic()+a.hours*3600
    target=ROOT/'outputs/finalization.json'
    while time.monotonic()<end:
        ready=True
        for track,expected in [('hh',300),('uf',300),('persona',20)]:
            path=ROOT/f'outputs/evaluation/generate-{track}/judged.summary.json'
            if not path.exists():
                ready=False
                continue
            data=json.loads(path.read_text())
            if data['sample_count']!=expected or data['failures']:
                ready=False
        if ready:
            report=[sys.executable,'-X','utf8','scripts/build_report.py']
            if a.repo_url:
                report+=['--repo-url',a.repo_url]
            result=subprocess.run(report,cwd=ROOT)
            if result.returncode:
                target.write_text(json.dumps({'status':'report_failed','exit_code':result.returncode}))
                return
            result=subprocess.run([sys.executable,'-X','utf8','scripts/package_submission.py'],cwd=ROOT)
            target.write_text(json.dumps({'status':'packaged' if result.returncode==0 else 'packaging_failed',
                                         'repository_url':a.repo_url,'submission_ready':bool(a.repo_url)}))
            return
        time.sleep(60)
    target.write_text(json.dumps({'status':'waiting_for_resolved_judgments'}))


if __name__=='__main__':
    main()
