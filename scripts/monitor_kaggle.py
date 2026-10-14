"""Watch Kaggle status and live metrics; persist evidence without credentials.

Runs independently of the chat. Does not alter or restart the remote notebook.
"""
import argparse
import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.kaggle_cli  # Loads the local .env without displaying credentials.
import kaggle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', default='umarmazhar2025/pa2-sft-training')
    p.add_argument('--output', default='outputs/monitor/sft')
    p.add_argument('--hours', type=float, default=4)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--run-name', default='sft')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    health = {'kernel': a.kernel, 'stage': 'unknown', 'alerts': [], 'latest_metric': None}
    seen = set()

    def now():
        return datetime.now(timezone.utc).isoformat()

    def persist():
        with lock:
            snapshot = dict(health)
            snapshot['checked_at_utc'] = now()
            tmp = out/'status.tmp'
            tmp.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            tmp.replace(out/'status.json')

    def alert(message):
        with lock:
            if message not in health['alerts']:
                health['alerts'].append(message)
                with (out/'alerts.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'time':now(), 'message':message})+'\n')

    def stream():
        while not stop.is_set():
            try:
                for event in kaggle.api.kernels_logs_stream(a.kernel):
                    if stop.is_set():
                        return
                    data = event.get('data', '')
                    for line in data.splitlines():
                        start = line.find('{"phase"')
                        if start >= 0:
                            try:
                                metric = json.loads(line[start:])
                            except ValueError:
                                continue
                            key = json.dumps(metric, sort_keys=True)
                            if key in seen:
                                continue
                            seen.add(key)
                            with lock:
                                previous = health['latest_metric']
                                if metric['phase'] == 'validation' and metric['step'] == 0:
                                    health['stage'] = a.run_name if a.run_name != 'sft' else ('full_sft' if previous else 'pilot')
                                health['latest_metric'] = metric
                                if metric['phase'] == 'train':
                                    health['latest_train_metric'] = metric
                                else:
                                    health['latest_validation_metric'] = metric
                            with (out/'metrics.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps({'received_at':now(), **metric})+'\n')
                            if not math.isfinite(metric.get('loss', 0)):
                                alert('Non-finite loss observed')
                            if metric.get('skipped_update'):
                                alert(f"Skipped optimizer update at step {metric['step']}")
                            persist()
                        if any(s in line.lower() for s in ('out of memory', 'floatingpointerror', 'traceback (most recent call last)')):
                            alert(line[:500])
                    with lock:
                        health['last_log_event_at_utc'] = now()
                return
            except Exception as exc:
                with lock:
                    health['stream_error_type'] = type(exc).__name__
                stop.wait(15)

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    deadline = time.monotonic()+a.hours*3600
    while time.monotonic() < deadline:
        try:
            response = kaggle.api.kernels_status(a.kernel)
            status = str(response.status)
            with lock:
                health['remote_status'] = status
                health['failure_message'] = response.failure_message or ''
            persist()
            if any(s in status.upper() for s in ('COMPLETE', 'ERROR', 'CANCEL')):
                if 'COMPLETE' in status.upper():
                    # Preserve small verification artifacts, avoiding checkpoint downloads here.
                    kaggle.api.kernels_output(a.kernel, path=str(out/'artifacts'),
                        file_pattern=rf'outputs/{a.run_name}/(metrics\.(jsonl|json)|config\.json|completed\.json)$', quiet=True)
                else:
                    alert(f'Remote run ended with {status}')
                stop.set()
                persist()
                return
        except Exception as exc:
            with lock:
                health['poll_error_type'] = type(exc).__name__
            persist()
        stop.wait(a.interval)
    with lock:
        health['monitor_ended'] = 'time limit reached; remote training was not stopped'
    stop.set()
    persist()


if __name__ == '__main__':
    main()
