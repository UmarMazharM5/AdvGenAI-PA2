"""Read a short snapshot of a running notebook's live log."""
import argparse
import threading
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.kaggle_cli
import kaggle

p = argparse.ArgumentParser()
p.add_argument('kernel')
p.add_argument('--seconds', type=float, default=8)
p.add_argument('--tail', type=int, default=12000)
a = p.parse_args()
parts = []
def collect():
    try:
        for event in kaggle.api.kernels_logs_stream(a.kernel):
            parts.append(event.get('data', ''))
    except Exception as exc:
        parts.append(type(exc).__name__)
threading.Thread(target=collect, daemon=True).start()
time.sleep(a.seconds)
print(''.join(parts)[-a.tail:])
