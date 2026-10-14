"""Load local credentials without printing them, then invoke Kaggle CLI."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(ROOT / '.tools'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from kaggle.cli import main

if __name__ == '__main__':
    main()
