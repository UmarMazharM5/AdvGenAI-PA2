"""Provision Kaggle-supported model-proxy credentials without displaying them."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.kaggle_cli
import kaggle
from dotenv import set_key

def refresh_credentials():
    values = kaggle.api._fetch_model_proxy_env(source='auth')
    target = Path(__file__).resolve().parents[1]/'.env.benchmarks'
    for key, value in values.items():
        set_key(str(target), key, value)
    return values

if __name__ == '__main__':
    refresh_credentials()
    print('Kaggle model-proxy credentials saved locally; token not displayed.')
