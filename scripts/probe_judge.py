"""List model identifiers supported by the authenticated Kaggle model proxy."""
from scripts.judge_pairwise_kaggle import proxy_environment
import requests

if __name__ == '__main__':
    base, key = proxy_environment()
    response = requests.get(base+'/models', headers={'Authorization':'Bearer '+key}, timeout=30)
    print('Kaggle model list status:',response.status_code)
    if response.ok:
        print('\n'.join(m['id'] for m in response.json().get('data', [])))
