"""Download only the official Phase 2 files using existing HF authorization."""
import hashlib
import json
import sys
import subprocess
import urllib.request
from pathlib import Path
from huggingface_hub import get_token

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
token = get_token()
if not token:
    command = "/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python -c 'from huggingface_hub import get_token; print(get_token() or chr(32))'"
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'stud1', command],
                            capture_output=True, text=True, timeout=30)
    token = result.stdout.strip() if result.returncode == 0 else None
    if not token or not token.startswith('hf_'):
        raise RuntimeError('No authorized Hugging Face credential is available')
repo = 'Exploration-Lab/CS-781-Capstone'
req = urllib.request.Request('https://huggingface.co/api/datasets/'+repo,
                             headers={'Authorization': 'Bearer '+token})
info = json.load(urllib.request.urlopen(req, timeout=35))
revision = info['sha']
files = ['Train/labels.txt', 'Train/fold1-text.json', 'Train/fold1-gold.json',
         'Train/fold2-text.json', 'Train/fold2-gold.json', 'Test/fold3-text.json']
manifest = {'repository': repo, 'revision': revision, 'files': {}}
for name in files:
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/Phase_2_LNER/{name}',
                                 headers={'Authorization': 'Bearer '+token})
    raw = urllib.request.urlopen(req, timeout=60).read()
    path.write_bytes(raw)
    manifest['files'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if name.endswith('.json'):
        value = json.loads(raw)
        first = next(iter(value))
        print(json.dumps({'file': name, 'records': len(value), 'first_id': first,
                          'sample': str(value[first])[:350]}), flush=True)
    else:
        print(raw.decode(), flush=True)
(root/'download_manifest.json').write_text(json.dumps(manifest, indent=2))
