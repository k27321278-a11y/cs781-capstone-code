"""Launch the CPU watcher without changing the identified live D trainer."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legal_d
from legal_d.common import source_hashes

code = Path(__file__).resolve().parents[1]
config = code / 'configs/d222.json'
root = Path(json.loads(config.read_text())['output_root'])
run = json.loads((root / 'pretrain/run.json').read_text())
if source_hashes() != run['fingerprint']['source_hashes']:
    raise ValueError('Live training code changed')
directory = root / 'milestones'
directory.mkdir(exist_ok=True)
pid_file = directory / 'watch.pid'
if pid_file.exists():
    old = int(pid_file.read_text())
    process = Path('/proc', str(old), 'cmdline')
    if process.exists() and b'milestone_watch.py' in process.read_bytes():
        print(json.dumps({'phase': 'already_running', 'pid': old}))
        sys.exit(0)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
with (directory / 'watch.log').open('ab', buffering=0) as log:
    child = subprocess.Popen([sys.executable, '-u', str(code / 'scripts/milestone_watch.py'), '--config', str(config)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=code, env=env, start_new_session=True)
pid_file.write_text(str(child.pid) + '\n')
print(json.dumps({'phase': 'watcher_launched', 'pid': child.pid, 'source_fingerprint_unchanged': True}))
