"""Detached, capped boost launch alongside the identified own D222 process."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_boost import source_fingerprint

config = json.loads((ROOT / 'config.json').read_text())
out = Path(config['output_root'])
out.mkdir(parents=True, exist_ok=True)
pid_file = out / 'trainer.pid'
if pid_file.exists():
    pid = int(pid_file.read_text())
    path = Path('/proc', str(pid), 'cmdline')
    if path.exists() and b'run_boost.py' in path.read_bytes():
        print(json.dumps({'phase': 'already_running', 'pid': pid}))
        sys.exit(0)
uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
for line in apps.splitlines():
    gpu, pid = [v.strip() for v in line.split(',')]
    if gpu != uuid:
        continue
    cmd = Path('/proc', pid, 'cmdline').read_bytes()
    cwd = Path('/proc', pid, 'cwd').resolve()
    if b'legal_d.pipeline' not in cmd or cwd != Path('/home/stud1/cs781_d222_20261011'):
        raise RuntimeError('GPU1 contains an unrelated process; boost launch deferred')
free, total = [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).split(',')]
if free < total*config['gpu_memory_fraction']+512:
    raise RuntimeError('Insufficient GPU memory headroom for capped boost')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
preflight = subprocess.run([sys.executable, '-u', str(ROOT / 'run_boost.py'), '--config', str(ROOT / 'config.json'), '--preflight-only'],
                           env=env, cwd=ROOT, capture_output=True, text=True)
(out / 'preflight.log').write_text(preflight.stdout + preflight.stderr)
if preflight.returncode:
    print(preflight.stdout + preflight.stderr)
    raise RuntimeError('Boost preflight failed; training not started')
checked = json.loads((out / 'preflight.json').read_text())
if checked['source_hashes'] != source_fingerprint() or checked['config'] != config:
    raise ValueError('Preflight source/config changed')
with (out / 'training.log').open('ab', buffering=0) as log:
    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_boost.py'), '--config', str(ROOT / 'config.json')],
                             env=env, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
pid_file.write_text(str(child.pid) + '\n')
time.sleep(3)
if child.poll() is not None:
    raise RuntimeError('Boost process exited during startup; inspect preserved log')
print(json.dumps({'phase': 'launched', 'pid': child.pid, 'physical_gpu': 1, 'concurrent_own_D222': True,
                  'memory_fraction_cap': config['gpu_memory_fraction'], 'preflight_loss': checked['loss'],
                  'preflight_peak_allocated_bytes': checked['peak_allocated_bytes'], 'merge_error': checked['adapter_merge_max_absolute_error']}))
