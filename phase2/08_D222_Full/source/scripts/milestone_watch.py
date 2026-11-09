"""CPU sidecar: prune only evicted D candidates and freeze 100M/200M models.

Never edits trainer source/config, signals a process, or touches shared runs.
"""
import argparse
import fcntl
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legal_d
import torch
from legal_d.common import save, source_hashes
from lner.data import atomic_json, sha256

EXPECTED = Path('/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_d222_20261011')
TARGETS = (100_000_000, 200_000_000)


def read(path):
    return json.loads(path.read_text())


def same_process(pid, expected):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split()[21] == expected
    except FileNotFoundError:
        return False


def prune(root):
    folder = root / 'pretrain'
    inventory = folder / 'candidate_inventory.json'
    if not inventory.exists():
        return
    raw = inventory.read_bytes()
    keep = json.loads(raw)
    if not keep:
        return
    protected = {Path(item['path']).resolve() for item in keep}
    minimum = min(item['step'] for item in keep)
    for path in sorted(folder.glob('candidate_*.pt')):
        match = re.fullmatch(r'candidate_(\d+)\.pt', path.name)
        if not match or int(match[1]) >= minimum or path.resolve() in protected:
            continue
        if path.is_symlink() or path.resolve().parent != folder.resolve():
            raise ValueError('Unsafe candidate path')
        # A newly saved candidate is never eligible; recheck the atomic inventory.
        if inventory.read_bytes() != raw:
            return
        size = path.stat().st_size
        entry = {'path': str(path), 'bytes': size, 'reason': 'evicted_below_oldest_retained_candidate', 'time': time.time()}
        with (root / 'milestones/cleanup.jsonl').open('a') as handle:
            handle.write(json.dumps({'action': 'intent', **entry}) + '\n')
        path.unlink()
        with (root / 'milestones/cleanup.jsonl').open('a') as handle:
            handle.write(json.dumps({'action': 'deleted', **entry}) + '\n')
        print(json.dumps({'deleted': str(path), 'bytes_freed': size}), flush=True)


def capture(root):
    folder = root / 'pretrain'
    status = folder / 'status.json'
    if not status.exists():
        return
    live = read(status)
    count = live.get('target_tokens', live.get('trainer_target_tokens', 0))
    pending = [t for t in TARGETS if count >= t and not (root / 'milestones' / f'{t // 1_000_000}M.json').exists()]
    if not pending:
        return
    # Atomic trainer rename makes an opened inode immutable even on a later save.
    with (folder / 'latest.pt').open('rb') as handle:
        payload = torch.load(handle, map_location='cpu', weights_only=False)
    original = read(folder / 'run.json')
    if payload['fingerprint'] != original['fingerprint'] or source_hashes() != payload['fingerprint']['source_hashes']:
        raise ValueError('Milestone lineage differs from live trainer')
    achieved = payload['state']['target_tokens']
    for target in pending:
        if achieved < target:
            continue
        path = root / 'milestones' / f'pretrained_{target // 1_000_000}M.pt'
        # NER requires model/state/evidence, not three copies of optimizer buffers.
        save(path, {k: payload[k] for k in ('model', 'state', 'evidence', 'fingerprint')})
        record = {'phase': 'captured', 'requested_target_tokens': target, 'actual_target_tokens': achieved,
                  'step': payload['state']['step'], 'checkpoint': str(path), 'sha256': sha256(path),
                  'selection': 'first_complete_latest_seen_at_or_above_target',
                  'exact_training_resume': False, 'original_fingerprint': payload['fingerprint'], 'captured_at': time.time()}
        atomic_json(path.with_name(f'{target // 1_000_000}M.json'), record)
        print(json.dumps({k: v for k, v in record.items() if k != 'original_fingerprint'}), flush=True)
    del payload
    gc.collect()


def main(config_path):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    root = Path(config['output_root']).resolve()
    if root != EXPECTED.resolve() or root.is_symlink():
        raise ValueError('Retention is restricted to the named D222 artifact root')
    directory = root / 'milestones'
    directory.mkdir(exist_ok=True)
    lock = (directory / 'watch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    pid = int((root / 'queue.pid').read_text())
    identity = Path('/proc', str(pid), 'stat').read_text().split()[21]
    if b'legal_d.pipeline' not in Path('/proc', str(pid), 'cmdline').read_bytes():
        raise ValueError('Unexpected active trainer identity')
    atomic_json(directory / 'watch_status.json', {'phase': 'watching', 'pid': os.getpid(), 'trainer_pid': pid})
    while True:
        try:
            prune(root)
            capture(root)
            complete = root / 'pretrain/complete.json'
            if complete.exists():
                achieved = read(complete)['trainer_target_tokens']
                for target in TARGETS:
                    record = directory / f'{target // 1_000_000}M.json'
                    if not record.exists() and achieved < target:
                        atomic_json(record, {'phase': 'not_reached', 'requested_target_tokens': target, 'actual_target_tokens': achieved})
            if not same_process(pid, identity):
                available = [t for t in TARGETS if (directory / f'{t // 1_000_000}M.json').exists() and read(directory / f'{t // 1_000_000}M.json')['phase'] == 'captured']
                if not available:
                    atomic_json(directory / 'watch_status.json', {'phase': 'no_milestones_available'})
                    return
                delivered = list(root.glob('ner/*/complete.json'))
                if len(delivered) < 2 or any(sha256(read(p)['submission']) != read(p)['sha256'] for p in delivered):
                    atomic_json(directory / 'watch_status.json', {'phase': 'blocked', 'reason': 'Original NER exports not verified; preserve queue for diagnosis'})
                    return
                gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
                if gpu not in apps:
                    atomic_json(directory / 'watch_status.json', {'phase': 'milestone_ner_running', 'gpu_uuid': gpu})
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4')
                    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('milestone_ner.py')), '--config', str(config_path)], env=env, check=True)
                    atomic_json(directory / 'watch_status.json', {'phase': 'complete'})
                    return
                atomic_json(directory / 'watch_status.json', {'phase': 'waiting_for_free_gpu1', 'gpu_uuid': gpu})
        except Exception as error:
            atomic_json(directory / 'watch_failure.json', {'error': str(error), 'time': time.time()})
            print(f'{type(error).__name__}: {error}', flush=True)
            # Never proceed to GPU training after a retention/capture/NER failure.
            raise
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)
