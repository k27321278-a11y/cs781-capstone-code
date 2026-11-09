import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from phase4_cjpe.common import atomic_json, now


def worker(path):
    from .train import train_segment
    config = json.loads(Path(path).read_text())
    root = Path(config['run_root'])
    try:
        result = train_segment(config)
        atomic_json(root / 'job_result.json', result)
    except Exception as error:
        record = dict(error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), failed_utc=now())
        atomic_json(root / 'failure.json', record)
        print('WORKER_FAILED ' + json.dumps(dict(name=config['name'], **record)), flush=True)
        raise


def launch(configs, output_root):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < len(configs):
        raise RuntimeError('Requested isolated GPUs are not available')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, config in enumerate(configs):
        folder = Path(config['run_root'])
        folder.mkdir(parents=True, exist_ok=True)
        prior = folder / 'ownership.json'
        if prior.exists():
            owner = json.loads(prior.read_text())
            command = Path('/proc') / str(owner['pid']) / 'cmdline'
            if command.exists() and str(folder / 'job_config.json').encode() in command.read_bytes():
                raise RuntimeError('This exact job already has a live owner')
        path = folder / 'job_config.json'
        atomic_json(path, config)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(index), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env['PYTHONPATH'] = str(Path(__file__).parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        log = (folder / 'worker.log').open('a', encoding='utf-8')
        process = subprocess.Popen([sys.executable, '-u', '-m', 'b1_study.run', '--job', str(path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        atomic_json(prior, dict(pid=process.pid, physical_gpu=index, started_utc=now(), config=str(path)))
        print('WORKER_STARTED ' + json.dumps(dict(name=config['name'], physical_gpu=index, pid=process.pid)), flush=True)
        jobs.append(dict(process=process, log=log, config=config, offset=0))
    pending = jobs[:]
    while pending:
        for item in pending[:]:
            folder = Path(item['config']['run_root'])
            with (folder / 'worker.log').open(encoding='utf-8', errors='replace') as stream:
                stream.seek(item['offset'])
                lines = stream.readlines()
                item['offset'] = stream.tell()
            for line in lines:
                # Forward actual progress into published/live parent logs.
                print(item['config']['name'] + ' ' + line.rstrip(), flush=True)
            code = item['process'].poll()
            if code is not None:
                item['log'].close()
                terminal = folder / ('job_result.json' if code == 0 else 'failure.json')
                result = json.loads(terminal.read_text()) if terminal.exists() else dict(error='Missing terminal record', exit_code=code)
                atomic_json(root / ('TERMINAL_' + item['config']['name'] + '.json'), result)
                print('JOB_TERMINAL ' + json.dumps(dict(name=item['config']['name'], exit_code=code,
                      state=result.get('state', 'failed'), step=result.get('optimizer_updates'))), flush=True)
                pending.remove(item)
        if pending:
            time.sleep(10)
    summary = []
    for item in jobs:
        folder = Path(item['config']['run_root'])
        f = folder / 'job_result.json'
        summary.append(json.loads(f.read_text()) if f.exists() else dict(name=item['config']['name'], state='failed'))
    atomic_json(root / 'SEGMENT_SUMMARY.json', dict(jobs=summary, finished_utc=now()))
    if any(r.get('state') not in ('complete', 'paused') for r in summary):
        raise RuntimeError('One or more owned jobs failed; preserve published diagnostics before repair')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    worker(parser.parse_args().job)
