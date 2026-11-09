"""Wait for the exact existing B trainer; then run evaluation/export once."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

from cs781.io import write_json_atomic
from cs781.checkpoints import run_lock


def trainer_alive(pid,run,proc_root=Path('/proc')):
    cmd=Path(proc_root)/str(pid)/'cmdline'
    text=cmd.read_bytes().replace(b'\0',b' ').decode(errors='replace') if cmd.exists() else ''
    # The Python -c launcher joins root/run at runtime. Match both components
    # when the fully joined path is not present in the OS command line.
    return 'train_b_improved' in text and (str(run) in text or
        (str(run.parent) in text and run.name in text))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-pid',type=int,required=True)
    parser.add_argument('--pipeline-script',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    run=args.run_dir.resolve();phase=run/'pipeline_status.json'
    with run_lock(run/'completion-controller'):
        try:
            while trainer_alive(args.training_pid,run):
                write_json_atomic(phase,{'phase':'waiting_for_training','training_pid':args.training_pid})
                time.sleep(60)
            status=json.loads((run/'training_status.json').read_text())
            if status.get('status')!='complete':raise RuntimeError('B trainer ended without completion; no automatic restart')
            fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            write_json_atomic(run/'completion_controller.json',{'code_sha256':fingerprint,'training_pid':args.training_pid})
            sys.argv=[str(args.pipeline_script),'--run-dir',str(run),*remaining]
            runpy.run_path(str(args.pipeline_script),run_name='__main__')
        except Exception as exc:
            write_json_atomic(phase,{'phase':'failed','error':type(exc).__name__+': '+str(exc)})
            print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
