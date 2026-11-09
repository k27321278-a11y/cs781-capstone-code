"""Notebook-owned worker supervision; no UI keep-alive or account automation."""
import argparse,copy,json,os,subprocess,sys,time,tempfile
from pathlib import Path
from .common import atomic_json,digest,now

def read_json(path):
 try:return json.loads(Path(path).read_text(encoding='utf-8'))
 except (FileNotFoundError,json.JSONDecodeError,OSError):return {}

def supervise(config):
 # Linux Colab/Kaggle: a lock prevents a second supervisor in this same runtime.
 import fcntl
 lock_path=Path(tempfile.gettempdir())/('phase4-'+digest(str(Path(config['run_root']).resolve()))+'.lock')
 with lock_path.open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('A supervisor already owns this exact run in this runtime')
  return _supervise(config)

def _stop_owned_child(proc):
 if proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()

def _supervise(config):
 base=Path(config['run_root']);base.mkdir(parents=True,exist_ok=True)
 state_path=base/'supervisor.json';identity=digest(config);state=read_json(state_path)
 if state and state.get('original_config_sha256')!=identity:raise ValueError('Supervisor config changed: use a new run name')
 old_pid=state.get('worker_pid')
 if old_pid:
  # After an interrupted/restarted notebook kernel, never create a second writer.
  command_file=Path('/proc')/str(old_pid)/'cmdline'
  try:old_command=command_file.read_bytes().split(b'\0')
  except FileNotFoundError:old_command=[]
  expected=str(Path(state.get('active_config',config)['run_root'])/'worker_config.json').encode()
  if b'phase4_cjpe.runtime' in old_command and expected in old_command:
   raise RuntimeError('The previous owned training worker is still running (PID '+str(old_pid)+'). Inspect its status; do not launch a duplicate.')
 active=state.get('active_config',config);attempt=int(state.get('attempt',0))
 while True:
  root=Path(active['run_root']);root.mkdir(parents=True,exist_ok=True)
  cfg_path=root/'worker_config.json';atomic_json(cfg_path,active)
  state={'original_config_sha256':identity,'active_config':active,'attempt':attempt,'updated_utc':now()};atomic_json(state_path,state)
  log=root/'worker.log';command=[sys.executable,'-u','-m','phase4_cjpe.runtime','--worker',str(cfg_path)]
  env=os.environ.copy();source=str(Path(__file__).resolve().parents[1]);env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
  env['PYTHONUNBUFFERED']='1'
  print('Starting/resuming:',active['name'],'run:',root,flush=True)
  began=time.monotonic();last_changed=began;last_stamp=None;shown=None
  with log.open('a',encoding='utf-8') as f:
   proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
   state['worker_pid']=proc.pid;atomic_json(state_path,state)
   try:
    while proc.poll() is None:
     progress=read_json(root/'status.json')
     # During the independent teacher stage, show the teacher's real progress.
     teacher=read_json(root/'teacher/status.json')
     if teacher and not (root/'teacher/complete.json').exists():progress=teacher
     stamp=progress.get('updated_utc')
     if stamp and stamp!=last_stamp:last_stamp=stamp;last_changed=time.monotonic()
     view=(progress.get('phase'),progress.get('step'),progress.get('cases'),progress.get('evaluated_cases'))
     if view!=shown:
      print(json.dumps({'worker_pid':proc.pid,'runtime_minutes':round((time.monotonic()-began)/60,1),**progress}),flush=True);shown=view
     atomic_json(base/'supervisor_status.json',{'checked_utc':now(),'worker_pid':proc.pid,'worker_alive':True,
      'active_run_root':str(root),'progress':progress,'minutes_without_progress':round((time.monotonic()-last_changed)/60,2)})
     if time.monotonic()-last_changed>active.get('stall_minutes',45)*60:
      _stop_owned_child(proc)
      atomic_json(root/'failure.json',{'type':'ProgressStalled','message':'No worker progress within the configured stall bound','failed_utc':now()})
      break
     time.sleep(20)
   finally:
    # Cell interruption must not orphan its child and leave GPU/file writers alive.
    _stop_owned_child(proc)
    atomic_json(base/'supervisor_status.json',{'checked_utc':now(),'worker_pid':proc.pid,'worker_alive':False,'active_run_root':str(root)})
  complete=read_json(root/'complete.json')
  if proc.returncode==0 and complete:
   state.update(completed=True,active_run_root=str(root),submission=complete['submission'],updated_utc=now());atomic_json(state_path,state)
   print('FINISHED. Upload this result ZIP:',complete['submission'],flush=True);return complete
  failure=read_json(root/'failure.json');message=failure.get('message','Worker exited; inspect '+str(log))
  lower=message.lower();is_oom='out of memory' in lower or failure.get('type')=='OutOfMemoryError'
  transient=failure.get('type') in ('ConnectionError','TimeoutError','OSError','HTTPError','PermissionError')
  if attempt>=2 or not (is_oom or transient):raise RuntimeError('Training stopped safely: '+message+'; log: '+str(log))
  attempt+=1
  if is_oom:
   previous=active;active=copy.deepcopy(active);active['run_root']=str(base/('oom_recovery_'+str(attempt)))
   active['max_tokens']=max(64,active['max_tokens']//2)
   if previous['max_tokens']<=64:active['chunks_per_case']=max(1,previous['chunks_per_case']//2)
   selected=Path(previous['run_root'])/'best.pt'
   if selected.exists():active['warm_start']=str(selected)
   active['recovery_parent']={'run_root':previous['run_root'],'reason':message,'optimizer':'fresh when warm-starting; new config is not exact resume'}
   print('OOM recovery uses separate run:',active['run_root'],'tokens/chunk:',active['max_tokens'],flush=True)
  else:
   print('Retrying transient failure with unchanged configuration/checkpoint:',message,flush=True);time.sleep(15)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True);args=parser.parse_args()
 from .train import worker
 worker(json.loads(Path(args.worker).read_text(encoding='utf-8')))

if __name__=='__main__':main()
