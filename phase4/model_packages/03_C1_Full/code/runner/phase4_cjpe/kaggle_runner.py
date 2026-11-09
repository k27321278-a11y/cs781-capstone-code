"""Two isolated GPU workers, bounded recovery, and verified incremental exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def read_json(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):return {}


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.incoming')
    tmp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def assert_no_live_worker(root,platform=None):
    root=Path(root);owner=read_json(root/'ownership.json')
    if owner.get('pid') and (platform or os.name)!='nt':
        try:command=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes()
        except FileNotFoundError:command=b''
        if str(root/'job_config.json').encode() in command:raise RuntimeError('This run already has a live worker')


def gpu_env(index):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(index),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    source=str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
    return env


def validate_configs(configs):
    if not 1<=len(configs)<=2:raise ValueError('Exactly one or two configs per pair')
    roots=[str(Path(c['run_root']).resolve()) for c in configs]
    if len(set(roots))!=len(roots):raise ValueError('Workers must have distinct run roots')
    if len({c['name'] for c in configs})!=len(configs):raise ValueError('Duplicate experiment names')


def verify_zip(path,expected_ids):
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['prediction.json'] or z.testzip():raise ValueError('Wrong ZIP layout or CRC')
        rows=json.loads(z.read('prediction.json'))
    if not isinstance(rows,list):raise ValueError('Predictions must be a list')
    keys=[]
    for r in rows:
        if not isinstance(r,dict) or set(r)!={'id','pred'} or type(r['id']) is not str or type(r['pred']) is not int or r['pred'] not in (0,1):
            raise ValueError('Invalid submission row')
        keys.append(r['id'])
    if len(keys)!=len(set(keys)) or len(keys)!=len(expected_ids) or set(keys)!=set(expected_ids):
        raise ValueError('Wrong or repeated test IDs')
    return dict(sha256=sha(path),bytes=Path(path).stat().st_size,cases=len(keys),verified_submission=True)


def probe_device(env):
    probe='import torch,json; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(json.dumps({"name":torch.cuda.get_device_name(0),"count":torch.cuda.device_count()}))'
    result=subprocess.run([sys.executable,'-c',probe],env=env,capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError('GPU isolation preflight failed: '+result.stderr[-1500:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def stop_process(proc):
    if proc.poll() is not None:return
    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
    else:proc.terminate()
    try:proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
        else:proc.kill()
        proc.wait()


def collect_export(root,output_root,name,expected_ids,seen,callback):
    result=read_json(Path(root)/'latest_export.json')
    if not result or not Path(result.get('submission','')).is_file():return
    check=verify_zip(result['submission'],expected_ids)
    marker=(result.get('stage'),check['sha256'])
    if marker in seen:return
    folder=Path(output_root)/'SUBMISSIONS';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/('UPLOAD_Phase4_'+name+'_'+str(result.get('stage','result'))+'.zip')
    temp=dest.with_suffix('.incoming');shutil.copy2(result['submission'],temp);temp.replace(dest)
    record={**result,**check,'submission':str(dest),'name':name}
    atomic_json(dest.with_suffix('.result.json'),record)
    seen.add(marker);print('ZIP_READY '+json.dumps(record),flush=True)
    if callback:callback(record)


def run_pair(configs,output_root,on_worker_complete=None,expected_ids=None,
             worker_module='phase4_cjpe.kaggle_runner',probe_cuda=True,poll_seconds=10):
    validate_configs(configs)
    if expected_ids is None:raise ValueError('Official test ID set is required')
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    workers=[];results=[]
    def launch(w):
        root=Path(w['active']['run_root']);root.mkdir(parents=True,exist_ok=True)
        cfg=root/'job_config.json';atomic_json(cfg,w['active'])
        log=(root/'worker.log').open('a',encoding='utf-8')
        env=gpu_env(w['gpu'])
        if probe_cuda and not w.get('device'):w['device']=probe_device(env)
        proc=subprocess.Popen([sys.executable,'-u','-m',worker_module,'--job',str(cfg)],
              stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt')
        w.update(proc=proc,log=log,last_progress=time.monotonic(),last_stamp=None)
        atomic_json(Path(w['original']['run_root'])/'scheduler_state.json',
            {'original_config_sha256':identity(w['original']),'active_config':w['active'],'attempt':w['attempt']})
        atomic_json(root/'ownership.json',{'pid':proc.pid,'gpu':w['gpu'],'config_sha256':identity(w['active'])})
        print('WORKER_STARTED '+json.dumps({'name':w['active']['name'],'pid':proc.pid,'physical_gpu':w['gpu']}),flush=True)
    try:
        for gpu,config in enumerate(configs):
            root=Path(config['run_root']);prior=read_json(root/'job_config.json')
            if prior and identity(prior)!=identity(config):raise ValueError('Run configuration changed; choose a new run root')
            owner=read_json(root/'ownership.json')
            if owner.get('pid') and os.name!='nt':
                try:command=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes()
                except FileNotFoundError:command=b''
                if str(root/'job_config.json').encode() in command:raise RuntimeError('This run already has a live worker')
            saved=read_json(root/'scheduler_state.json')
            if saved and saved.get('original_config_sha256')!=identity(config):raise ValueError('Saved scheduler configuration changed')
            active=saved.get('active_config',dict(config))
            assert_no_live_worker(active['run_root'])
            w={'active':active,'original':dict(config),'gpu':gpu,'attempt':int(saved.get('attempt',0)),'seen':set()}
            workers.append(w);launch(w)
        while workers:
            for w in workers[:]:
                root=Path(w['active']['run_root']);proc=w['proc']
                progress=read_json(root/'status.json')
                teacher=read_json(root/'teacher/status.json')
                if teacher and not (root/'teacher/complete.json').exists():progress=teacher
                stamp=progress.get('updated_utc')
                if stamp and stamp!=w['last_stamp']:
                    w['last_stamp']=stamp;w['last_progress']=time.monotonic()
                try:collect_export(root,output_root,w['original']['name'],expected_ids,w['seen'],on_worker_complete)
                except (ValueError,zipfile.BadZipFile) as exc:
                    stop_process(proc);w['invalid_export']=True
                    atomic_json(root/'failure.json',{'type':'InvalidSubmission','message':str(exc)})
                atomic_json(output_root/('GPU_'+str(w['gpu'])+'_STATUS.json'),
                    {'name':w['active']['name'],'pid':proc.pid,'gpu':w['gpu'],'progress':progress,
                     'checked_unix':time.time(),'worker_alive':proc.poll() is None})
                end=float(os.environ.get('PHASE4_SESSION_END','inf'))
                stalled=time.monotonic()-w['last_progress']>w['active'].get('stall_minutes',45)*60
                if proc.poll() is None and (stalled or time.time()>=end):
                    stop_process(proc)
                    atomic_json(root/'failure.json',{'type':'ProgressStalled' if stalled else 'SessionBoundary',
                        'message':'No progress within bound' if stalled else 'Session reserve reached; latest checkpoint preserved'})
                if proc.poll() is None:continue
                w['log'].close()
                result=read_json(root/'job_result.json') or read_json(root/'complete.json')
                if proc.returncode==0 and result and not w.get('invalid_export'):
                    state='paused' if result.get('session_paused') else 'complete'
                    record={**result,'name':w['original']['name'],'state':state,'gpu':w['gpu'],'active_run_root':str(root)}
                    results.append(record);workers.remove(w)
                    atomic_json(output_root/('TERMINAL_'+w['original']['name']+'.json'),record)
                    if on_worker_complete:on_worker_complete(record)
                    continue
                failure=read_json(root/'failure.json');kind=failure.get('type','UnknownExit')
                message=failure.get('message','Worker exited without result')
                transient=kind in ('ConnectionError','TimeoutError','OSError','HTTPError')
                oom=kind=='OutOfMemoryError' or 'out of memory' in message.lower()
                if w['attempt']<2 and (transient or oom) and time.time()<end-1800:
                    w['attempt']+=1
                    if oom:
                        old=w['active'];new=dict(old);new['run_root']=str(Path(w['original']['run_root'])/('oom_recovery_'+str(w['attempt'])))
                        new['max_tokens']=max(64,old['max_tokens']//2)
                        if old['max_tokens']<=64:new['chunks_per_case']=max(1,old['chunks_per_case']//2)
                        selected=Path(old['run_root'])/'best.pt'
                        if selected.exists():new['warm_start']=str(selected)
                        new['recovery_parent']={'run_root':old['run_root'],'reason':message,'optimizer':'fresh; configuration changed'}
                        w['active']=new
                    print('BOUNDED_RETRY '+json.dumps({'name':w['active']['name'],'reason':message,'attempt':w['attempt']}),flush=True)
                    launch(w);continue
                record={'name':w['original']['name'],'state':'paused' if kind=='SessionBoundary' else 'failed',
                        'failure':failure,'returncode':proc.returncode,'active_run_root':str(root),'gpu':w['gpu']}
                results.append(record);workers.remove(w)
                atomic_json(output_root/('TERMINAL_'+w['original']['name']+'.json'),record)
                if on_worker_complete:on_worker_complete(record)
            atomic_json(output_root/'PAIR_STATUS.json',{'running':[w['active']['name'] for w in workers],'terminal':results,'checked_unix':time.time()})
            if workers:time.sleep(poll_seconds)
    finally:
        for w in workers:
            stop_process(w['proc']);w['log'].close()
    return results


def run_schedule(jobs,output_root,expected_ids,on_worker_complete=None):
    output_root=Path(output_root);results=[]
    for start in range(0,len(jobs),2):
        if time.time()>=float(os.environ.get('PHASE4_SESSION_END','inf'))-3600:
            atomic_json(output_root/'PENDING_JOBS.json',jobs[start:]);break
        pair=[dict(j) for j in jobs[start:start+2]]
        ready=[]
        for cfg in pair:
            if cfg.get('experiment')=='distillation':
                teachers=[r for r in results if r.get('family')=='d222' and r.get('method')=='lorma' and r['state']=='complete']
                if teachers:cfg['teacher_checkpoint']=teachers[0]['selected_checkpoint']
                elif not cfg.get('teacher_checkpoint'):
                    record={'name':cfg['name'],'state':'pending','reason':'Selected clean D222 LoRMA teacher unavailable'}
                    results.append(record);atomic_json(output_root/('TERMINAL_'+cfg['name']+'.json'),record)
                    continue
            ready.append(cfg)
        if ready:results.extend(run_pair(ready,output_root,on_worker_complete,expected_ids))
        atomic_json(output_root/'RUN_SUMMARY.json',{'experiments':results,'updated_unix':time.time()})
    dest=output_root/'ALL_VERIFIED_SUBMISSIONS.zip'
    with zipfile.ZipFile(dest.with_suffix('.incoming'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((output_root/'SUBMISSIONS').glob('UPLOAD_*.zip')):z.write(p,p.name)
    dest.with_suffix('.incoming').replace(dest)
    print('ALL_READY_ZIPS '+str(dest),flush=True)
    return results


def job_main(config_path):
    import torch
    from .train import worker
    cfg=read_json(config_path);root=Path(cfg['run_root'])
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if torch.cuda.device_count()!=1:raise RuntimeError('Child must see exactly one GPU')
        torch.set_num_threads(2)
        result=worker(cfg)
        result.update(family=cfg['family'],method=cfg['method'])
        atomic_json(root/'job_result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();job_main(args.job)
