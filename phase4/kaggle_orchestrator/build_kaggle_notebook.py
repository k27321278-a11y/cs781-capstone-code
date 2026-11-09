"""Build a self-contained Kaggle orchestrator; do not touch original notebooks."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SOURCE=ROOT/'source'


def build():
    original=json.loads((PROJECT/'inputs/ORIGINAL_INPUT_MANIFEST.json').read_text())
    expected={f:r['members']['sha256'] for f,r in original.items()}
    stream=io.BytesIO();source_hashes={}
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SOURCE.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.txt','.md'):continue
            name=p.relative_to(SOURCE).as_posix();z.write(p,name)
            source_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    raw=stream.getvalue();digest=hashlib.sha256(raw).hexdigest()
    (ROOT/'Kaggle_Phase4_EvidenceV2_Source.zip').write_bytes(raw)
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(source_hashes,indent=2))
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'id':f'kaggle-phase4-{len(cells):02d}','metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':ast.parse(text);c.update(outputs=[],execution_count=None)
        cells.append(c)
    add('markdown', '''# CS781 Phase 4 — two T4 workers with resumable checkpoints

Private, after-deadline experiments starting from the four own Phase 1 input bundles. Six adaptations: D222 LoRMA+, B1 partial, C1 full, A0 distillation, D222 adapters, C1 BitFit. A shared new evidence head pools tokens, ordered chunks and document features; these head changes do not count as extra adaptation methods.

Attach **CS781 Phase4 Private Inputs 20261022** and a private dataset containing the corrected official `train.json`, `dev.json`, `test.json`. This notebook intentionally stops when corrected data are missing. It does not download gated files or use leaked labels.

Select **GPU T4 x2**. Run preflight while present. Once both workers are confirmed progressing and saving checkpoints, **Save & Run All** runs as a background Kaggle version. Stop the separate interactive session after the background run starts to avoid duplicate GPU usage. The run is bounded by the session limit; arbitrary future service interruptions are not guaranteed to repair themselves.

Output: `/kaggle/working/CS781_Phase4_EvidenceV2_20261022/SUBMISSIONS`. Each completed experiment emits early/final upload ZIPs; use one chosen ZIP per actual method. Checkpoints and records remain in the corresponding run folder. A combined ZIP contains whichever verified submissions are ready. Results are measured, not guaranteed.
''')
    add('code', '''import os,time,sys,subprocess,importlib.util
from pathlib import Path
SESSION_STARTED=time.time()
os.environ['PHASE4_SESSION_END']=str(SESSION_STARTED+11.4*3600)
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
needed={'ijson':'ijson>=3.2,<4','sklearn':'scikit-learn>=1.4,<2','tokenizers':'tokenizers'}
missing=[package for module,package in needed.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
print('Dependencies ready; notebook session clock started.')
''')
    add('code',f'''import base64,hashlib,io,json,zipfile,shutil
raw=base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest()=={digest!r}
CODE_ROOT=Path('/kaggle/working/phase4_evidence_v2_source')
CODE_ROOT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for info in z.infolist():
        p=Path(info.filename)
        assert not p.is_absolute() and '..' not in p.parts
        dest=CODE_ROOT/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
sys.path.insert(0,str(CODE_ROOT))
PERSIST=Path('/kaggle/working/CS781_Phase4_EvidenceV2_20261022')
WORK=Path('/kaggle/temp/cs781_phase4_evidence_v2')
PERSIST.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
(PERSIST/'EXACT_SOURCE.zip').write_bytes(raw)
from phase4_cjpe.common import sha,atomic_json
EXPECTED={expected!r}
BUNDLES={{}}
for manifest_path in Path('/kaggle/input').rglob('INPUT_MANIFEST.json'):
    try:manifest=json.loads(manifest_path.read_text())
    except (ValueError,OSError):continue
    family=manifest.get('family')
    if family not in EXPECTED:continue
    if family in BUNDLES:raise ValueError('Ambiguous duplicate family input: '+family)
    bundle=manifest_path.parent
    assert manifest['sha256']==EXPECTED[family], 'Wrong original model manifest'
    for filename,expected_sha in EXPECTED[family].items():
        assert sha(bundle/filename)==expected_sha, 'Input member hash mismatch: '+str(bundle/filename)
    BUNDLES[family]=bundle
    print('Verified original model bundle:',family,bundle)
assert set(BUNDLES)==set(EXPECTED), 'Attach all four own model bundles using Add Input.'
assert shutil.disk_usage(PERSIST).free>12*1024**3, 'Need at least 12 GiB free output space for selected and resumable states.'
atomic_json(PERSIST/'INPUT_PROVENANCE.json',{{f:{{'path':str(p),'sha256':EXPECTED[f]}} for f,p in BUNDLES.items()}})
print('Four own input bundles verified. No checkpoint has been trained or changed.')
''')
    add('code', '''# Corrected official data preflight: accept the three attached immutable files only.
SIZES={'train.json':736645732,'dev.json':44963788,'test.json':35267668}
found={}
for filename,size in SIZES.items():
    candidates=[p for p in Path('/kaggle/input').rglob(filename) if p.stat().st_size==size]
    if len(candidates)!=1:
        raise RuntimeError('DATA REQUIRED: attach corrected official '+filename+' ('+str(size)+' bytes). Found '+str(len(candidates))+' exact-size candidates. GPU training has not started.')
    found[filename]=candidates[0]
DATA_ROOT=WORK/'official_data'
for filename,p in found.items():
    relative=Path('Test' if filename=='test.json' else 'Train')/filename
    dest=DATA_ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(p)
atomic_json(PERSIST/'OFFICIAL_DATA_PROVENANCE.json',{
    'repository':'Exploration-Lab/CS-781-Capstone',
    'expected_revision':'93b0059be756c6c461f3a225de25aabda7d5f84a',
    'revision_is_source_metadata_not_a_content_hash_claim':True,
    'files':{name:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in found.items()}})
from phase4_cjpe.data import index_data
db,DATA_AUDIT=index_data(DATA_ROOT,WORK/'shared_official_index',progress_root=PERSIST/'data_preflight')
atomic_json(PERSIST/'DATA_AUDIT.json',DATA_AUDIT)
print('Corrected data passed duplicate and cross-split checks:',DATA_AUDIT['counts'])
''')
    add('code', '''# Actual GPU / own-checkpoint preflight. Synthetic token IDs; no optimizer step.
import gc,torch
from phase4_cjpe.models import make_model
assert torch.cuda.is_available() and torch.cuda.device_count()==2, 'Choose GPU T4 x2 before launch.'
print('Allocated:',[(i,torch.cuda.get_device_name(i)) for i in range(2)])
SMOKE=[]
for n,(family,method) in enumerate([('d222','lorma'),('b1','partial'),('c1','full'),('d222','adapters'),('a0','full'),('c1','bitfit')]):
    index=n%2;device=torch.device('cuda:'+str(index));torch.cuda.set_device(index)
    model,evidence=make_model(BUNDLES[family],family,method,8,16);model.to(device).train()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        logits=model([[1,4,5,6,7,8,9,2],[1,9,8,7,6,5,4,2]],[0.,1.],device)
        loss=torch.nn.functional.cross_entropy(logits.float(),torch.tensor([1],device=device))
    loss.backward()
    gradients=[p.grad for p in model.backbone.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and any(bool(p.abs().sum()>0) for p in gradients), 'No adaptation gradient'
    assert torch.isfinite(logits).all() and all(bool(torch.isfinite(g).all()) for g in gradients), 'Nonfinite preflight'
    SMOKE.append({'family':family,'method':method,'gpu':index,'parameters':evidence['parameters'],
      'trainable':evidence['trainable_parameters'],'finite':True,'nonzero_backbone_gradient':True})
    del gradients,model,logits,loss;gc.collect();torch.cuda.empty_cache()
atomic_json(PERSIST/'GPU_PREFLIGHT.json',SMOKE)
print('All six own-checkpoint adaptation preflights passed:',SMOKE)
''')
    add('code', '''# Six new, truthfully after-deadline experiments. Keep configs unchanged for exact resume.
definitions=[('01_D222_LoRMA','d222','lorma',2e-4,2.5),('02_B1_Partial','b1','partial',2e-5,2.5),
 ('03_C1_Full','c1','full',1e-5,3.5),('05_D222_Adapters','d222','adapters',3e-4,2.5),
 ('04_A0_Distillation','a0','full',1e-5,2.5),('06_C1_BitFit','c1','bitfit',2e-4,2.5)]
JOBS=[]
for name,family,method,lr,hours in definitions:
    name=name+'_KaggleEvidenceV2'
    cfg=dict(name=name,family=family,method=method,experiment='distillation' if family=='a0' else 'adaptation',
       rank=8,alpha=16,seed=781,max_tokens=256,chunks_per_case=8,region_chars=8192,
       max_train_cases=12000,holdout_per_class=256,max_epochs=4,accumulation=8,
       backbone_lr=lr,head_lr=3e-4,training_hours=hours,eval_every=100,save_every=10,patience=5,
       teacher_hours=1.5,kd_temperature=2.,kd_weight=.5,stall_minutes=45,export_reserve_minutes=45,
       deadline_utc='2026-10-21T18:29:00Z',allow_after_deadline=True,
       run_root=str(PERSIST/'runs'/name),work_root=str(WORK),data_root=str(DATA_ROOT),bundle=str(BUNDLES[family]))
    if family=='a0':cfg['teacher_bundle']=str(BUNDLES['d222'])
    JOBS.append(cfg)
atomic_json(PERSIST/'RUN_SCHEDULE.json',JOBS)
import sqlite3
with sqlite3.connect(db) as conn:EXPECTED_TEST_IDS=[r[0] for r in conn.execute('SELECT id FROM cases WHERE split="test" ORDER BY id')]
print('Schedule:',[(j['name'],j['training_hours']) for j in JOBS])
print('Session remaining hours:',round((float(os.environ['PHASE4_SESSION_END'])-time.time())/3600,2))
''')
    add('code', '''# Launch two isolated workers and collect each verified ZIP as it appears.
from phase4_cjpe.kaggle_runner import run_schedule
from IPython.display import display,FileLink
def completed(record):
    if record.get('verified_submission'):
        display(FileLink(record['submission']))
    elif record.get('state'):
        print('EXPERIMENT TERMINAL:',record['name'],record['state'])
RESULTS=run_schedule(JOBS,PERSIST,EXPECTED_TEST_IDS,on_worker_complete=completed)
display(FileLink(str(PERSIST/'ALL_VERIFIED_SUBMISSIONS.zip')))
print('Completed/paused/failed records are in RUN_SUMMARY.json. Raw checkpoints are under runs/.')
''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
    out=ROOT/'CS781_Phase4_T4x2_Orchestrator.ipynb';out.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    (ROOT/'BUILD_RECORD.json').write_text(json.dumps({'notebook':str(out),'source_sha256':digest,
       'source_files':len(source_hashes),'cells':len(cells),'code_cells_parsed':sum(c['cell_type']=='code' for c in cells),
       'gpu_execution_verified':False},indent=2))
    print(out)


if __name__=='__main__':build()
