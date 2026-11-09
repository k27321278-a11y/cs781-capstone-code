"""Build the isolated B1 study notebook; never overwrite the earlier runner."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    vendor = json.loads((ROOT / 'VENDOR_MANIFEST.json').read_text())
    fixed = json.loads((ROOT / 'FIXED_HOLDOUT_IDS.json').read_text())
    control_path = ROOT / 'SEGMENT_CONTROL.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else dict(segment=1, resume_dataset=None, expected_resume_manifest=None)
    blob = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(blob, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((ROOT / 'source' / 'b1_study').glob('*.py')):
            name = p.relative_to(ROOT / 'source').as_posix()
            contents = p.read_bytes()
            archive.writestr(name, contents)
            manifest[name] = hashlib.sha256(contents).hexdigest()
    raw = blob.getvalue()
    cells = []

    def cell(kind, text):
        value = dict(cell_type=kind, id='b1-study-' + str(len(cells)), metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(text)
            value.update(outputs=[], execution_count=None)
        cells.append(value)

    cell('markdown', '''# CS781 B1: complete-pass reference and document hierarchy

One independently trained model per prediction. Own Phase1 B1 weights and tokenizer only. This segment runs a full-data reference on GPU0 and a 512-token-window document hierarchy on GPU1. The hierarchy uses two document transformer blocks and a bidirectional GRU. Encoder growth and optional FFN expansion are implemented but are not launched before earlier evidence.

Three-hour training segments save exact resumable state. A budget boundary is **paused**, not converged. Published outputs must be downloaded and verified before another segment continues. No live `/kaggle/working` checkpoint is claimed as a D-drive backup. All experiments are after the original course deadline. Accuracy is measured, never guaranteed.
''')
    cell('code', '''import os,sys,time,subprocess,importlib.util
from pathlib import Path
SESSION_STARTED=time.time()
SESSION_END=SESSION_STARTED+5.2*3600
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
needed={'ijson':'ijson>=3.2,<4','sklearn':'scikit-learn>=1.4,<2','tokenizers':'tokenizers'}
missing=[p for m,p in needed.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
print('B1_STUDY_BOOT',{'session_start':SESSION_STARTED,'session_end':SESSION_END})
'''.replace('SESSION_STARTED+5.2*3600', 'SESSION_STARTED+' + str(float(control.get('session_hours', 5.2))) + '*3600'))
    cell('code', f'''import base64,hashlib,io,json,zipfile,shutil
EXPECTED_VENDOR={vendor!r}
EXPECTED_STUDY={manifest!r}
matches=[]
for p in Path('/kaggle/input').rglob('SOURCE_MANIFEST.json'):
    try:
        if json.loads(p.read_text())==EXPECTED_VENDOR:matches.append(p)
    except (ValueError,OSError):pass
assert len(matches)==1,'Attach exact recovered course runner dataset.'
CODE=Path('/kaggle/working/b1_study_source');CODE.mkdir(parents=True,exist_ok=True)
for relative,wanted in EXPECTED_VENDOR.items():
    data=(matches[0].parent/relative).read_bytes()
    assert hashlib.sha256(data).hexdigest()==wanted,'Changed vendor source:'+relative
    dest=CODE/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
raw=base64.b64decode({base64.b64encode(raw).decode()!r})
assert hashlib.sha256(raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name,wanted in EXPECTED_STUDY.items():
        data=archive.read(name);assert hashlib.sha256(data).hexdigest()==wanted
        dest=CODE/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
sys.path.insert(0,str(CODE))
from phase4_cjpe.common import atomic_json,sha
PERSIST=Path('/kaggle/working/CS781_B1_Hierarchy_Depth');PERSIST.mkdir(parents=True,exist_ok=True)
WORK=Path('/kaggle/temp/cs781_b1_study');WORK.mkdir(parents=True,exist_ok=True)
(PERSIST/'STUDY_SOURCE.zip').write_bytes(raw)
atomic_json(PERSIST/'SOURCE_MANIFEST.json',dict(vendor=EXPECTED_VENDOR,study=EXPECTED_STUDY))
FIXED=CODE/'FIXED_HOLDOUT_IDS.json';atomic_json(FIXED,{fixed!r})
EXPECTED_RECOVERED={{'Train/train.json':{{'sha256':'49bfc5f4d441833e2468e4c280c390ce361e2bfaffd5fbf1ad54157cddda164c','bytes':611284455}},'Train/dev.json':{{'sha256':'a071b080d3ffcd9789009cf46f6e4f1008142de0f44db9b97df9b64b6ee201c9','bytes':22387465}},'Test/test.json':{{'sha256':'88913c10c288a6f77a030d0391904d40b645fc242f2774f0ef8216e9f5afa792','bytes':35050736}}}}
recovered=[]
for p in Path('/kaggle/input').rglob('RECOVERY_MANIFEST.json'):
    try:
        if json.loads(p.read_text()).get('files')==EXPECTED_RECOVERED:recovered.append(p)
    except (ValueError,OSError):pass
assert len(recovered)==1,'Corrected recovered data unavailable or ambiguous.'
DATA=recovered[0].parent
for relative,wanted in EXPECTED_RECOVERED.items():
    assert (DATA/relative).stat().st_size==wanted['bytes'] and sha(DATA/relative)==wanted['sha256']
bundles=[]
for p in Path('/kaggle/input').rglob('INPUT_MANIFEST.json'):
    try:
        if json.loads(p.read_text()).get('family')=='b1':bundles.append(p.parent)
    except (ValueError,OSError):pass
assert len(bundles)==1,'Attach exact own Phase1 B1 input.'
BUNDLE=bundles[0]
assert sha(BUNDLE/'checkpoint.pt')=='d93a2f1796e90b9fdbe4a7b542f76eb6065dab619feed4a1ef5a9952e2376a1f'
assert sha(BUNDLE/'tokenizer.json')=='5c0bd30290e0bd6919253b1ff864f52053924808aabd33482df92cc4bd471c8e'
assert shutil.disk_usage(PERSIST).free>8*1024**3,'Need8GiB for best/latest states and exports.'
atomic_json(PERSIST/'INPUT_PROVENANCE.json',dict(data=EXPECTED_RECOVERED,bundle=str(BUNDLE),parent_sha256=sha(BUNDLE/'checkpoint.pt')))
CONTROL={control!r}
RESUME_ROOTS={{}}
if CONTROL.get('resume_dataset'):
    candidates=[]
    for p in Path('/kaggle/input').rglob('RESUME_MANIFEST.json'):
        if sha(p)==CONTROL['expected_resume_manifest']:candidates.append(p)
    assert len(candidates)==1,'Exact verified previous-segment resume manifest required.'
    manifest_path=candidates[0];resume=json.loads(manifest_path.read_text())
    for job in resume['jobs']:
        dest=WORK/'resume'/job['name'];dest.mkdir(parents=True,exist_ok=True)
        for name,record in job['files'].items():
            source=manifest_path.parent/record['filename']
            assert source.stat().st_size==record['bytes'] and sha(source)==record['sha256'],'Resume file mismatch'
            target=dest/name
            if not target.exists():target.symlink_to(source)
        RESUME_ROOTS[job['name']]=str(dest)
print('B1_STUDY_INPUTS_VERIFIED',{{'bundle':str(BUNDLE),'data':str(DATA)}})
''')
    cell('code', '''import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==2,'Two isolated GPUs are required for this comparison.'
print('ALLOCATED_GPUS',[(i,torch.cuda.get_device_name(i)) for i in range(2)])
BASE=dict(seed=781,max_epochs=4,accumulation=8,warmup_updates=100,patience=5,backbone_lr=1e-5,head_lr=3e-4,
          encoder_depth=6,expand_ffn=False,chunk_batch=4,segment=CONTROL['segment'],segment_hours=3.,resume_root=None,
          bundle=str(BUNDLE),data_root=str(DATA),fixed_holdout=str(FIXED),session_end_unix=SESSION_END)
definitions=[dict(name='B1_FullData_Reference',representation='reference',head_mode='reference',max_tokens=256,chunks_per_case=8,overlap=0,window_policy='uniform'),
             dict(name='B1_DocumentHierarchy',representation='windows',head_mode='transformer_gru',max_tokens=512,chunks_per_case=16,overlap=100,window_policy='tail')]
JOBS=[]
for definition in definitions:
    config=dict(BASE,**definition)
    config.update(run_root=str(PERSIST/'runs'/config['name']),work_root=str(WORK/config['name']))
    config['resume_root']=RESUME_ROOTS.get(config['name'])
    JOBS.append(config)
atomic_json(PERSIST/'RUN_SCHEDULE.json',JOBS)
from b1_study.run import launch
RESULTS=launch(JOBS,PERSIST)
print('SEGMENT_FINISHED',[(r['name'],r['state'],r.get('optimizer_updates'),r.get('completed_epochs')) for r in RESULTS])
''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python', version='3.12')), nbformat=4, nbformat_minor=5)
    filename = 'cs781-phase4-b1-hierarchy-depth.ipynb'
    (ROOT / filename).write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    metadata = dict(id='krshnakarthikeyagn/cs781-b1-hierarchy-and-depth', title='CS781 B1 Hierarchy and Depth',
        code_file=filename, language='python', kernel_type='notebook', is_private=True, enable_gpu=True,
        enable_tpu=False, enable_internet=True, dataset_sources=['krshnakarthikeyagn/cs781-phase4-private-inputs-20261022',
        'krshnakarthikeyagn/cs781-phase4-recovered-data-runner-20261022'], kernel_sources=[], competition_sources=[],
        model_sources=[], machine_shape='NvidiaTeslaT4')
    if control.get('resume_dataset'):
        metadata['dataset_sources'].append(control['resume_dataset'])
    (ROOT / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (ROOT / 'BUILD_RECORD.json').write_text(json.dumps(dict(study=manifest, vendor=vendor,
            notebook_sha256=hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()), indent=2))
    print(json.dumps(dict(notebook=str(ROOT / filename), code_cells=sum(c['cell_type'] == 'code' for c in cells), source_files=len(manifest))))


if __name__ == '__main__':
    build()
