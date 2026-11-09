"""Generate independent Colab notebooks and byte-identical checkpoint upload bundles."""
import base64,hashlib,io,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase3_t4';OUT.mkdir(parents=True,exist_ok=True)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

PARENTS={
 'a0':('outputs/CS781_A0_Complete_20261006.zip','artifacts/checkpoint_resume.pt','data/prepared/tokenizer.json'),
 'b1':('outputs/CS781_B1_OwnBPE_Complete_20261006.zip','artifacts/b1_improved/checkpoints/latest.pt','data/tokenizer.json'),
 'c1':('outputs/CS781_C1_OwnBPE_Complete_20261006.zip','artifacts/c1_scratch/checkpoints/latest.pt','data/tokenizer.json'),
 'd222':('outputs/extension_archives/CS781_D222_Complete_20261011.zip','artifacts/pretrain/best.pt','tokenizers/own_bpe.json')}

def bundles():
 inventory={}
 for family,(relative,weights,tok) in PARENTS.items():
  archive=ROOT/relative;destination=OUT/f'Phase3_{family.upper()}_Input.zip'
  if destination.exists():
   with zipfile.ZipFile(destination) as z:manifest=json.loads(z.read('INPUT_MANIFEST.json'))
  else:
   hashes={};manifest={'family':family,'original_archive':relative,'original_members':{'checkpoint.pt':weights,'tokenizer.json':tok},
    'note':'Exact original checkpoint member, including optimizer state when originally present. Not stripped or retrained. Load only trusted own archives.'}
   tmp=destination.with_suffix('.partial')
   with zipfile.ZipFile(archive) as src,zipfile.ZipFile(tmp,'w',allowZip64=True) as dst:
    for original,name in ((weights,'checkpoint.pt'),(tok,'tokenizer.json')):
     h=hashlib.sha256()
     with src.open(original) as a,dst.open(name,'w',force_zip64=True) as b:
      while True:
       chunk=a.read(8*1024*1024)
       if not chunk:break
       h.update(chunk);b.write(chunk)
     hashes[name]=h.hexdigest()
    manifest['sha256']=hashes;dst.writestr('INPUT_MANIFEST.json',json.dumps(manifest,indent=2))
   tmp.replace(destination)
  inventory[family]={'path':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination),'members':manifest}
  print('PACKAGED',family,destination.stat().st_size,flush=True)
 return inventory

def cell(kind,text):
 obj={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
 if kind=='code':obj.update({'execution_count':None,'outputs':[]})
 return obj

def build():
 # Exclude server-only unused source modules, caches and the generator itself.
 payload=io.BytesIO()
 with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted((ROOT/'phase3_rr').rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.name!='build_notebooks.py':
    if 'legal_d' in p.parts and p.name not in ('__init__.py','model.py'):continue
    z.write(p,p.relative_to(ROOT).as_posix())
 blob=base64.b64encode(payload.getvalue()).decode();code_sha=hashlib.sha256(payload.getvalue()).hexdigest()
 (OUT/'Phase3_RR_Source.zip').write_bytes(payload.getvalue())
 jobs=[('01_D222_LoRMA','d222','lorma',2e-4),('02_B1_LoRA','b1','lora',2e-4),('03_C1_IA3','c1','ia3',3e-4),('04_A0_Distillation','a0','full',1e-5)]
 for name,family,method,lr in jobs:
  distill=name.startswith('04')
  cells=[cell('markdown',f'''# Phase 3: {name.replace('_',' ')} — own model, one T4

This is a separate, self-contained notebook. Use **Runtime → Change runtime type → T4 GPU** in Colab. Kaggle users must enable GPU and Internet and persist session output using Save Version.

No pretrained model is downloaded: only your archived Phase1 model and official RR data are used. Adapters are implemented from scratch. FP16 is used on T4. Scores and GPU execution are **not verified in the authoring environment**.

Four notebooks cover four model families, LoRMA/LoRA/IA3 and mandatory distillation. Four alone do **not** satisfy the document's opening six-submission minimum. Record the real hardware and LLM assistance honestly.

## Files to upload to your own Drive
Required: **Phase3_{family.upper()}_Input.zip**{', **Phase3_D222_Input.zip** for the teacher' if distill else ''}. Upload via Drive, not a giant browser `files.upload()` call. Put them in `MyDrive/CS781_Phase3/inputs/`.

Optional official dataset ZIP: put **Phase_3_RR.zip** in the same folder. Otherwise log into Hugging Face with an account already granted access and let the notebook fetch the seven official files. Never paste credentials into a shared notebook.

Persistent output: `MyDrive/CS781_Phase3/runs/{name}/`. Do not change settings while resuming the same run. Start a new run name for new hyperparameters.
''')]
  cells.append(cell('code','''# Keep Colab's installed CUDA-enabled torch. No PEFT or transformers models.
%pip -q install "tokenizers>=0.20,<0.23" "huggingface_hub>=0.26,<2" "scikit-learn>=1.4,<2"
import sys,torch,platform,shutil
print(platform.python_version(),torch.__version__)
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
print('ACTUAL GPU:',torch.cuda.get_device_name(0))
print('Free local GB:',round(shutil.disk_usage('/').free/1e9,2))
'''))
  cells.append(cell('code',f'''# Embedded source; no separate source upload required.
import base64,hashlib,io,zipfile,pathlib,sys
source_bytes=base64.b64decode({blob!r})
assert hashlib.sha256(source_bytes).hexdigest()=={code_sha!r}
CODE_ROOT=pathlib.Path('/content/phase3_code' if pathlib.Path('/content').exists() else '/kaggle/working/phase3_code')
CODE_ROOT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as z:
    for item in z.infolist():
        p=pathlib.PurePosixPath(item.filename)
        assert not p.is_absolute() and '..' not in p.parts
        destination=CODE_ROOT/item.filename
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(z.read(item))
sys.path.insert(0,str(CODE_ROOT))
print('Source SHA-256:',{code_sha!r})
'''))
  cells.append(cell('code',f'''# EDIT THESE PATHS if you use Kaggle or a different Drive folder.
from pathlib import Path
IS_COLAB=Path('/content').exists()
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PERSIST=Path('/content/drive/MyDrive/CS781_Phase3')
    INPUT_DIR=PERSIST/'inputs'
    WORK_ROOT=Path('/content/phase3_work')
else:
    INPUT_DIR=Path('/kaggle/input/cs781-phase3-inputs') # edit to your uploaded Kaggle dataset
    PERSIST=Path('/kaggle/working/CS781_Phase3')
    WORK_ROOT=Path('/kaggle/working/phase3_work')
WORK_ROOT.mkdir(parents=True,exist_ok=True)
RUN_ROOT=PERSIST/'runs'/{name!r}
RUN_ROOT.mkdir(parents=True,exist_ok=True)
required=[{('Phase3_'+family.upper()+'_Input.zip')!r}]{"+['Phase3_D222_Input.zip']" if distill else ''}
missing=[str(INPUT_DIR/n) for n in required if not (INPUT_DIR/n).exists()]
assert not missing, 'Upload these exact files first: '+str(missing)
print('Persistent run:',RUN_ROOT)
'''))
  cells.append(cell('code','''# Official data: upload ZIP OR authenticated download. Existing extracted files are reused.
from phase3_rr.data import acquire_data,load_official
DATA_ROOT=WORK_ROOT/'Phase_3_RR'
DATA_ZIP=INPUT_DIR/'Phase_3_RR.zip'
LOGIN_HUGGINGFACE=True # set False if your token is already cached
expected_files=['Train/'+n for n in ['CL_train.json','CL_dev.json','IT_train.json','IT_dev.json','label_vocab.json']]+['Test/CL_test.json','Test/IT_test.json']
if not all((DATA_ROOT/n).exists() for n in expected_files):
    if DATA_ZIP.exists():
        acquire_data(DATA_ROOT,uploaded_zip=DATA_ZIP)
    else:
        if LOGIN_HUGGINGFACE:
            from huggingface_hub import notebook_login
            notebook_login()
        acquire_data(DATA_ROOT,download=True)
dataset=load_official(DATA_ROOT)
for split in ['train','dev','test']:
    docs=dataset[split]
    print(split,len(docs),'documents',sum(len(d['text']) for d in docs),'sentences')
'''))
  cells.append(cell('code',f'''# Runtime estimates come from ACTUAL GPU calibration, not the initial planning estimate.
# Change RUN_ROOT to a new name before changing any setting of a previously started run.
CONFIG=dict(name={name!r},family={family!r},method={method!r},bundle=str(INPUT_DIR/{('Phase3_'+family.upper()+'_Input.zip')!r}),
    data_root=str(DATA_ROOT),run_root=str(RUN_ROOT),work_root=str(WORK_ROOT),seed=781,
    rank=8,alpha=16,max_tokens=256,context_sentences=1,accumulation=8,
    backbone_lr={lr},head_lr=0.0003,max_epochs=8,patience=3,
    training_hours=3.0,save_every=50,temperature=2.0,distill_weight=0.5)
{'''CONFIG.update(teacher_bundle=str(INPUT_DIR/'Phase3_D222_Input.zip'),teacher_hours=2.0,
    existing_teacher_run='') # optional verified completed notebook1 run directory
# Default is independent: trains its own teacher, then student. Allow extra time for teacher-logit generation.
''' if distill else ''}
print(CONFIG)
'''))
  cells.append(cell('code',f'''from phase3_rr.train import {'distill' if distill else 'run'},package_records
RESULT={'distill' if distill else 'run'}(CONFIG)
print('HELD-OUT macro F1:',RESULT['metrics']['all']['macro_f1'])
print('CL:',RESULT['metrics']['CL']['macro_f1'],'IT:',RESULT['metrics']['IT']['macro_f1'])
print('CODABENCH ZIP:',RESULT['submission'])
print('SHA-256:',RESULT['sha256'])
print('No private leaderboard score is known until you submit.')
'''))
  cells.append(cell('code','''# Download the prediction ZIP, NOT the source/model archive, to submit.
from pathlib import Path
submission=Path(RESULT['submission'])
assert submission.exists()
if IS_COLAB:
    from google.colab import files
    files.download(str(submission))
else:
    from IPython.display import FileLink,display
    display(FileLink(str(submission)))
'''))
  cells.append(cell('code','''# Optional larger archive for records; not a Codabench upload.
records=package_records(RESULT)
print('Selected weights/source/tokenizer/config/run records:',records)
print('Full optimizer resume checkpoint remains separately at',RUN_ROOT/'latest.pt')
'''))
  cells.append(cell('markdown','''## Resume / troubleshooting

- After disconnect, rerun the cells with the same Drive paths and configuration. `latest.pt` restores optimizer/scaler/RNG and epoch/group position. `best.pt` is selected by dev macroF1.
- Preserve both files. Do not replace input weights, edit source or change hyperparameters under the same run directory; a fingerprint mismatch is deliberate.
- T4 does not need BF16 or FlashAttention installation. If memory is insufficient, use a NEW run name and reduce `max_tokens` to192 or128. Accumulation controls effective batch size, not activation memory per sentence.
- A Hugging Face401/403 means dataset access/login needs attention. Upload the original official dataset ZIP instead; do not substitute a different dataset.
- Unsupported data schema or colliding CL/IT IDs stops safely; inspect the official supplied evaluator rather than silently rewriting IDs.
- The upload ZIP must contain only root `prediction.json`, all official document IDs, and one exact official label per original sentence.
- Training code can run correctly without improving F1; inspect per-label/domain errors and compare methods on the SAME dev sets.
- Cite LoRA (Hu et al.2021), LoRMA (Bihany et al.2025), IA3 (Liu et al.2022) and distillation (Hinton et al.2015), plus the capstone/IL-TUR data. Declare coding-assistant use and actual GPU honestly.
'''))
  notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'name':name+'.ipynb'},'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':cells}
  for i,c in enumerate(cells):c['id']=f'cell-{i:03d}'
  (OUT/(name+'.ipynb')).write_text(json.dumps(notebook,indent=1),encoding='utf-8')
 inventory=bundles();(OUT/'UPLOAD_INPUTS.json').write_text(json.dumps(inventory,indent=2))
 (OUT/'README.md').write_text('''# Phase3 — four independent T4 notebooks

Upload each .ipynb to Colab; select T4. Put input ZIPs in MyDrive/CS781_Phase3/inputs/ on that session's account:

|Notebook|Required model bundles|
|---|---|
|01_D222_LoRMA|Phase3_D222_Input.zip|
|02_B1_LoRA|Phase3_B1_Input.zip|
|03_C1_IA3|Phase3_C1_Input.zip|
|04_A0_Distillation|Phase3_A0_Input.zip + Phase3_D222_Input.zip|

All notebook code is embedded; no source ZIP upload needed. Official Phase_3_RR.zip is optional if authenticated Hugging Face download works. Dataset access may require approval. Do not upload private keys or SSH passwords.

Model input bundles contain original exact checkpoint files, including optimizer buffers where originally present. Local Python has no compatible torch installed, so these are selectively repackaged originals, NOT stripped model-only conversions. Notebook loading discards old optimizer state and starts a new task optimizer. Original complete archives are untouched. See UPLOAD_INPUTS.json for byte counts, SHA and parent member paths.

Timing: initially allow1–4h for each adaptation, possibly longer depending on the actual data. Notebook4 additionally needs its teacher and training-logit generation, so allow4–8h or more. Colab/Kaggle speed and session availability vary. The notebook calibrates actual inference and reserves export time; training_hours is a budget, not a convergence promise. Parallel sessions can overlap but do not guarantee completion before the16October23:59IST deadline. Use your own permitted accounts and follow the platform's terms/quotas; no identity misrepresentation or quota-evasion code is included.

Every completed run creates UPLOAD_Phase3_<experiment>.zip in its persistent run folder. Only that ZIP goes to Codabench. MODEL_RECORDS_*.zip is for records. Actual prediction ZIPs do not exist until you train; none are fabricated here.

Four experiments are the requested reduced scope; they do not satisfy the opening six-submission rule alone. Three adaptation techniques plus mandatory distillation cover four own-model families. No local T4 training or leaderboard gain has been established.
''',encoding='utf-8')
 with zipfile.ZipFile(OUT/'Phase3_Four_Notebooks_and_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():
   if p.suffix=='.ipynb' or p.name in ('Phase3_RR_Source.zip','README.md','UPLOAD_INPUTS.json'):z.write(p,p.name)
 print('NOTEBOOKS AND INPUT BUNDLES READY',flush=True)

if __name__=='__main__':build()
