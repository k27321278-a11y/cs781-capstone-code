import ast,hashlib,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'official_data'
nb=json.loads((ROOT/'CS781_Phase4_T4x2_Orchestrator.ipynb').read_text())
hashes=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
recovery=json.loads((DATA/'RECOVERY_MANIFEST.json').read_text())
original=''.join(nb['cells'][2]['source'])
tail=original[original.index('sys.path.insert(0,str(CODE_ROOT))'):]
prefix=f'''import hashlib,io,json,zipfile,shutil
EXPECTED_SOURCE={hashes!r}
manifests=[]
for p in Path('/kaggle/input').rglob('SOURCE_MANIFEST.json'):
    try:
        if json.loads(p.read_text())==EXPECTED_SOURCE:manifests.append(p)
    except (ValueError,OSError):pass
assert len(manifests)==1, 'Attach the private CS781 recovered data and runner dataset.'
CODE_ROOT=Path('/kaggle/working/phase4_evidence_v2_source')
CODE_ROOT.mkdir(parents=True,exist_ok=True)
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
    for relative,digest in EXPECTED_SOURCE.items():
        contents=(manifests[0].parent/relative).read_bytes()
        assert hashlib.sha256(contents).hexdigest()==digest, 'Source hash mismatch: '+relative
        dest=CODE_ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
        archive.writestr(relative,contents)
raw=stream.getvalue()
'''
nb['cells'][2]['source']=(prefix+tail).splitlines(keepends=True)
expected=recovery['files']
nb['cells'][3]['source']=f'''# Exact recovered data from own saved course cache and official split exports.
EXPECTED_RECOVERED={expected!r}
matches=[]
for p in Path('/kaggle/input').rglob('RECOVERY_MANIFEST.json'):
    try:
        if json.loads(p.read_text()).get('files')==EXPECTED_RECOVERED:matches.append(p)
    except (ValueError,OSError):pass
assert len(matches)==1, 'Attach the private recovered course data dataset.'
recovery=json.loads(matches[0].read_text())
DATA_ROOT=WORK/'official_data'
for relative,metadata in EXPECTED_RECOVERED.items():
    source=matches[0].parent/relative
    assert source.stat().st_size==metadata['bytes'] and sha(source)==metadata['sha256'], 'Data hash mismatch: '+relative
    dest=DATA_ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(source)
atomic_json(PERSIST/'OFFICIAL_DATA_PROVENANCE.json',recovery)
from phase4_cjpe.data import index_data
db,DATA_AUDIT=index_data(DATA_ROOT,WORK/'shared_official_index',progress_root=PERSIST/'data_preflight')
assert {{k:v['cases'] for k,v in DATA_AUDIT['counts'].items()}}=={{'train':32305,'dev':994,'test':1517}}
atomic_json(PERSIST/'DATA_AUDIT.json',DATA_AUDIT)
print('Recovered data passed exact hashes and cross-split checks:',DATA_AUDIT['counts'])
'''.splitlines(keepends=True)
nb['cells'][0]['source']='''# CS781 Phase 4: reviewed runner, recovered own course data

Attach both private datasets: original model inputs and recovered course data plus runner. Select GPU T4 x2. Six actual adaptations run two at a time, with exact checkpoint resume, train-only model selection and verified prediction ZIP exports. Data were recovered from own saved course cache using the existing official dev/test split IDs; original JSON formatting, duplicate copies and expert fields are absent. Test labels are absent.

Run All for a GPU preflight and actual progress check. Save & Run All can run a background Kaggle version; avoid a simultaneous interactive job. Per-session service limits still apply. All experiments are truthfully after the original deadline. Output is under /kaggle/working/CS781_Phase4_EvidenceV2_20261022.
'''.splitlines(keepends=True)
for cell in nb['cells']:
 if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
out=ROOT/'CS781_Phase4_Kaggle_Run.ipynb'
out.write_text(json.dumps(nb,indent=1),encoding='utf-8')
archive=ROOT/'CS781_Phase4_Recovered_Data_Runner_20261022.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for relative in (*expected,'RECOVERY_MANIFEST.json'):z.write(DATA/relative,relative)
 z.write(ROOT/'SOURCE_MANIFEST.json','runner_source/SOURCE_MANIFEST.json')
 for relative in hashes:z.write(ROOT/'source'/relative,'runner_source/'+relative)
print(json.dumps({'notebook':str(out),'notebook_bytes':out.stat().st_size,'dataset_zip':str(archive),'zip_bytes':archive.stat().st_size,'source_files':len(hashes)}),flush=True)
