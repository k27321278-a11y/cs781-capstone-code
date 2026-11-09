"""Small upload archives containing exact original checkpoint members, not retrained weights."""
import json,shutil,zipfile
from pathlib import Path,PurePosixPath
from .common import sha,atomic_json

def resolve_bundle(path,cache):
 path=Path(path);cache=Path(cache)
 if path.is_dir():root=path
 else:
  root=cache/path.stem;root.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(path) as z:
   manifest=json.loads(z.read('INPUT_MANIFEST.json'))
   for info in z.infolist():
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive member')
    if info.filename not in ('checkpoint.pt','tokenizer.json','INPUT_MANIFEST.json'):raise ValueError('Unexpected input bundle member')
    dest=root/info.filename
    expected=manifest['sha256'].get(info.filename)
    if info.filename=='INPUT_MANIFEST.json' or not dest.exists() or dest.stat().st_size!=info.file_size or (expected and sha(dest)!=expected):
     tmp=dest.with_suffix(dest.suffix+'.partial')
     with z.open(info) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
     if expected and sha(tmp)!=expected:raise ValueError('Corrupted input archive member: '+info.filename)
     tmp.replace(dest)
 manifest=json.loads((root/'INPUT_MANIFEST.json').read_text())
 for name,digest in manifest['sha256'].items():
  if sha(root/name)!=digest:raise ValueError('Input SHA mismatch: '+name)
 return root,manifest
