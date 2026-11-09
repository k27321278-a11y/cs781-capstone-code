"""Persistent records and full-source fingerprints."""
import hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from phase3_rr.common import sha,atomic_json,save,seed_all,hardware

def now(): return datetime.now(timezone.utc).isoformat()
def source_hashes():
 root=Path(__file__).resolve().parents[1]
 return {p.relative_to(root).as_posix():sha(p) for name in ('phase4_cjpe','phase3_rr') for p in sorted((root/name).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and (p.suffix=='.py' or p.name in ('UnicodeData-13.0.0.txt','UnicodeData-14.0.0.txt'))}
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def status(root,phase,**values):
 root=Path(root);old={}
 try: old=json.loads((root/'status.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError): pass
 obj={**old,**values,'phase':phase,'updated_utc':now()}
 atomic_json(root/'status.json',obj)
 if values.get('step') is not None or phase!=old.get('phase'): print(json.dumps(obj,ensure_ascii=False),flush=True)
 return obj

