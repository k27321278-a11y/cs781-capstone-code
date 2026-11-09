"""Verify and download the improved LegalSpan prediction as soon as complete."""
import json
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from lner.data import sha256, atomic_json, validate_predictions

config = json.loads((ROOT / 'config.json').read_text())
texts = json.loads((ROOT / 'data/Phase_2_LNER/Test/fold3-text.json').read_text())
labels = (ROOT / 'data/Phase_2_LNER/Train/labels.txt').read_text().split()
out = ROOT.parent / 'outputs/phase2_20261011'
print('LEGALSPAN_ROLE_SPAN_DOWNLOADER_STARTED', flush=True)
while True:
    try:
        code = 'import json,pathlib; p=pathlib.Path(' + repr(config['output_root']) + ')/"complete.json"; print(p.read_text() if p.exists() else "null")'
        remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'stud1', 'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=30)
        if remote.returncode:
            raise ConnectionError(remote.stderr.strip())
        record = json.loads(remote.stdout)
        if record:
            destination = out / Path(record['submission']).name
            partial = destination.with_suffix('.partial')
            subprocess.run(['scp', '-B', '-q', 'stud1:' + record['submission'], str(partial)], check=True, timeout=120)
            if sha256(partial) != record['sha256']:
                raise ValueError('Boost ZIP download SHA differs')
            with zipfile.ZipFile(partial) as archive:
                if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
                    raise ValueError('Boost ZIP contract differs')
                predictions = json.loads(archive.read('prediction.json'))
                validate_predictions(predictions, texts, labels)
                if len(predictions) != 35:
                    raise ValueError('Boost official IDs differ')
            partial.replace(destination)
            downloads = Path.home() / 'Downloads' / destination.name
            shutil.copy2(destination, downloads)
            if sha256(downloads) != record['sha256']:
                raise ValueError('Windows Downloads SHA differs')
            record.update({'local_path': str(destination), 'downloads_path': str(downloads)})
            atomic_json(out / 'legalspan_role_span_downloaded.json', record)
            print('DOWNLOADED_AND_VERIFIED ' + destination.name + ' ' + record['sha256'], flush=True)
            break
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), flush=True)
    time.sleep(60)
