"""Two serial LoRMA NER adaptations from frozen 100M/200M LM checkpoints."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legal_d
from legal_d.common import source_hashes
from legal_d.ner_train import prepare_ner, train_candidate
from lner.data import atomic_json, sha256, validate_predictions


def main(path):
    original = json.loads(Path(path).read_text())
    base = Path(original['output_root'])
    lock = (base / 'milestones/ner_queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Milestone NER must use physical GPU1')
    for million in (100, 200):
        milestone_path = base / 'milestones' / f'{million}M.json'
        if not milestone_path.exists():
            continue
        milestone = json.loads(milestone_path.read_text())
        if milestone['phase'] != 'captured':
            continue
        checkpoint = milestone['checkpoint']
        if sha256(checkpoint) != milestone['sha256'] or source_hashes() != milestone['original_fingerprint']['source_hashes']:
            raise ValueError('Frozen milestone provenance differs')
        config = copy.deepcopy(original)
        run = base / 'milestones' / f'ner_{million}M'
        run.mkdir(exist_ok=True)
        child_config = run / 'config.json'
        if child_config.exists():
            config = json.loads(child_config.read_text())
        else:
            # Preserve official deadline separately; never label late training as on-time.
            config['official_phase2_deadline_utc'] = original['pretrain']['deadline_utc']
            config['pretrain']['deadline_utc'] = datetime.fromtimestamp(time.time() + 3 * 3600, timezone.utc).isoformat()
            config['output_root'] = str(run)
            config['milestone'] = {'requested_tokens': million * 1_000_000, 'actual_tokens': milestone['actual_target_tokens'], 'sha256': milestone['sha256']}
            atomic_json(child_config, config)
        dataset, windows = prepare_ner(config)
        candidate = dict(next(c for c in config['ner']['candidates'] if c['method'] == 'lorma'))
        candidate['name'] = f'lorma_{million}M'
        atomic_json(base / 'milestones/ner_queue_status.json', {'phase': 'ner_training', 'milestone': million, 'actual_tokens': milestone['actual_target_tokens']})
        result = train_candidate(config, candidate, checkpoint, dataset, windows, export=True, train_until=datetime.fromisoformat(config['pretrain']['deadline_utc']).timestamp() - 3600)
        source = Path(result['submission'])
        destination = base / 'milestones/deliverables' / f'UPLOAD_Phase2_D222_{million}M_LORMA.zip'
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
                raise ValueError('Invalid milestone archive')
            prediction = json.loads(archive.read('prediction.json'))
            validate_predictions(prediction, dataset['test'], dataset['labels'])
            if len(prediction) != 35:
                raise ValueError('Official ID count differs')
        shutil.copy2(source, destination.with_suffix('.partial'))
        destination.with_suffix('.partial').replace(destination)
        if sha256(destination) != result['sha256']:
            raise ValueError('Milestone export SHA mismatch')
        record = {**result, 'submission': str(destination), 'requested_lm_tokens': million * 1_000_000,
                  'actual_lm_tokens': milestone['actual_target_tokens'], 'official_phase2_deadline_utc': original['pretrain']['deadline_utc'],
                  'ready_before_official_deadline': time.time() < datetime.fromisoformat(original['pretrain']['deadline_utc']).timestamp(),
                  'finished_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(base / 'milestones' / f'{million}M_phase2_complete.json', record)
        print(json.dumps({k: v for k, v in record.items() if k != 'fingerprint'}), flush=True)
    atomic_json(base / 'milestones/ner_queue_status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)
