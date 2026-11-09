"""Exact two-of-four span agreement; fixed validation-selected rule, no retraining."""
import json
import sys
import zipfile
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'phase2'))
from lner.data import sha256, export_zip, validate_predictions, strict_score, atomic_json


def consensus(models):
    ids = set(models[0])
    if any(set(model) != ids for model in models):
        raise ValueError('Ensemble source IDs differ')
    result = {}
    for key in sorted(ids):
        votes = Counter(span for model in models for span in {tuple(s[:3]) for s in model[key]})
        proposed = sorted((s for s, n in votes.items() if n >= 2), key=lambda s: (-votes[s], s[1]-s[0], s))
        accepted = []
        for span in proposed:
            if not any(max(span[0], other[0]) < min(span[1], other[1]) for other in accepted):
                accepted.append(span)
        result[key] = [list(s) for s in sorted(accepted)]
    return result


def main():
    folder = ROOT / 'outputs/phase2_20261011'
    inventory = json.loads((folder / 'download_inventory.json').read_text())
    names = ('01_legalspan_lorma', '02_b1_lora', '03_c1_full', '04_a0_distilled')
    validation, test, provenance = [], [], []
    truth = None
    for name in names:
        record = inventory[name]
        submission = Path(record['local_path'])
        architecture = folder / f'CS781_Phase2_{name}_Complete.zip'
        if not submission.exists():
            import hashlib
            with zipfile.ZipFile(architecture) as saved:
                content = saved.read('submission/' + submission.name)
            if hashlib.sha256(content).hexdigest() != record['sha256']:
                raise ValueError('Archived original submission hash differs')
            submission.write_bytes(content)
        if sha256(submission) != record['sha256']:
            raise ValueError('Original prediction ZIP hash differs')
        with zipfile.ZipFile(submission) as archive:
            if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
                raise ValueError('Original ZIP contract differs')
            test.append(json.loads(archive.read('prediction.json')))
        with zipfile.ZipFile(architecture) as archive:
            gold = json.loads(archive.read('run/validation_truth.json'))
            if truth is not None and gold != truth:
                raise ValueError('Held-out truth differs')
            truth = gold
            validation.append(json.loads(archive.read('run/validation_predictions.json')))
        provenance.append({'model': name, 'prediction_sha256': record['sha256'], 'checkpoint_sha256': record['best_checkpoint_sha256']})
    labels = (ROOT / 'phase2/data/Phase_2_LNER/Train/labels.txt').read_text().split()
    texts = json.loads((ROOT / 'phase2/data/Phase_2_LNER/Test/fold3-text.json').read_text())
    predicted = consensus(test)
    validate_predictions(predicted, texts, labels)
    if len(predicted) != 35:
        raise ValueError('Official ID count differs')
    archive = folder / 'UPLOAD_Phase2_05_Consensus_2of4.zip'
    existing_download = Path.home() / 'Downloads' / archive.name
    if existing_download.exists():
        with zipfile.ZipFile(existing_download) as saved:
            if saved.namelist() != ['prediction.json'] or saved.testzip() is not None or json.loads(saved.read('prediction.json')) != predicted:
                raise ValueError('Existing Downloads ensemble differs; preserve it for inspection')
        shutil.copy2(existing_download, archive)
        digest = sha256(archive)
    else:
        digest = export_zip(archive, predicted, texts, labels)
    validation_prediction = consensus(validation)
    metrics = strict_score(truth, validation_prediction, labels)
    atomic_json(folder / 'consensus_validation_predictions.json', validation_prediction)
    record = {'phase': 'complete', 'method': 'two_of_four_exact_span_agreement', 'sha256': digest,
              'submission': str(archive), 'official_ids': len(predicted), 'entity_spans': sum(map(len, predicted.values())),
              'validation': metrics, 'rule_selected_on_validation': True, 'source_models': provenance,
              'training_required': False, 'codabench_score': None}
    atomic_json(folder / 'consensus_complete.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'validation'}, indent=2))
    print('VALIDATION_STRICT_MACRO_F1', metrics['macro_f1'])


if __name__ == '__main__':
    main()
