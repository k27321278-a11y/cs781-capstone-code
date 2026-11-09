import datetime
import hashlib
import json
import pathlib
import sys
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = pathlib.Path(sys.argv[1])
ready = pathlib.Path(sys.argv[2])
outputs = root / 'outputs'
models = {
    'v7_transformer': ('v7', 'B1_Window_TransformerOnly'),
    'v6_transformer': ('v6', 'B1_Window_TransformerOnly'),
    'v5_gru': ('v5', 'B1_Window_GRUOnly'),
    'v4_reference': ('v4', 'B1_FullData_Reference'),
}
assert (root / 'VERIFIED_HEADS_V7_SEGMENT3_BACKUPS.json').exists()
assert (root / 'VERIFIED_HEADS_V7_SEGMENT3_EXPORTS.json').exists()


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_model(version, name):
    base = outputs / version / 'CS781_B1_Hierarchy_Depth' / 'runs' / name / 'exports'
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith('segment_')]
    exp = max(dirs, key=lambda p: int(p.name.split('_')[-1]))
    data = {}
    hashes = {}
    for split in ('calibration', 'dev', 'test'):
        path = exp / f'{split}_probabilities.npz'
        hashes[split] = {'path': str(path), 'sha256': sha256(path)}
        with np.load(path, allow_pickle=False) as arr:
            ids = arr['ids'].tolist()
            logits = arr['logits']
            assert logits.shape == (len(ids), 2)
            assert len(ids) == len(set(ids))
            margin = (logits[:, 1] - logits[:, 0]).astype(np.float64)
            assert np.isfinite(margin).all()
            entry = {'ids': ids, 'margin': margin}
            if split == 'calibration':
                entry['labels'] = arr['labels'].astype(int)
            elif split == 'dev':
                rows = json.loads((exp / 'dev_truth.json').read_text(encoding='utf-8'))
                gold = {r['id']: int(r['gold']) for r in rows}
                assert len(gold) == len(ids)
                entry['labels'] = np.array([gold[i] for i in ids], dtype=int)
            data[split] = entry
    return exp, data, hashes

loaded = {key: load_model(*spec) for key, spec in models.items()}
order = list(models)
base = loaded[order[0]][1]
with zipfile.ZipFile(ready / 'B1_Transformer_v7.zip') as archive:
    official_ids = [r['id'] for r in json.loads(archive.read('prediction.json'))]
assert len(official_ids) == len(set(official_ids)) == 1517


def matrix(split):
    ids = base[split]['ids']
    assert len(ids) == (1517 if split == 'test' else 512 if split == 'calibration' else 994)
    columns = []
    labels = base[split].get('labels')
    for key in order:
        entry = loaded[key][1][split]
        positions = {identifier: i for i, identifier in enumerate(entry['ids'])}
        assert set(positions) == set(ids)
        idx = [positions[identifier] for identifier in ids]
        columns.append(entry['margin'][idx])
        if labels is not None:
            assert np.array_equal(entry['labels'][idx], labels)
    return ids, np.column_stack(columns), labels

cal_ids, x_cal, y_cal = matrix('calibration')
dev_ids, x_dev, y_dev = matrix('dev')
test_ids, x_test, _ = matrix('test')
assert set(test_ids) == set(official_ids)
assert np.unique(y_cal, return_counts=True)[1].tolist() == [256, 256]
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=0))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=781)
cv_probability = cross_val_predict(model, x_cal, y_cal, cv=folds, method='predict_proba')[:, 1]
cv_f1 = float(f1_score(y_cal, cv_probability >= 0.5, average='macro'))
model.fit(x_cal, y_cal)
dev_probability = model.predict_proba(x_dev)[:, 1]
dev_f1 = float(f1_score(y_dev, dev_probability >= 0.5, average='macro'))
test_probability = model.predict_proba(x_test)[:, 1]
lookup = {identifier: int(prob >= 0.5) for identifier, prob in zip(test_ids, test_probability)}
rows = [{'id': identifier, 'pred': lookup[identifier]} for identifier in official_ids]
assert len(rows) == 1517 and all(type(row['pred']) is int and row['pred'] in (0, 1) for row in rows)

name = 'B1_Stack_v7v6v5v4.zip'
assert len(name) <= 64
zip_path = ready / name
if zip_path.exists():
    raise RuntimeError('Refusing to overwrite existing submission ZIP')
encoded = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('prediction.json', encoded)
with zipfile.ZipFile(zip_path) as archive:
    assert archive.namelist() == ['prediction.json'] and archive.testzip() is None
    restored = json.loads(archive.read('prediction.json'))
    assert restored == rows

report = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'kind': 'separate_explicitly_requested_ensemble_not_single_model_B1',
    'selection_rule': 'Four inputs fixed from prior evidence; L2 logistic stack C=0.1 fit on 512 disjoint train calibration examples; five-fold calibration CV; threshold 0.5. Official dev used only to report.',
    'models': order,
    'input_hashes': {key: loaded[key][2] for key in order},
    'calibration_cases': len(cal_ids),
    'calibration_cv_macro_f1': cv_f1,
    'official_dev_cases': len(dev_ids),
    'official_dev_macro_f1': dev_f1,
    'test_cases': len(rows),
    'test_positive_predictions': sum(row['pred'] for row in rows),
    'coefficients': model[-1].coef_.tolist(),
    'intercept': model[-1].intercept_.tolist(),
    'zip_path': str(zip_path),
    'zip_sha256': sha256(zip_path),
    'zip_bytes': zip_path.stat().st_size,
    'zip_crc_verified': True,
    'exact_official_ids_verified': True,
    'hidden_codabench_score': None,
}
report_path = root / 'VERIFIED_B1_STACK_V7V6V5V4.json'
if report_path.exists():
    raise RuntimeError('Refusing to overwrite existing report')
report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'zip': str(zip_path), 'dev_f1': dev_f1, 'calibration_cv_f1': cv_f1, 'sha256': report['zip_sha256'], 'bytes': report['zip_bytes']}))
