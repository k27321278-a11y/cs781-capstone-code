import json
import sqlite3
from pathlib import Path

import numpy as np

from phase4_cjpe.common import atomic_json, digest, sha, status
from phase4_cjpe.data import index_data, chunks as reference_chunks
from phase3_rr.data import Tokenizer


def split_ids(db, fixed_holdout, seed=781):
    """Keep prior validation IDs and group equal normalized training texts."""
    with sqlite3.connect(db) as con:
        rows = con.execute('SELECT id,label,text_sha FROM cases WHERE split="train" ORDER BY id').fetchall()
        other = {s: [r[0] for r in con.execute('SELECT id FROM cases WHERE split=? ORDER BY id', (s,))]
                 for s in ('dev', 'test')}
    by_id = {r[0]: r for r in rows}
    holdout = set(fixed_holdout)
    if len(holdout) != 512 or not holdout <= by_id.keys():
        raise ValueError('Expected the original 512 validation IDs')
    groups = {}
    for key, label, text_hash in rows:
        groups.setdefault(text_hash or ('empty:' + key), []).append((key, label))
    if any(len({label for _, label in group}) != 1 for group in groups.values()):
        raise ValueError('Identical normalized training text has conflicting labels')
    reserved_hashes = {by_id[k][2] or ('empty:' + k) for k in holdout}
    rng = np.random.default_rng(seed + 17)
    calibration = []
    calibration_hashes = set()
    for label in (0, 1):
        candidates = sorted(h for h, g in groups.items() if h not in reserved_hashes and g[0][1] == label)
        if len(candidates) < 256:
            raise ValueError('Insufficient disjoint calibration groups')
        for i in rng.permutation(len(candidates))[:256]:
            h = candidates[int(i)]
            calibration.append(groups[h][0][0])
            calibration_hashes.add(h)
    excluded = reserved_hashes | calibration_hashes
    train = sorted(g[0][0] for h, g in groups.items() if h not in excluded)
    result = dict(train=train, holdout=sorted(holdout), calibration=sorted(calibration), **other)
    hashes = {s: {by_id[k][2] or ('empty:' + k) for k in result[s]}
              for s in ('train', 'holdout', 'calibration')}
    for a, b in (('train', 'holdout'), ('train', 'calibration'), ('holdout', 'calibration')):
        if set(result[a]) & set(result[b]) or hashes[a] & hashes[b]:
            raise ValueError('Training/validation/calibration leakage')
    return result, {'normalized_training_groups': len(groups), 'official_training_ids': len(rows),
                    'training_ids_used': len(train), 'validation_ids_preserved': True,
                    'calibration_counts': [256, 256], 'group_disjoint': True,
                    'excluded_duplicate_training_ids': len(rows) - len(train) - 1024}


def window_tokens(tokens, max_tokens=512, count=16, overlap=100, policy='tail'):
    budget = max_tokens - 2
    if budget < 16 or not 0 <= overlap < budget or count < 2:
        raise ValueError('Invalid window dimensions')
    n = len(tokens)
    if not n:
        return [[1, 3, 2]], [0.], {'full_tokens': 0, 'unique_tokens': 0, 'retained_tokens': 0,
                                   'coverage_fraction': 1., 'windows': 1}
    stride = budget - overlap
    starts = list(range(0, max(1, n - budget + 1), stride))
    starts = sorted(set(starts + [max(0, n - budget)]))
    if len(starts) > count:
        positions = np.arange(len(starts))
        # Weighted quantiles allocate more windows to the final third; always retain both ends.
        if policy == 'tail':
            weights = np.where(positions >= len(starts) * 2 / 3, 2., 1.)
            cumulative = np.cumsum(weights)
            chosen = np.searchsorted(cumulative, np.linspace(cumulative[0], cumulative[-1], count))
        elif policy == 'uniform':
            chosen = np.rint(np.linspace(0, len(starts) - 1, count)).astype(int)
        else:
            raise ValueError('Unknown window selection policy')
        selected = set(map(int, chosen)) | {0, len(starts) - 1}
        for i in np.rint(np.linspace(0, len(starts) - 1, len(starts))).astype(int):
            if len(selected) >= count:
                break
            selected.add(int(i))
        starts = [starts[i] for i in sorted(selected)]
    intervals = [(s, min(n, s + budget)) for s in starts]
    covered = 0
    previous_end = 0
    for start, end in intervals:
        covered += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    sequences = [[1] + tokens[a:b] + [2] for a, b in intervals]
    positions = [a / max(1, n - 1) for a, _ in intervals]
    assert len(set(starts)) == len(starts) <= count and starts[0] == 0
    assert intervals[-1][1] == n and 0 <= covered <= n
    return sequences, positions, {'full_tokens': n, 'unique_tokens': covered,
                                 'retained_tokens': sum(b - a for a, b in intervals),
                                 'coverage_fraction': covered / n, 'windows': len(sequences)}


class Store:
    def __init__(self, path, ids):
        self.path = str(path)
        self.ids = ids
        self.con = sqlite3.connect(path)

    def row(self, split, key):
        value = self.con.execute('SELECT payload FROM tokens WHERE split=? AND id=?', (split, key)).fetchone()
        if value is None:
            raise KeyError((split, key))
        return json.loads(value[0])

    def rows(self, split):
        for key in self.ids[split]:
            yield self.row(split, key)

    def close(self):
        self.con.close()


def prepare_data(config, bundle):
    root = Path(config['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    work = Path(config['work_root'])
    db, audit = index_data(config['data_root'], work / 'official_index', progress_root=root)
    fixed = json.loads(Path(config['fixed_holdout']).read_text())
    ids, split_audit = split_ids(db, fixed, config['seed'])
    fp = {'data': audit['files'], 'ids': digest(ids), 'tokenizer': sha(Path(bundle) / 'tokenizer.json'),
          'mode': config['representation'], 'max_tokens': config['max_tokens'], 'windows': config['chunks_per_case'],
          'overlap': config['overlap'], 'window_policy': config['window_policy'], 'source': sha(__file__)}
    cache = work / ('tokens_' + digest(fp)[:18])
    cache.mkdir(parents=True, exist_ok=True)
    dest, marker = cache / 'tokens.sqlite', cache / 'complete.json'
    if dest.exists() and marker.exists():
        previous = json.loads(marker.read_text())
        if previous['fingerprint'] != fp:
            raise ValueError('Cache fingerprint mismatch')
        coverage = previous['coverage']
    else:
        tok = Tokenizer(Path(bundle) / 'tokenizer.json')
        temp = cache / 'tokens.building.sqlite'
        if temp.exists():
            temp.unlink()  # Only this generated incomplete cache.
        con = sqlite3.connect(db)
        out = sqlite3.connect(temp)
        out.execute('CREATE TABLE tokens(split TEXT,id TEXT,payload TEXT,PRIMARY KEY(split,id))')
        coverage = {}
        try:
            for split, keys in ids.items():
                stats = dict(cases=0, full_tokens=0, unique_tokens=0, retained_tokens=0, windows=0)
                origin = 'train' if split in ('holdout', 'calibration') else split
                for i, key in enumerate(keys):
                    text, label = con.execute('SELECT text,label FROM cases WHERE split=? AND id=?', (origin, key)).fetchone()
                    if config['representation'] == 'reference':
                        sequences, positions, _ = reference_chunks(text, tok, 256, 8, 8192)
                        # Full token coverage is intentionally not inferred for the legacy region sampler.
                        c = dict(full_tokens=0, unique_tokens=0, retained_tokens=sum(len(x) - 2 for x in sequences), windows=len(sequences))
                    else:
                        sequences, positions, c = window_tokens(tok.encode(text), config['max_tokens'],
                            config['chunks_per_case'], config['overlap'], config['window_policy'])
                    row = dict(id=key, label=label, chunks=sequences, positions=positions, coverage=c)
                    out.execute('INSERT INTO tokens VALUES(?,?,?)', (split, key, json.dumps(row)))
                    stats['cases'] += 1
                    for k in ('full_tokens', 'unique_tokens', 'retained_tokens', 'windows'):
                        stats[k] += c[k]
                    if (i + 1) % 100 == 0:
                        out.commit()
                        status(root, 'tokenizing', split=split, cases=i + 1, total=len(keys))
                out.commit()
                stats['coverage_fraction'] = stats['unique_tokens'] / max(1, stats['full_tokens']) if config['representation'] != 'reference' else None
                coverage[split] = stats
            assert out.execute('PRAGMA quick_check').fetchone() == ('ok',)
        finally:
            out.close()
            con.close()
        temp.replace(dest)
        atomic_json(marker, dict(fingerprint=fp, coverage=coverage))
    atomic_json(root / 'splits.json', ids)
    atomic_json(root / 'data_audit.json', dict(official=audit, split_audit=split_audit))
    atomic_json(root / 'token_cache.json', dict(fingerprint=fp, coverage=coverage, path=str(dest)))
    return Store(dest, ids), dict(data=audit['files'], split_ids=digest(ids), token_cache=fp,
                                 split_audit=split_audit, coverage=coverage)
