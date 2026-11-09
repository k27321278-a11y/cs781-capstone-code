"""Train-only mixed windows with an own-tokenizer cache and immutable split IDs."""
import hashlib
import json
import os
import random
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from cs781.io import sha256_file, write_json_atomic
from cs781.splits import iter_catalog, catalog_metadata
from cs781.contracts import SOURCES
from .bpe import OwnBPE, Encoding, pieces, char_class

CUES = {'section', 'sections', 'sec', 'article', 'articles', 'act', 'rule', 'rules',
        'clause', 'clauses', 'subsection', 'scc', 'air', 'crpc', 'ipc', 'cpc'}


def role_ids(text, encoding):
    """Precedence: digit > statutory cue > capitalized name-like > structure.

    Each byte-BPE token stays inside one class piece. These are prompt heuristics,
    never gold entities; capitals include headings and can be false positives.
    """
    spans = []
    for p, (a, b) in pieces(text, errors='surrogateescape'):
        kind = char_class(p[0])
        role = 0 if any(0xdc80 <= ord(c) <= 0xdcff for c in p) else 1 if kind == 2 else 2 if p.lower() in CUES else 3 if kind == 1 and p[0].isupper() else 4 if kind == 3 or '\n' in p else 0
        spans.append((a, b, role))
    result, cursor = [], 0
    for a, b in encoding.offsets:
        while cursor < len(spans) and a >= spans[cursor][1]:
            cursor += 1
        if cursor >= len(spans) or b > spans[cursor][1]:
            raise ValueError('Token byte offsets violate reversible segmentation')
        result.append(spans[cursor][2])
    return result


def observed_role_ids(tokenizer, ids):
    # A token cut can bisect a word or UTF8 character. Decode only the observed
    # bytes; surrogateescape keeps their true offsets and marks invalid fragments
    # ordinary. Never consult the full document to complete a cue or word.
    raw = b''.join(tokenizer.bytes[i] for i in ids)
    offsets, position = [], 0
    for i in ids:
        end = position + len(tokenizer.bytes[i]); offsets.append((position, end)); position = end
    return role_ids(raw.decode('utf-8', errors='surrogateescape'), Encoding(ids, offsets))


def prepare(catalog, parent_data, tokenizer_path, output, max_seq_len=8192, seed=781):
    catalog, parent_data, tokenizer_path, output = map(Path, (catalog, parent_data, tokenizer_path, output))
    if output.exists():
        raise FileExistsError(output)
    tok = OwnBPE.load(tokenizer_path)
    meta = json.loads(tokenizer_path.with_suffix('.meta.json').read_text())
    catalog_hash = sha256_file(catalog)
    train_ids = {d.raw.id: d.raw.fingerprint for d in iter_catalog(catalog, 'train')}
    if not meta.get('complete') or meta['catalog_sha256'] != catalog_hash or meta['sha256'] != sha256_file(tokenizer_path) or not meta['training_inputs'] or any(train_ids.get(d['id']) != d['fingerprint'] for d in meta['training_inputs']):
        raise ValueError('Own tokenizer train-only provenance mismatch')
    stage = output.with_name(output.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    shutil.copyfile(tokenizer_path, stage / 'tokenizer.json')
    shutil.copyfile(tokenizer_path.with_suffix('.meta.json'), stage / 'tokenizer.meta.json')
    counts, sources, lengths, kinds = {}, {}, [], Counter()
    for split in ('train', 'dev', 'confirmation'):
        offsets, kind_offsets, source_count = [], {'word': [], 'token': []}, Counter()
        if split != 'train':
            prompts = json.loads((parent_data / f'{split}.prompts.json').read_text())
            refs = json.loads((parent_data / f'{split}.references.json').read_text())
            if set(prompts) != set(refs):
                raise ValueError('Held-out prompt/reference ID mismatch')
            docs = {d.raw.id: d for d in iter_catalog(catalog, split)}
            if not set(prompts).issubset(docs):
                raise ValueError('Held-out IDs do not belong to their catalog split')
            for suffix in ('prompts', 'references'):
                shutil.copyfile(parent_data / f'{split}.{suffix}.json', stage / f'{split}.{suffix}.json')
        with (stage / f'{split}.jsonl').open('wb') as f:
            def emit(doc, prefix, suffix, marks, kind, offset):
                length = len(prefix) + len(suffix) + 1
                if not prefix or not suffix or length > max_seq_len:
                    raise ValueError(f'{split} example exceeds full context: {doc.raw.id}, {length}; no truncation')
                row = {'id': doc.raw.id if split != 'train' else f'{doc.raw.id}:{kind}:{offset}',
                       'document_id': doc.raw.id, 'source': doc.raw.source, 'split': split,
                       'prefix_ids': prefix, 'suffix_ids': suffix, 'prompt_roles': marks,
                       'kind': kind, 'offset': offset}
                position = f.tell(); offsets.append(position); kind_offsets[kind].append(position)
                f.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
                source_count[doc.raw.source] += 1; kinds[kind] += 1; lengths.append(length)
            if split == 'train':
                for di, doc in enumerate(iter_catalog(catalog, split)):
                    words = doc.raw.text.split()
                    for start in range(0, max(0, len(words) - 640 + 1), 512):
                        text = ' '.join(words[start:start + 512]); target = ' ' + ' '.join(words[start+512:start+640])
                        p, s = tok.encode(text), tok.encode(target)
                        emit(doc, p.ids, s.ids, role_ids(text, p), 'word', start)
                    encoded = tok.encode(doc.raw.text); marks = role_ids(doc.raw.text, encoded)
                    # Nonoverlapping conditional chunks. One quarter of documents
                    # use long capacity; others use the original 2047-token chunk.
                    digest = int(hashlib.sha256(f'{seed}:{doc.raw.id}'.encode()).hexdigest()[:16], 16)
                    chunk_size = max_seq_len - 1 if digest % 4 == 0 else min(2047, max_seq_len - 1)
                    for start in range(0, len(encoded.ids), chunk_size):
                        chunk = encoded.ids[start:start + chunk_size]
                        if len(chunk) < 2: continue
                        rng = random.Random(int(hashlib.sha256(f'{seed}:{doc.raw.id}:{start}'.encode()).hexdigest()[:16], 16))
                        cut = rng.randint(max(1, len(chunk)//4), min(len(chunk)-1, 3*len(chunk)//4))
                        emit(doc, chunk[:cut], chunk[cut:], observed_role_ids(tok, chunk[:cut]), 'token', start)
                    if di % 1000 == 0: print(json.dumps({'phase': 'prepare', 'documents': di, 'examples': len(offsets)}), flush=True)
            else:
                for key in sorted(prompts):
                    text, suffix = prompts[key]['text'], refs[key]['text']
                    if len(text.split()) != 512 or len(suffix.split()) != 128:
                        raise ValueError('Held-out word contract differs')
                    p, s = tok.encode(text), tok.encode(' ' + suffix.strip())
                    emit(docs[key], p.ids, s.ids, role_ids(text, p), 'word', 0)
            f.flush(); os.fsync(f.fileno())
        if not offsets or set(source_count) != set(SOURCES):
            raise ValueError('Prepared split lacks examples or an official source')
        counts[split], sources[split] = len(offsets), dict(source_count)
        np.save(stage / f'{split}.offsets.npy', np.asarray(offsets, dtype=np.int64))
        if split == 'train':
            for kind, off in kind_offsets.items():
                if not off: raise ValueError('Mixed training stream lacks a window type')
                np.save(stage / f'train.{kind}.offsets.npy', np.asarray(off, dtype=np.int64))
    files = {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in stage.iterdir()}
    corpus = catalog_metadata(catalog)
    manifest = {'format': 'own-legalspan-data-v1', 'complete': True, 'files': files, 'counts': counts,
                'source_counts': sources, 'kind_counts': dict(kinds), 'max_seq_len': max_seq_len,
                'max_example_tokens': max(lengths), 'seed': seed, 'catalog_sha256': catalog_hash,
                'catalog': corpus, 'synthetic': corpus.get('corpus_config') is None,
                'sampling': 'alternate equal example counts: exact 512/128 word and nonoverlap random-prefix token windows',
                'roles': 'digit > statutory cue > capitalized name-like > structural punctuation/newline > ordinary',
                'roles_scope': 'observed_prefix_bytes_only',
                'heldout_parent_hashes': {p.name: sha256_file(p) for p in parent_data.glob('*.prompts.json')} }
    write_json_atomic(stage / 'manifest.json', manifest)
    os.rename(stage, output)
    return load_data(output)


def load_data(root):
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text())
    if m.get('format') != 'own-legalspan-data-v1' or m.get('complete') is not True or m.get('roles_scope') != 'observed_prefix_bytes_only':
        raise ValueError('Incomplete own data manifest')
    for name, expected in m['files'].items():
        p = root / name
        if p.stat().st_size != expected['bytes'] or sha256_file(p) != expected['sha256']:
            raise ValueError('Prepared file hash mismatch')
    return m


def rebuild_roles(parent, output):
    """Versioned deterministic repair: preserve every example/token/split.

    Legacy source files are verified before reading. It is retained as evidence;
    output receives fresh hashes and offsets and requires observed-prefix roles.
    """
    parent, output = Path(parent), Path(output)
    if output.exists(): raise FileExistsError(output)
    manifest = json.loads((parent/'manifest.json').read_text())
    if manifest.get('complete') is not True or manifest.get('format') != 'own-legalspan-data-v1':
        raise ValueError('Invalid parent cache')
    for name, record in manifest['files'].items():
        if sha256_file(parent/name) != record['sha256']: raise ValueError('Parent file hash mismatch')
    stage = output.with_name(output.name+'.staging-'+uuid.uuid4().hex); stage.mkdir(parents=True)
    for name in manifest['files']:
        if not name.endswith('.jsonl') and not name.endswith('.npy'): shutil.copyfile(parent/name, stage/name)
    tok = OwnBPE.load(parent/'tokenizer.json')
    for split in ('train', 'dev', 'confirmation'):
        offsets, kind_offsets = [], {'word': [], 'token': []}
        with (parent/f'{split}.jsonl').open() as src, (stage/f'{split}.jsonl').open('wb') as dst:
            for i, line in enumerate(src):
                row = json.loads(line); row['prompt_roles'] = observed_role_ids(tok, row['prefix_ids'])
                position = dst.tell(); offsets.append(position); kind_offsets[row['kind']].append(position)
                dst.write((json.dumps(row, separators=(',', ':'))+'\n').encode())
                if i % 10000 == 0: print(json.dumps({'phase': 'prefix_role_rebuild', 'split': split, 'examples': i}), flush=True)
            dst.flush(); os.fsync(dst.fileno())
        if len(offsets) != manifest['counts'][split]: raise ValueError('Role rebuild changed example count')
        np.save(stage/f'{split}.offsets.npy', np.asarray(offsets, dtype=np.int64))
        if split == 'train':
            for kind, off in kind_offsets.items(): np.save(stage/f'train.{kind}.offsets.npy', np.asarray(off, dtype=np.int64))
    manifest.update(roles_scope='observed_prefix_bytes_only', parent_role_cache_sha256=sha256_file(parent/'manifest.json'))
    manifest['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in stage.iterdir()}
    write_json_atomic(stage/'manifest.json', manifest); os.rename(stage, output)
    return load_data(output)


class MixedExamples:
    def __init__(self, root, seed=781):
        self.root, self.seed = Path(root), seed
        self.offsets = {k: np.load(self.root / f'train.{k}.offsets.npy', mmap_mode='r') for k in ('word', 'token')}
        self.orders = {}
        self.handle = (self.root / 'train.jsonl').open('rb')

    def at(self, cursor):
        kind = 'word' if cursor % 2 == 0 else 'token'
        epoch, index = divmod(cursor // 2, len(self.offsets[kind]))
        if self.orders.get(kind, (None,))[0] != epoch:
            order = list(range(len(self.offsets[kind])))
            random.Random(self.seed + epoch).shuffle(order)
            self.orders[kind] = epoch, order
        self.handle.seek(int(self.offsets[kind][self.orders[kind][1][index]]))
        return json.loads(self.handle.readline())


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    for arg in ('catalog', 'parent-data', 'tokenizer', 'output'): p.add_argument('--' + arg, required=True, type=Path)
    p.add_argument('--max-seq-len', type=int, default=8192)
    a = p.parse_args()
    prepare(a.catalog, a.parent_data, a.tokenizer, a.output, a.max_seq_len)
