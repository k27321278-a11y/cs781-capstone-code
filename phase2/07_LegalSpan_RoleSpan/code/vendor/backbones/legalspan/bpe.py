"""Manual byte BPE: Sennrich et al. ACL 2016; no external trainer or codec.

Weighted piece counts, pair postings and a lazy maximum heap avoid corpus rescans
per merge. Encoding replays learned ranks; segmentation is lossless and explicit.
"""
import heapq
import json
import os
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPECIALS = ('<pad>', '<bos>', '<eos>', '<unk>')
POLICY = 'maximal whitespace / Unicode L,M / Nd / remaining runs; UTF8; no normalization'


def char_class(c):
    if c.isspace():
        return 0
    category = unicodedata.category(c)
    return 1 if category[0] in 'LM' else 2 if category == 'Nd' else 3


def pieces(text, errors='strict'):
    start, byte_start, byte_end, kind = 0, 0, 0, None
    for i, c in enumerate(text):
        current = char_class(c)
        if kind is not None and current != kind:
            yield text[start:i], (byte_start, byte_end)
            start, byte_start = i, byte_end
        kind = current
        byte_end += len(c.encode('utf-8', errors=errors))
    if text:
        yield text[start:], (byte_start, byte_end)


def merge_row(row, pair, new):
    output, i = [], 0
    while i < len(row):
        if i + 1 < len(row) and (row[i], row[i+1]) == pair:
            output.append(new)
            i += 2
        else:
            output.append(row[i])
            i += 1
    return tuple(output)


@dataclass
class Encoding:
    ids: list
    offsets: list


class OwnBPE:
    def __init__(self, merges=()):
        self.merges = [tuple(p) for p in merges]
        self.ranks = {pair: (rank, 260 + rank) for rank, pair in enumerate(self.merges)}
        self.bytes = [b''] * 4 + [bytes([b]) for b in range(256)]
        for pair in self.merges:
            if min(pair) < 4 or max(pair) >= len(self.bytes):
                raise ValueError('Invalid noncausal BPE merge table')
            self.bytes.append(self.bytes[pair[0]] + self.bytes[pair[1]])

    @classmethod
    def train(cls, texts, vocab_size=32768, progress=None):
        if vocab_size < 260:
            raise ValueError('Own byte BPE requires at least 260 entries')
        weights = Counter(p.encode('utf-8') for text in texts for p, _ in pieces(text))
        if not weights:
            raise ValueError('Empty BPE training corpus')
        rows, row_weights = [], []
        counts, postings = Counter(), defaultdict(set)
        for raw, weight in sorted(weights.items()):
            row = tuple(b + 4 for b in raw)
            rid = len(rows)
            rows.append(row); row_weights.append(weight)
            local = Counter(zip(row, row[1:]))
            for pair, n in local.items():
                counts[pair] += n * weight
                postings[pair].add(rid)
        del weights
        heap = [(-count, *pair) for pair, count in counts.items()]
        heapq.heapify(heap)
        merges = []
        while heap and len(merges) + 260 < vocab_size:
            negative, a, b = heapq.heappop(heap)
            pair = (a, b)
            if counts[pair] != -negative or negative == 0:
                continue
            new = 260 + len(merges)
            deltas = Counter()
            for rid in sorted(postings[pair]):
                old = rows[rid]
                row = merge_row(old, pair, new)
                old_counts = Counter(zip(old, old[1:]))
                new_counts = Counter(zip(row, row[1:]))
                for p in old_counts.keys() | new_counts.keys():
                    deltas[p] += (new_counts[p] - old_counts[p]) * row_weights[rid]
                    if p in new_counts:
                        postings[p].add(rid)
                    else:
                        postings[p].discard(rid)
                rows[rid] = row
            for p, delta in deltas.items():
                counts[p] += delta
                if counts[p] < 0:
                    raise RuntimeError('Negative weighted pair count')
                if counts[p]:
                    heapq.heappush(heap, (-counts[p], *p))
                else:
                    counts.pop(p, None)
                    postings.pop(p, None)
            merges.append(pair)
            if len(heap) > 4 * max(len(counts), 1):
                heap = [(-count, *p) for p, count in counts.items() if count]
                heapq.heapify(heap)
            if progress and (len(merges) % 1000 == 0 or len(merges) + 260 == vocab_size):
                progress({'merges': len(merges), 'active_pairs': len(counts), 'pieces': len(rows)})
        return cls(merges)

    @lru_cache(maxsize=200000)
    def _encode_piece(self, raw):
        row = tuple(b + 4 for b in raw)
        while len(row) > 1:
            options = [(self.ranks[p], p) for p in zip(row, row[1:]) if p in self.ranks]
            if not options:
                break
            (_, new), pair = min(options)
            row = merge_row(row, pair, new)
        return row

    def encode(self, text):
        ids, offsets = [], []
        for p, (start, _) in pieces(text):
            for token in self._encode_piece(p.encode('utf-8')):
                end = start + len(self.bytes[token])
                ids.append(token); offsets.append((start, end))
                start = end
        return Encoding(ids, offsets)

    def decode(self, ids, skip_special_tokens=True):
        raw = b''.join(self.bytes[i] if i >= 4 else b'' if skip_special_tokens else SPECIALS[i].encode() for i in ids)
        return raw.decode('utf-8', errors='replace')

    def token_to_id(self, text):
        return SPECIALS.index(text) if text in SPECIALS else None

    def get_vocab_size(self):
        return len(self.bytes)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as f:
            json.dump({'format': 'cs781-own-byte-bpe-v1', 'policy': POLICY,
                       'unicode_version': unicodedata.unidata_version, 'merges': self.merges}, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        if record.get('format') != 'cs781-own-byte-bpe-v1' or record.get('policy') != POLICY or record.get('unicode_version') != unicodedata.unidata_version:
            raise ValueError('Own BPE format, segmentation or Unicode version mismatch')
        return cls(record['merges'])


def fit_catalog(database, output, vocab_size=32768, max_chars=100000000):
    from cs781.splits import iter_catalog
    from cs781.contracts import SOURCES
    from cs781.io import write_json_atomic, sha256_file
    output = Path(output)
    if output.exists() or output.with_suffix('.meta.json').exists():
        raise FileExistsError(output)
    if max_chars < 4:
        raise ValueError('Positive per-source fitting budget required')
    inputs, used = [], dict.fromkeys(SOURCES, 0)
    budgets = {s: max_chars // 4 + int(i < max_chars % 4) for i, s in enumerate(SOURCES)}
    def texts():
        for doc in iter_catalog(database, 'train'):
            text = doc.raw.text[:budgets[doc.raw.source] - used[doc.raw.source]]
            if text:
                used[doc.raw.source] += len(text)
                inputs.append({'id': doc.raw.id, 'fingerprint': doc.raw.fingerprint,
                               'chars': len(text), 'bytes': len(text.encode()), 'source': doc.raw.source})
                yield text
    tok = OwnBPE.train(texts(), vocab_size, progress=lambda row: print(json.dumps(row), flush=True))
    if any(n == 0 for n in used.values()):
        raise ValueError('Tokenizer fitting requires all four official train sources')
    tok.save(output)
    write_json_atomic(output.with_suffix('.meta.json'), {
        'complete': True, 'implementation': 'manual incremental byte BPE v1',
        'sha256': sha256_file(output), 'catalog_sha256': sha256_file(database),
        'training_inputs': inputs, 'per_source_chars': used, 'chars': sum(used.values()),
        'requested_vocab': vocab_size, 'actual_vocab': tok.get_vocab_size(),
        'policy': POLICY, 'unicode_version': unicodedata.unidata_version,
        'special_ids': dict(zip(SPECIALS, range(4)))})
    return tok


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-chars', type=int, default=100000000)
    parser.add_argument('--vocab-size', type=int, default=32768)
    args = parser.parse_args()
    fit_catalog(Path(args.catalog), Path(args.output), args.vocab_size, args.max_chars)
