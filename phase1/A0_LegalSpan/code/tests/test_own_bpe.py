"""Catch frequency/overlap errors, irreversible byte segmentation and held-out fitting."""
import importlib.util
import random
from collections import Counter


def api():
    assert importlib.util.find_spec('legalspan.bpe') is not None, 'Own BPE codec is not implemented'
    from legalspan.bpe import OwnBPE, pieces
    return OwnBPE, pieces


def reference(texts, size):
    _, pieces = api()
    rows = Counter(tuple(b + 4 for b in p.encode()) for t in texts for p, _ in pieces(t))
    merges = []
    while 260 + len(merges) < size:
        counts = Counter()
        for row, weight in rows.items():
            for pair in zip(row, row[1:]):
                counts[pair] += weight
        if not counts:
            break
        pair = min(counts, key=lambda x: (-counts[x], x))
        new = 260 + len(merges)
        merged = Counter()
        for row, weight in rows.items():
            out, i = [], 0
            while i < len(row):
                if i + 1 < len(row) and row[i:i+2] == pair:
                    out.append(new); i += 2
                else:
                    out.append(row[i]); i += 1
            merged[tuple(out)] += weight
        rows = merged
        merges.append(pair)
    return merges


def test_incremental_learner_matches_independent_reference():
    BPE, _ = api()
    rng = random.Random(19)
    for texts in [['aaaaa aaaa aaa', 'abab aba'], ['भारत §123\nÁ', 'ABC ABC 22222'],
                  [''.join(rng.choice('ab c12!') for _ in range(40)) for _ in range(10)]]:
        assert BPE.train(texts, vocab_size=300).merges == reference(texts, 300)


def test_byte_roundtrip_offsets_literal_specials_and_reload(tmp_path):
    BPE, _ = api()
    text = 'भारत\tÁ १२३\r\n<bos>  repeatedaaaaa\u2003§'
    tok = BPE.train([text] * 3, vocab_size=310)
    encoded = tok.encode(text)
    assert tok.decode(encoded.ids) == text
    raw = text.encode()
    assert b''.join(raw[a:b] for a, b in encoded.offsets) == raw
    assert all(i >= 4 for i in encoded.ids)
    path = tmp_path / 'tokenizer.json'
    tok.save(path)
    loaded = BPE.load(path)
    assert loaded.encode(text).ids == encoded.ids
    assert loaded.get_vocab_size() == tok.get_vocab_size()
    assert tok.token_to_id('<eos>') == 2


def test_run_segmentation_retains_whitespace_and_unicode_categories():
    _, pieces = api()
    assert list(pieces('Á\t १२३.§')) == [('Á', (0, 3)), ('\t ', (3, 5)), ('१२३', (5, 14)), ('.§', (14, 17))]


def test_fit_catalog_never_reads_heldout_and_records_equal_source_caps(tmp_path):
    import sqlite3
    import json
    BPE, _ = api()
    import legalspan.bpe as module
    assert hasattr(module, 'fit_catalog'), 'Train-only catalog fitting not implemented'
    db = tmp_path / 'catalog.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE documents(id,source,text,fingerprint,cluster_id,split)')
        for i, source in enumerate(('acts', 'high_courts', 'supreme_court', 'tribunals')):
            c.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', (str(i), source, 'ababa '*20, str(i), str(i), 'train'))
        c.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', ('heldout', 'acts', 'zzzzzz '*20, 'h', 'h', 'dev'))
    path = tmp_path / 'tok.json'
    module.fit_catalog(db, path, vocab_size=280, max_chars=80)
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    assert {x['id'] for x in meta['training_inputs']} == {'0', '1', '2', '3'}
    assert meta['per_source_chars'] == dict.fromkeys(('acts', 'high_courts', 'supreme_court', 'tribunals'), 20)
    assert BPE.load(path).decode(BPE.load(path).encode('zzzzzz').ids) == 'zzzzzz'
