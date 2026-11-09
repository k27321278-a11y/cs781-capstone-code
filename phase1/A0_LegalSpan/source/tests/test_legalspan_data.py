"""Catch held-out leakage, missing IDs, incorrect roles and cross-document windows."""
import importlib.util
import json
import sqlite3
from pathlib import Path
import pytest
from legalspan.bpe import OwnBPE, fit_catalog


def api():
    assert importlib.util.find_spec('legalspan.data') is not None, 'Own-tokenizer data preparation not implemented'
    from legalspan.data import prepare, load_data, role_ids
    return prepare, load_data, role_ids


def fixture(tmp):
    db = tmp / 'catalog.sqlite'; parent = tmp / 'parent'; parent.mkdir()
    sources = ('acts', 'high_courts', 'supreme_court', 'tribunals')
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE documents(id,source,text,fingerprint,cluster_id,split)')
        c.execute('CREATE TABLE metadata(key,value)')
        c.execute('INSERT INTO metadata VALUES(?,?)', ('catalog', json.dumps({'complete': True, 'corpus_config': None})))
        for i, source in enumerate(sources):
            c.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', (str(i), source, 'Section 123 ABC ordinary. '*400, str(i), str(i), 'train'))
        for split in ('dev', 'confirmation'):
            prompts, refs = {}, {}
            for i, source in enumerate(sources):
                key = split + str(i)
                c.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', (key, source, 'heldout '*640, key, key, split))
                prompts[key] = {'text': 'heldout '*512}; refs[key] = {'text': 'heldout '*128}
            (parent / f'{split}.prompts.json').write_text(json.dumps(prompts))
            (parent / f'{split}.references.json').write_text(json.dumps(refs))
    tok = tmp / 'tokenizer.json'; fit_catalog(db, tok, vocab_size=300, max_chars=4000)
    return db, parent, tok


def test_own_dataset_exact_task_mix_and_retains_all_heldout_ids(tmp_path):
    prepare, load, roles = api()
    db, parent, tok = fixture(tmp_path)
    root = tmp_path / 'prepared'
    prepare(db, parent, tok, root, max_seq_len=8192)
    data = load(root)
    rows = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    assert all(row['document_id'] in {'0', '1', '2', '3'} for row in rows)
    assert {row['kind'] for row in rows} == {'word', 'token'}
    words = [row for row in rows if row['kind'] == 'word']
    tokenizer = OwnBPE.load(root / 'tokenizer.json')
    assert all(len(tokenizer.decode(r['prefix_ids']).split()) == 512 and len(tokenizer.decode(r['suffix_ids']).split()) == 128 for r in words)
    assert data['counts']['dev'] == data['counts']['confirmation'] == 4
    assert set(json.loads((root / 'dev.prompts.json').read_text())) == {'dev0', 'dev1', 'dev2', 'dev3'}
    text = 'Section 123 ABC ordinary.\n'
    enc = tokenizer.encode(text)
    mark = roles(text, enc)
    for (a, b), role in zip(enc.offsets, mark):
        raw = text.encode()[a:b].decode()
        if raw == 'Section': assert role == 2
        if raw == '123': assert role == 1
        if raw == 'ABC': assert role == 3


def test_overlength_and_modified_cache_are_rejected(tmp_path):
    prepare, load, _ = api()
    db, parent, tok = fixture(tmp_path)
    with pytest.raises(ValueError, match='context'):
        prepare(db, parent, tok, tmp_path / 'short', max_seq_len=64)
    root = tmp_path / 'prepared'
    prepare(db, parent, tok, root, max_seq_len=8192)
    with (root / 'train.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='hash'):
        load(root)


def test_roles_depend_only_on_observed_prefix_including_bisected_cue():
    _, _, _ = api()
    import legalspan.data as data
    assert hasattr(data, 'observed_role_ids'), 'Prefix-only role derivation missing'
    tok = OwnBPE()
    prefix = tok.encode('Sect').ids
    assert data.observed_role_ids(tok, prefix) == [3] * 4
    # It must not receive statutory-cue role2 merely because unseen continuation
    # would finish Section. Known full Section and incomplete UTF8 are separate.
    assert data.observed_role_ids(tok, tok.encode('Section').ids) == [2] * 7
    partial = [4 + 0xe0, 4 + 0xa4]
    assert data.observed_role_ids(tok, partial) == [0, 0]


def test_versioned_role_rebuild_preserves_every_token_and_id(tmp_path):
    prepare, load, _ = api()
    import legalspan.data as module
    assert hasattr(module, 'rebuild_roles'), 'Versioned role-cache rebuild missing'
    db, parent, tok = fixture(tmp_path); original = tmp_path / 'old'; new = tmp_path / 'new'
    prepare(db, parent, tok, original)
    module.rebuild_roles(original, new)
    a = [json.loads(s) for s in (original/'train.jsonl').read_text().splitlines()]
    b = [json.loads(s) for s in (new/'train.jsonl').read_text().splitlines()]
    assert [(r['id'], r['prefix_ids'], r['suffix_ids'], r['kind']) for r in a] == [(r['id'], r['prefix_ids'], r['suffix_ids'], r['kind']) for r in b]
    assert load(new)['counts'] == load(original)['counts']
