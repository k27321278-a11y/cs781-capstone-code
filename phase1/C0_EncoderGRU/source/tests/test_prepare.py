import importlib
import json
import sqlite3

import pytest


def api():
    try:
        return importlib.import_module("cs781.prepare")
    except ImportError as exc:
        pytest.fail(f"Preparation implementation missing: {exc}")


def fit(catalog, tmp_path):
    from cs781.tokenizer import fit_tokenizer
    path = tmp_path / "tokenizer.json"
    fit_tokenizer(catalog, path, 512)
    return path


def test_partial_data_cannot_load(tmp_path):
    p = api()
    from cs781.contracts import IncompleteDataError
    tmp_path.joinpath("train.jsonl").write_text('{"partial":true}')
    with pytest.raises(IncompleteDataError):
        p.load_prepared(tmp_path)


def test_preparation_finalizes_only_complete_data(catalog, tmp_path):
    p = api()
    data = p.prepare_data(catalog, fit(catalog,tmp_path), tmp_path/"data", 781)
    assert data.train_examples > 0
    assert data.dev_examples == data.confirmation_examples == 4
    assert p.load_prepared(data.root) == data
    prompts = json.loads((data.root / "dev.prompts.json").read_text())
    refs = json.loads((data.root / "dev.references.json").read_text())
    assert set(prompts) == set(refs)
    assert all(len(v["text"].split()) == 512 for v in prompts.values())
    assert all(len(v["text"].split()) == 128 for v in refs.values())
    (data.root/"train.jsonl").write_text("truncated")
    from cs781.contracts import IncompleteDataError
    with pytest.raises(IncompleteDataError):
        p.load_prepared(data.root)


def test_empty_heldout_or_missing_source_does_not_publish(catalog, tmp_path):
    p = api()
    with sqlite3.connect(catalog) as db:
        db.execute("DELETE FROM documents WHERE split='dev'")
    output = tmp_path / "data"
    with pytest.raises(Exception):
        p.prepare_data(catalog, fit(catalog,tmp_path), output, 781)
    assert not output.exists()
    assert list(tmp_path.glob("data.staging-*"))


def test_existing_data_is_preserved(catalog, tmp_path):
    p = api()
    output = tmp_path / "data"
    output.mkdir()
    (output/"sentinel").write_text("old")
    with pytest.raises(FileExistsError):
        p.prepare_data(catalog, fit(catalog,tmp_path), output, 781)
    assert (output/"sentinel").read_text() == "old"
