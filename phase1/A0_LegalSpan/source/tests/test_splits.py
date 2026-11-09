import sqlite3
import importlib
import threading

import pytest


def api():
    try:
        return importlib.import_module("cs781.splits"), importlib.import_module("cs781.contracts")
    except ImportError as exc:
        pytest.fail(f"Split implementation is missing: {exc}")


def documents(c):
    text = "The court considered the appeal under section 12 and held the petition was admissible. " * 30 + " ".join(f"word{i}" for i in range(1000))
    return [c.RawDocument("a", "acts", text, ""), c.RawDocument("b", "high_courts", text.upper(), ""),
            c.RawDocument("c", "supreme_court", text + " costs ", ""),
            c.RawDocument("short1", "tribunals", "one", ""), c.RawDocument("short2", "tribunals", "two", "")]


def assignments(splits, db):
    return {d.raw.id: (d.cluster_id, d.split) for split in ["train", "dev", "confirmation"] for d in splits.iter_catalog(db, split)}


def test_duplicate_clusters_do_not_cross_splits(tmp_path):
    s, c = api()
    db = tmp_path / "catalog.sqlite"
    s.catalog_documents(documents(c), db, 781)
    a = assignments(s, db)
    assert a["a"] == a["b"] == a["c"]
    assert a["short1"][0] != a["short2"][0]


def test_split_is_independent_of_input_order(tmp_path):
    s, c = api()
    s.catalog_documents(documents(c), tmp_path / "one.sqlite", 781)
    s.catalog_documents(reversed(documents(c)), tmp_path / "two.sqlite", 781)
    assert assignments(s, tmp_path / "one.sqlite") == assignments(s, tmp_path / "two.sqlite")


def test_id_collision_and_overwrite_are_rejected(tmp_path):
    s, c = api()
    db = tmp_path / "catalog.sqlite"
    with pytest.raises(ValueError):
        s.catalog_documents([c.RawDocument("same", "acts", "one", ""), c.RawDocument("same", "acts", "two", "")], db, 781)
    assert not db.exists()
    good = tmp_path / "good.sqlite"
    s.catalog_documents(documents(c), good, 781)
    with pytest.raises(FileExistsError):
        s.catalog_documents(documents(c), good, 781)


def test_catalog_iterator_can_resume_in_tokenizer_worker_thread(tmp_path):
    s, c = api()
    db = tmp_path / "catalog.sqlite"
    s.catalog_documents(documents(c), db, 781)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE documents SET split='train'")

    rows = s.iter_catalog(db, "train")
    first = next(rows)
    remaining, failures = [], []

    def consume_remaining():
        try:
            remaining.extend(rows)
        except Exception as exc:  # surfaced in the test thread below
            failures.append(exc)

    worker = threading.Thread(target=consume_remaining)
    worker.start()
    worker.join()
    assert not failures
    assert first.raw.id == "a"
    assert len(remaining) == 4
