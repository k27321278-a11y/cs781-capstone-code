import sqlite3

import pytest


@pytest.fixture
def catalog(tmp_path):
    from cs781.contracts import RawDocument, SOURCES
    from cs781.splits import catalog_documents
    db = tmp_path / "catalog.sqlite"
    documents = [RawDocument(f"{s}:{i}", s, " ".join(f"{s}{i}word{j}" for j in range(700)), "") for s in SOURCES for i in range(3)]
    catalog_documents(documents, db, 781)
    # Hand-assigned fixture split; real production splitting is tested separately.
    with sqlite3.connect(db) as conn:
        for i, split in enumerate(("train", "dev", "confirmation")):
            conn.execute("UPDATE documents SET split=? WHERE id LIKE ?", (split, f"%:{i}"))
    return db


@pytest.fixture
def tiny_config():
    from cs781.contracts import ModelConfig
    return ModelConfig(vocab_size=512, max_seq_len=128, n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128)
