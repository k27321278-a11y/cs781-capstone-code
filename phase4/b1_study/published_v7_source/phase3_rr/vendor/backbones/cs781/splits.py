import hashlib
import json
import os
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .contracts import Document, RawDocument, SOURCES


def _simhash(words):
    if len(words) < 5:
        return None
    weights = [0] * 64
    counts = Counter(tuple(words[i:i + 5]) for i in range(len(words) - 4))
    for shingle, count in counts.items():
        bits = int.from_bytes(hashlib.sha256(" ".join(shingle).encode()).digest()[:8], "big")
        for b in range(64):
            weights[b] += count if (bits >> b) & 1 else -count
    return sum(1 << b for b, value in enumerate(weights) if value >= 0)


def _split(cluster, seed):
    value = int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < .98 else "dev" if value < .99 else "confirmation"


def catalog_documents(documents, database: Path, seed: int, *, corpus_config=None):
    database = Path(database)
    if database.exists():
        raise FileExistsError(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    staging = database.with_name(database.name + f".staging-{uuid.uuid4().hex}")
    parent = {}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    with sqlite3.connect(staging) as db:
        db.executescript("CREATE TABLE documents(id TEXT PRIMARY KEY,source TEXT,text TEXT,fingerprint TEXT,normalized TEXT,cluster_id TEXT,split TEXT); CREATE TABLE bands(band INTEGER,value INTEGER,fingerprint TEXT,simhash TEXT); CREATE INDEX band_lookup ON bands(band,value); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);")
        for doc in documents:
            if doc.source not in SOURCES or not doc.id or not doc.text.strip():
                raise ValueError("Invalid document source, ID or text")
            fingerprint = hashlib.sha256(doc.text.encode()).hexdigest()
            normalized = hashlib.sha256(" ".join(doc.text.lower().split()).encode()).hexdigest()
            old = db.execute("SELECT fingerprint FROM documents WHERE id=?", (doc.id,)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise ValueError(f"ID collision: {doc.id}")
                continue
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,NULL,NULL)", (doc.id, doc.source, doc.text, fingerprint, normalized))
            if normalized in parent:
                continue
            parent[normalized] = normalized
            sim = _simhash(doc.text.lower().split())
            if sim is not None:
                candidates = set()
                for band in range(4):
                    value = (sim >> (16 * band)) & 65535
                    candidates.update(db.execute("SELECT fingerprint,simhash FROM bands WHERE band=? AND value=?", (band, value)))
                for candidate, bits in candidates:
                    if (sim ^ int(bits, 16)).bit_count() <= 3:
                        union(normalized, candidate)
                db.executemany("INSERT INTO bands VALUES (?,?,?,?)", [(b, (sim >> (16*b)) & 65535, normalized, f"{sim:016x}") for b in range(4)])
        for normalized in parent:
            cluster = find(normalized)
            db.execute("UPDATE documents SET cluster_id=?,split=? WHERE normalized=?", (cluster, _split(cluster, seed), normalized))
        db.execute("CREATE INDEX split_lookup ON documents(split,id)")
        meta = {"complete": True, "seed": seed, "duplicate_algorithm": "sha256 + simhash64/5word/hamming3/4bands", "source_counts": dict(db.execute("SELECT source,COUNT(*) FROM documents GROUP BY source")), "corpus_config": asdict(corpus_config) if corpus_config else None}
        if not parent:
            raise ValueError("Empty corpus")
        db.execute("INSERT INTO metadata VALUES ('catalog',?)", (json.dumps(meta, sort_keys=True),))
    if database.exists():
        raise FileExistsError(database)
    os.replace(staging, database)


def iter_catalog(database: Path, split: str):
    if split not in ("train", "dev", "confirmation") or not Path(database).is_file():
        raise ValueError("Missing catalog or invalid split")
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True, check_same_thread=False) as db:
        for row in db.execute("SELECT id,source,text,fingerprint,cluster_id,split FROM documents WHERE split=? ORDER BY id", (split,)):
            yield Document(RawDocument(*row[:4]), row[4], row[5])


def catalog_metadata(database):
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
        return json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
