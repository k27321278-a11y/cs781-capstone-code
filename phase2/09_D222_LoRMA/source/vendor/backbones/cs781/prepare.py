import hashlib
import json
import os
import random
import shutil
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import ContinuationExample, IncompleteDataError, PreparedData, SOURCES
from .io import sha256_file, write_json_atomic
from .splits import catalog_metadata, iter_catalog
from .tokenizer import load_tokenizer, tokenizer_report, validate_tokenizer_provenance


def prepare_data(database, tokenizer_path, output, seed, *, max_seq_len=2048):
    output, tokenizer_path = Path(output), Path(tokenizer_path)
    if output.exists():
        raise FileExistsError(output)
    if max_seq_len < 4:
        raise ValueError("Context too small")
    validate_tokenizer_provenance(tokenizer_path, database)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
    shutil.copyfile(tokenizer_path.with_suffix(".meta.json"), staging / "tokenizer.meta.json")
    counts, source_counts, excluded, overlength = {}, {}, {}, {}
    for split in ("train", "dev", "confirmation"):
        prompts, refs, offsets = {}, {}, []
        counts[split], excluded[split], overlength[split] = 0, 0, 0
        sources = Counter()
        with (staging / f"{split}.jsonl").open("wb") as handle:
            tokenizer = load_tokenizer(tokenizer_path)
            for doc in iter_catalog(database, split):
                if split == "train":
                    ids = tokenizer.encode(doc.raw.text).ids
                    windows = []
                    for start in range(0, len(ids), max_seq_len - 1):
                        chunk = ids[start:start + max_seq_len - 1]
                        if len(chunk) < 2:
                            continue
                        rng = random.Random(int(hashlib.sha256(f"{seed}:{doc.raw.id}:{start}".encode()).hexdigest()[:16], 16))
                        cut = rng.randint(max(1, len(chunk)//4), min(len(chunk)-1, 3*len(chunk)//4))
                        windows.append((f"{doc.raw.id}:{start}", chunk[:cut], chunk[cut:]))
                else:
                    words = doc.raw.text.split()
                    if len(words) < 640:
                        excluded[split] += 1
                        continue
                    value = int(hashlib.sha256(f"{seed}:{doc.raw.id}".encode()).hexdigest()[:16], 16)
                    start = value % (len(words) - 640 + 1)
                    prefix = " ".join(words[start:start + 512])
                    suffix = " ".join(words[start + 512:start + 640])
                    eid = doc.raw.id
                    prompts[eid], refs[eid] = {"text": prefix}, {"text": suffix}
                    windows = [(eid, tokenizer.encode(prefix).ids, tokenizer.encode(suffix).ids)]
                for eid, prefix, suffix in windows:
                    if not prefix or not suffix:
                        excluded[split] += 1
                        continue
                    e = ContinuationExample(eid, doc.raw.id, doc.raw.source, split, prefix, suffix)
                    offsets.append(handle.tell())
                    handle.write((json.dumps(asdict(e), ensure_ascii=False) + "\n").encode())
                    counts[split] += 1
                    sources[doc.raw.source] += 1
                    overlength[split] += int(len(prefix) + len(suffix) + 1 > max_seq_len)
            handle.flush()
            os.fsync(handle.fileno())
        if not counts[split] or set(sources) != set(SOURCES):
            raise IncompleteDataError(f"{split} is empty or lacks an official source; staging preserved")
        source_counts[split] = dict(sources)
        np.save(staging / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))
        if split != "train":
            write_json_atomic(staging / f"{split}.prompts.json", prompts)
            write_json_atomic(staging / f"{split}.references.json", refs)
    files = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in staging.iterdir() if p.is_file()}
    catalog = catalog_metadata(database)
    manifest = {"complete": True, "counts": counts, "source_counts": source_counts, "excluded_short": excluded,
                "overlength": overlength, "max_seq_len": max_seq_len, "seed": seed, "files": files,
                "catalog_sha256": sha256_file(database), "catalog": catalog, "synthetic": catalog["corpus_config"] is None,
                "tokenizer_report": tokenizer_report(database, tokenizer_path),
                "training_window_policy": "nonoverlapping token windows; seeded random 25–75% prefix; suffix+EOS supervised"}
    write_json_atomic(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(output)
    os.rename(staging, output)
    return load_prepared(output)


def load_prepared(root):
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError("Incomplete manifest")
        required = {"tokenizer.json", "tokenizer.meta.json"}
        for split in ("train", "dev", "confirmation"):
            required.update({f"{split}.jsonl", f"{split}.offsets.npy"})
            if split != "train":
                required.update({f"{split}.prompts.json", f"{split}.references.json"})
            if manifest["counts"][split] <= 0 or set(manifest["source_counts"][split]) != set(SOURCES):
                raise ValueError("Missing split/source")
            offsets = np.load(root / f"{split}.offsets.npy", allow_pickle=False)
            if len(offsets) != manifest["counts"][split] or offsets.ndim != 1 or offsets[0] != 0 or (np.diff(offsets) <= 0).any():
                raise ValueError("Bad example offsets")
        if set(manifest["files"]) != required:
            raise ValueError("Wrong cache files")
        for filename, expected in manifest["files"].items():
            path = root / filename
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise ValueError("Truncated or mismatched cache")
        return PreparedData(root, sha256_file(root / "manifest.json"), *(manifest["counts"][s] for s in ("train","dev","confirmation")), manifest["files"]["tokenizer.json"]["sha256"])
    except (OSError, KeyError, ValueError, TypeError, IndexError):
        raise IncompleteDataError("Data must have a complete, nonempty, hash-verified manifest") from None
