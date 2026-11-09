"""Project-owned byte BPE and train-only provenance for all CS781 variants."""
import json
from pathlib import Path

from .contracts import ProvenanceError, SPECIAL_TOKENS, SOURCES
from .io import sha256_file
from .own_bpe import OwnBPE, fit_catalog
from .splits import iter_catalog


def fit_tokenizer(database: Path, output: Path, vocab_size=32768, max_chars=100_000_000):
    fit_catalog(Path(database), Path(output), vocab_size, max_chars)
    return sha256_file(output)


def load_tokenizer(path):
    return OwnBPE.load(path)


def validate_tokenizer_provenance(path: Path, database: Path):
    path = Path(path)
    try:
        metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        tokenizer = load_tokenizer(path)
        if (not metadata.get("complete")
                or metadata.get("implementation") != "manual incremental byte BPE v1"
                or metadata["sha256"] != sha256_file(path)
                or metadata["catalog_sha256"] != sha256_file(database)
                or metadata.get("actual_vocab") != tokenizer.get_vocab_size()
                or metadata.get("special_ids") != {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
                or not all(metadata.get("per_source_chars", {}).get(source, 0) > 0 for source in SOURCES)):
            raise ValueError("Own BPE hash, vocabulary or four-source provenance mismatch")
        train = {doc.raw.id: doc.raw.fingerprint for doc in iter_catalog(database, "train")}
        inputs = metadata["training_inputs"]
        if not inputs or any(train.get(row["id"]) != row["fingerprint"] for row in inputs):
            raise ValueError("BPE inputs do not belong to training split")
    except (OSError, ValueError, KeyError, TypeError):
        raise ProvenanceError("Own BPE provenance is missing, mismatched or contains held-out documents") from None


def tokenizer_report(database, tokenizer_path):
    import numpy as np

    tokenizer = load_tokenizer(tokenizer_path)
    totals, prompt_lengths = {}, []
    for split in ("train", "dev", "confirmation"):
        for doc in iter_catalog(database, split):
            source = totals.setdefault(doc.raw.source, {"sampled_documents": 0, "words": 0, "tokens": 0})
            words = doc.raw.text.split()
            if source["sampled_documents"] < 64:
                source["sampled_documents"] += 1
                source["words"] += len(words)
                source["tokens"] += len(tokenizer.encode(doc.raw.text).ids)
            if split != "train" and len(words) >= 512:
                prompt_lengths.append(len(tokenizer.encode(" ".join(words[:512])).ids))
    percentiles = (dict(zip(("p50", "p95", "p99", "max"),
                            np.percentile(prompt_lengths, (50, 95, 99, 100)).tolist()))
                   if prompt_lengths else {})
    return {"per_source_sample": totals, "heldout_512word_prompt_count": len(prompt_lengths),
            "prompt_token_percentiles": percentiles}
