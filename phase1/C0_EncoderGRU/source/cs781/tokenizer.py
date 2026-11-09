import json
import os
import uuid
from pathlib import Path

from .contracts import ProvenanceError, SPECIAL_TOKENS, SOURCES
from .io import sha256_file, write_json_atomic
from .splits import iter_catalog


def fit_tokenizer(database: Path, output: Path, vocab_size=32768, max_chars=100_000_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    output = Path(output)
    metadata_path = output.with_suffix(".meta.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(output)
    if vocab_size < 260 or max_chars < 1:
        raise ValueError("Byte BPE needs >=260 vocabulary entries and a positive character budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs, chars = [], 0
    used = dict.fromkeys(SOURCES,0)
    budgets = {s:max_chars//4+int(i<max_chars%4) for i,s in enumerate(SOURCES)}
    def texts():
        nonlocal chars
        for doc in iter_catalog(database, "train"):
            text = doc.raw.text[:min(max_chars-chars,budgets[doc.raw.source]-used[doc.raw.source])]
            if not text:
                if chars>=max_chars:
                    break
                continue
            inputs.append({"id": doc.raw.id, "fingerprint": doc.raw.fingerprint, "chars": len(text)})
            chars += len(text)
            used[doc.raw.source] += len(text)
            yield text
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if not inputs:
        raise ProvenanceError("No training documents for BPE")
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    tokenizer.save(str(temporary))
    digest = sha256_file(temporary)
    os.replace(temporary, output)
    write_json_atomic(metadata_path, {"complete": True, "sha256": digest, "catalog_sha256": sha256_file(database),
                                      "training_inputs": inputs, "chars": chars, "requested_vocab": vocab_size,
                                      "actual_vocab": tokenizer.get_vocab_size(),
                                      "per_source_chars":used,"source_budget_policy":"equal character ceilings, no reallocation from missing sources",
                                      "special_ids": {s: tokenizer.token_to_id(s) for s in SPECIAL_TOKENS}})
    return digest


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def validate_tokenizer_provenance(path: Path, database: Path):
    path = Path(path)
    try:
        metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if not metadata.get("complete") or metadata["sha256"] != sha256_file(path) or metadata["catalog_sha256"] != sha256_file(database):
            raise ValueError("Hash or completion mismatch")
        train = {d.raw.id: d.raw.fingerprint for d in iter_catalog(database, "train")}
        inputs = metadata["training_inputs"]
        if not inputs or any(train.get(d["id"]) != d["fingerprint"] for d in inputs):
            raise ValueError("BPE inputs do not belong to training split")
    except (OSError, ValueError, KeyError, TypeError):
        raise ProvenanceError("Tokenizer provenance is missing, mismatched or contains held-out documents") from None


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
    return {"per_source_sample": totals, "heldout_512word_prompt_count": len(prompt_lengths),
            "prompt_token_percentiles": dict(zip(["p50", "p95", "p99", "max"], np.percentile(prompt_lengths, [50,95,99,100]).tolist())) if prompt_lengths else {}}
