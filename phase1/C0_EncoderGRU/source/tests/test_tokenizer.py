import importlib
import json

import pytest


def api():
    try:
        return importlib.import_module("cs781.tokenizer")
    except ImportError as exc:
        pytest.fail(f"Tokenizer implementation is missing: {exc}")


def test_tokenizer_consumes_only_train_documents(catalog, tmp_path):
    t = api()
    output = tmp_path / "tokenizer.json"
    digest = t.fit_tokenizer(catalog, output, vocab_size=512)
    meta = json.loads(output.with_suffix(".meta.json").read_text())
    assert {d["id"] for d in meta["training_inputs"]} == {f"{s}:0" for s in ("acts", "high_courts", "supreme_court", "tribunals")}
    assert meta["sha256"] == digest
    t.validate_tokenizer_provenance(output, catalog)


def test_legacy_metadata_without_fingerprints_is_not_proof(catalog, tmp_path):
    t = api()
    from cs781.contracts import ProvenanceError
    output = tmp_path / "legacy.json"
    output.write_text("{}")
    output.with_suffix(".meta.json").write_text(json.dumps({"docs_seen": 4009}))
    with pytest.raises(ProvenanceError):
        t.validate_tokenizer_provenance(output, catalog)


def test_unicode_citations_round_trip_and_hash_tamper(catalog, tmp_path):
    t = api()
    from cs781.contracts import ProvenanceError
    output = tmp_path / "tokenizer.json"
    t.fit_tokenizer(catalog, output, vocab_size=512)
    tokenizer = t.load_tokenizer(output)
    text = "§ 12(3), AIR 2024 SC 123 — न्यायालय ₹100"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    output.write_text(output.read_text() + " ")
    with pytest.raises(ProvenanceError):
        t.validate_tokenizer_provenance(output, catalog)


def test_existing_or_partial_tokenizer_is_not_overwritten(catalog, tmp_path):
    t = api()
    output = tmp_path / "tokenizer.json"
    output.write_text("partial")
    with pytest.raises(FileExistsError):
        t.fit_tokenizer(catalog, output, vocab_size=512)
    assert output.read_text() == "partial"


def test_character_budget_is_balanced_across_sources(catalog,tmp_path):
    t=api()
    output=tmp_path/"balanced.json"
    t.fit_tokenizer(catalog,output,vocab_size=512,max_chars=1000)
    meta=json.loads(output.with_suffix(".meta.json").read_text())
    assert {d["id"].split(":")[0] for d in meta["training_inputs"]}=={"acts","high_courts","supreme_court","tribunals"}
    assert meta["chars"]==1000
