import importlib
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from cs781.contracts import ModelConfig


def api():
    return importlib.import_module("cs781.submission")


def official_test_records(count=1000):
    prompt = " ".join(["word"] * 512)
    return {str(i): {"text": prompt} for i in range(count)}


def test_official_test_reader_preserves_all_string_ids_and_prompts():
    prompts = api().extract_official_test_prompts(official_test_records())
    assert len(prompts) == 1000
    assert list(prompts) == [str(i) for i in range(1000)]
    assert prompts["1"].split() == ["word"] * 512
    assert prompts["999"] == prompts["1"]


def test_official_test_reader_rejects_missing_ids():
    with pytest.raises(ValueError, match="1000"):
        api().extract_official_test_prompts(official_test_records(999))


@pytest.mark.parametrize("record", [
    {"body": "word"},
    {"text": "word", "extra": "not in the pinned schema"},
    {"text": "word " * 511},
    {"text": None},
])
def test_official_test_reader_rejects_wrong_schema_or_word_count(record):
    records = official_test_records()
    records["1"] = record
    with pytest.raises(ValueError):
        api().extract_official_test_prompts(records)


def test_official_test_reader_rejects_non_string_or_blank_ids():
    records = official_test_records()
    records[""] = records.pop("1")
    with pytest.raises(ValueError):
        api().extract_official_test_prompts(records)
    records = official_test_records()
    records[1] = records.pop("1")
    with pytest.raises(ValueError):
        api().extract_official_test_prompts(records)


def valid_provenance():
    revision = "a" * 40
    metadata = {
        "synthetic_smoke": False,
        "data_manifest_sha256": "d" * 64,
        "tokenizer_sha256": "e" * 64,
        "code_sha256": "f" * 64,
        "unique_parameters": 110_000_000,
        "model_config": asdict(ModelConfig()),
    }
    manifest = {
        "synthetic": False,
        "max_seq_len": 2048,
        "catalog": {"corpus_config": {
            "repo_id": "Exploration-Lab/CS-781-Capstone",
            "revision": revision,
        }},
    }
    data = SimpleNamespace(manifest_sha256="d" * 64, tokenizer_sha256="e" * 64)
    return metadata, manifest, data, revision


def test_official_checkpoint_provenance_accepts_matching_real_run():
    cli = importlib.import_module("cs781.cli")
    metadata, manifest, data, revision = valid_provenance()
    config, actual_revision = cli._verify_official_checkpoint_provenance(
        metadata, metadata.copy(), manifest, data
    )
    assert config.max_seq_len == 2048
    assert actual_revision == revision


@pytest.mark.parametrize("change", ["synthetic", "data", "run", "repo"])
def test_official_checkpoint_provenance_rejects_untrusted_lineage(change):
    cli = importlib.import_module("cs781.cli")
    metadata, manifest, data, _ = valid_provenance()
    run_metadata = metadata.copy()
    if change == "synthetic":
        metadata["synthetic_smoke"] = True
        run_metadata = metadata.copy()
    elif change == "data":
        data.manifest_sha256 = "0" * 64
    elif change == "run":
        run_metadata["code_sha256"] = "0" * 64
    else:
        manifest["catalog"]["corpus_config"]["repo_id"] = "someone/else"
    with pytest.raises(ValueError):
        cli._verify_official_checkpoint_provenance(metadata, run_metadata, manifest, data)


def test_checkpoint_path_must_be_inside_a_known_project_run(tmp_path):
    cli = importlib.import_module("cs781.cli")
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    (checkpoint.parent.parent / "run.json").write_text("{}")
    resolved, run_dir, _ = cli._checkpoint_run_directory(checkpoint)
    assert resolved == checkpoint.resolve()
    assert run_dir == checkpoint.parent.parent.resolve()
    arbitrary = tmp_path / "arbitrary.pt"
    arbitrary.write_bytes(b"placeholder")
    with pytest.raises(ValueError):
        cli._checkpoint_run_directory(arbitrary)


def test_predict_official_test_command_is_registered_and_dispatched(monkeypatch):
    cli = importlib.import_module("cs781.cli")
    captured = {}

    def fake_predict(args):
        captured["args"] = args

    monkeypatch.setattr(cli, "_predict_official_test", fake_predict, raising=False)
    result = cli.main([
        "predict-official-test",
        "--data", "prepared",
        "--checkpoint", "run/checkpoints/latest.pt",
        "--output", "submission.zip",
    ])

    assert result == 0
    assert captured["args"].data == "prepared"
    assert captured["args"].checkpoint == "run/checkpoints/latest.pt"
    assert captured["args"].output == "submission.zip"
    assert captured["args"].policy == 0
