import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_official_secondary_metrics_match_fixture(monkeypatch):
    path = Path(__file__).parents[1]/"vendor/phase1_evaluate.py"
    assert path.is_file(), "Published evaluator not preserved yet"
    monkeypatch.setitem(sys.modules,"bert_score",types.ModuleType("bert_score"))
    spec = importlib.util.spec_from_file_location("official_fixture",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hyp,ref = "The court decides an appeal.","The courts were deciding appeals."
    assert module.rouge_l_f1(hyp,ref) == pytest.approx(.2)
    assert module.bleu4([hyp.split()],[ref.split()]) == 0


def test_scorer_output_parser_requires_all_finite_metrics():
    try:
        e = importlib.import_module("cs781.evaluate")
    except ImportError as exc:
        pytest.fail(f"Scorer adapter missing: {exc}")
    actual = e.parse_metrics("Examples: 1\nBERTScore F1: 0.800000\nROUGE-L F1: 0.200000\nBLEU-4: 0.000000\n")
    assert actual == {"bertscore_f1":.8,"rouge_l_f1":.2,"bleu4":0}
    with pytest.raises(ValueError):
        e.parse_metrics("BERTScore F1: nan")


def _scorer_cache(tmp_path):
    e = importlib.import_module("cs781.evaluate")
    cache = tmp_path / "cache"
    snapshot = cache / "models--roberta-large" / "snapshots" / e.SCORER_REVISION
    snapshot.mkdir(parents=True)
    ref = cache / "models--roberta-large" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(e.SCORER_REVISION, encoding="utf-8")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
    records = []
    for name in names:
        payload = f"verified fixture for {name}".encode()
        (snapshot / name).write_bytes(payload)
        records.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    record = {
        "requested_id": e.SCORER_ALIAS,
        "resolved_id": e.SCORER_REPOSITORY,
        "revision": e.SCORER_REVISION,
        "snapshot": str(snapshot),
        "files": records,
        "offline_config_tokenizer_resolution": "PASS",
    }
    (tmp_path / "bert_score_model_provenance.json").write_text(json.dumps(record), encoding="utf-8")
    return cache


def test_official_scorer_uses_verified_pinned_cache_offline(tmp_path, monkeypatch):
    e = importlib.import_module("cs781.evaluate")
    cache = _scorer_cache(tmp_path)
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    evaluator = tmp_path / "official.py"
    truth.write_text("{}", encoding="utf-8")
    prediction.write_text("{}", encoding="utf-8")
    evaluator.write_text("# preserved evaluator fixture\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout="Examples: 1\nBERTScore F1: 0.800000\nROUGE-L F1: 0.200000\nBLEU-4: 0.000000\n",
        )

    monkeypatch.setattr(e.subprocess, "run", fake_run)
    metrics = e.score_predictions(truth, prediction, evaluator, cache)
    assert metrics["bertscore_f1"] == pytest.approx(0.8)
    assert captured["env"]["HF_HUB_CACHE"] == str(cache.resolve())
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
    report = json.loads(prediction.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert report["bert_score_settings"]["revision"] == e.SCORER_REVISION
    assert report["bert_score_settings"]["offline"] is True


def test_official_scorer_rejects_unpinned_cache_alias(tmp_path):
    e = importlib.import_module("cs781.evaluate")
    cache = _scorer_cache(tmp_path)
    (cache / "models--roberta-large" / "refs" / "main").write_text("different-revision", encoding="utf-8")
    with pytest.raises(ValueError, match="not pinned"):
        e.load_scorer_provenance(cache)
