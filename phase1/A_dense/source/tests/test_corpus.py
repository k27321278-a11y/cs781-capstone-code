import importlib
import io
import json

import pytest


def api():
    try:
        return importlib.import_module("cs781.corpus"), importlib.import_module("cs781.contracts")
    except ImportError as exc:
        pytest.fail(f"Corpus implementation is missing: {exc}")


def test_source_failure_is_not_a_fallback(monkeypatch):
    corpus, c = api()
    def broken(config, source):
        yield {"id": "1", "text": "The judgment of the court."}
        raise OSError("disconnected")
    monkeypatch.setattr(corpus, "_source_rows", broken)
    cfg = c.CorpusConfig(revision="a" * 40)
    with pytest.raises(c.SourceReadError):
        list(corpus.iter_official_documents(cfg))


def test_missing_id_depends_on_full_content(monkeypatch):
    corpus, c = api()
    texts = ["x " * 300 + "first", "x " * 300 + "second"]
    monkeypatch.setattr(corpus, "_source_rows", lambda config, source: iter([{"text": t} for t in texts]))
    docs = list(corpus.iter_official_documents(c.CorpusConfig(revision="a" * 40)))
    assert len({d.id for d in docs}) == 8
    assert docs[0].text == texts[0]


def test_empty_source_and_unpinned_revision_are_rejected(monkeypatch):
    corpus, c = api()
    monkeypatch.setattr(corpus, "_source_rows", lambda config, source: iter([]))
    with pytest.raises(c.SourceReadError):
        list(corpus.iter_official_documents(c.CorpusConfig(revision="a" * 40)))
    with pytest.raises(ValueError):
        c.CorpusConfig(revision="main")


def test_authenticated_reader_uses_cached_login(monkeypatch):
    corpus,c = api()
    install_response(monkeypatch, b'[{"id":"one","text":"A court judgment."}]')
    docs=list(corpus.iter_official_documents(c.CorpusConfig(revision="a"*40)))
    assert {d.source for d in docs} == set(c.SOURCES)


def install_response(monkeypatch, payload, raw=None):
    import datasets
    import requests
    from huggingface_hub import utils

    class Response:
        def __init__(self):
            self.raw = raw if raw is not None else io.BytesIO(payload)
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self,*args):
            self.raw.close()

    def cached_headers(*,token,**kwargs):
        assert token is True
        return {"Authorization":"Bearer fixture-only"}

    def get(url,**kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer fixture-only"
        assert kwargs["stream"] is True
        assert kwargs["timeout"][1] <= 120
        return Response()

    monkeypatch.setattr(utils,"build_hf_headers",cached_headers)
    monkeypatch.setattr(requests,"get",get)
    # Isolate the old loader's network calls during RED. Its installed array
    # fallback materializes JSON before yielding; the new parser remains real.
    monkeypatch.setattr(datasets,"load_dataset",lambda *a,**k: json.load(raw) if raw is not None else json.loads(payload))


def test_json_array_returns_first_record_without_reading_whole_file(monkeypatch):
    corpus,c = api()
    payload = (b'[{"id":"first","text":"A court judgment."},' + b' '*131072
               + json.dumps({"id":"second","text":"x"*1_000_000}).encode() + b']')

    class BoundedStream(io.BytesIO):
        def read(self,size=-1):
            assert 0 <= size <= 65536, "Reader requested an unbounded/full-file read"
            return super().read(size)
        def readinto(self,buffer):
            assert len(buffer) <= 65536
            return super().readinto(buffer)

    stream=BoundedStream(payload)
    install_response(monkeypatch,payload,raw=stream)
    reader=corpus._source_rows(c.CorpusConfig(revision="a"*40),"acts")
    try:
        assert next(reader) == {"id":"first","text":"A court judgment."}
        assert stream.tell() <= 65536
    finally:
        reader.close()


def test_truncated_json_array_fails_without_source_fallback(monkeypatch):
    corpus,c = api()
    install_response(monkeypatch,b'[{"text":"A court judgment."},')
    with pytest.raises(c.SourceReadError):
        list(corpus.iter_official_documents(c.CorpusConfig(revision="a"*40)))


def test_source_subset_is_bounded_without_dropping_sources(monkeypatch):
    corpus,c = api()
    monkeypatch.setattr(corpus,"_source_rows",lambda config,source: iter([{"id":str(i),"text":"court "*20} for i in range(10)]))
    cfg=c.CorpusConfig(revision="a"*40,max_chars_per_source=200)
    docs=list(corpus.iter_official_documents(cfg))
    assert len(docs)==8
    assert {d.source for d in docs}==set(c.SOURCES)
