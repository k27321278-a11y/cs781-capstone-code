import hashlib
from urllib.parse import quote

from .contracts import CorpusConfig, RawDocument, SOURCES, SourceReadError


def _source_rows(config, source):
    import ijson
    import requests
    from huggingface_hub.utils import build_hf_headers
    url = f"https://huggingface.co/datasets/{config.repo_id}/resolve/{config.revision}/{quote(config.source_paths[source], safe='/')}"
    with requests.get(url,headers=build_hf_headers(token=True),stream=True,timeout=(15,120)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        yield from ijson.items(response.raw,"item",buf_size=65536,use_float=True)


def _text(row):
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        for key in ("text", "document", "content", "body", "judgment", "case_text"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                result = _text(value)
                if result:
                    return result
        if len(row) == 1:
            return _text(next(iter(row.values())))
    return ""


def iter_official_documents(config: CorpusConfig):
    for source in SOURCES:
        count, chars = 0, 0
        try:
            for row in _source_rows(config, source):
                text = _text(row)
                if not text.strip():
                    continue
                fingerprint = hashlib.sha256(text.encode()).hexdigest()
                rid = next((str(row[k]) for k in ("id", "ID", "doc_id", "document_id", "case_id", "uid")
                            if isinstance(row, dict) and row.get(k) is not None), fingerprint)
                count += 1
                yield RawDocument(f"{source}:{rid}", source, text, fingerprint)
                chars += len(text)
                if config.max_chars_per_source is not None and chars >= config.max_chars_per_source:
                    break  # Whole documents only; the last document may exceed the recorded ceiling.
            if not count:
                raise SourceReadError(f"No readable documents from {source}")
        except SourceReadError:
            raise
        except Exception as exc:
            # Exception URLs/headers can contain credentials. Log the source/type only.
            raise SourceReadError(f"Official source {source} failed ({type(exc).__name__}); no fallback") from None
