import json
import os
import uuid
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

from .io import sha256_file

OFFICIAL_TEST_COUNT = 1000


def load_json_strict(path):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonstandard JSON numeric constant")
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle,object_pairs_hook=pairs,parse_constant=constant)


def extract_official_test_prompts(records):
    if not isinstance(records,dict) or len(records)!=OFFICIAL_TEST_COUNT:
        raise ValueError(f"Pinned official test must contain exactly {OFFICIAL_TEST_COUNT} IDs")
    prompts = {}
    for key,record in records.items():
        if not isinstance(key,str) or not key.strip() or not isinstance(record,dict) or set(record)!={"text"}:
            raise ValueError("Official test rows must map nonempty string IDs to exactly one text field")
        text = record["text"]
        if not isinstance(text,str) or len(text.split())!=512:
            raise ValueError("Every official test prompt must contain exactly 512 words")
        prompts[key] = text
    return prompts


def validate_predictions(predictions,expected_prompts):
    if not predictions or set(predictions)!=set(expected_prompts):
        raise ValueError("Exact nonempty ID match required")
    for key,text in predictions.items():
        if not isinstance(key,str) or not key.strip() or not isinstance(text,str) or len(text.split())!=128:
            raise ValueError("String IDs and 128-word text required")
        prompt = expected_prompts[key]
        if not isinstance(prompt,str) or len(prompt.split())!=512:
            raise ValueError("512-word prompt required")
        normalized = " ".join(text.split())
        # Reject an entire output copied from the prompt, not legitimate short legal overlaps.
        if normalized in " ".join(prompt.split()):
            raise ValueError("Output is a prompt echo, not a continuation")


def write_submission(predictions,expected_prompts,output):
    validate_predictions(predictions,expected_prompts)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+".tmp-"+uuid.uuid4().hex)
    records = {i:{"text":text.strip()} for i,text in predictions.items()}
    with ZipFile(temporary,"w",compression=ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json",json.dumps(records,ensure_ascii=False,allow_nan=False))
    with ZipFile(temporary) as archive:
        if archive.namelist()!=["prediction.json"]:
            raise ValueError("Wrong ZIP root")
        actual = json.loads(archive.read("prediction.json"))
        validate_predictions({i:r["text"] for i,r in actual.items()},expected_prompts)
    os.replace(temporary,output)
    return sha256_file(output)
