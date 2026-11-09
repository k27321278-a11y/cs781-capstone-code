"""Reuse pinned official public test, roundtrip and profile all1000 own-BPE prompts."""
import argparse
import json
import shutil
from pathlib import Path
from cs781.cli import _download_official_test
from cs781.io import sha256_file, write_json_atomic
from cs781.submission import extract_official_test_prompts, load_json_strict
from legalspan.bpe import OwnBPE
from legalspan.pipeline import context_preflight
def main():
    p=argparse.ArgumentParser(); p.add_argument('--task',type=Path,required=True); a=p.parse_args()
    base=a.task.parent
    manifest=load_json_strict(a.task/'prepared/manifest.json')
    revision=manifest['catalog']['corpus_config']['revision']
    test=_download_official_test(revision,base/'official_100m_chars_per_source/hf_hub')
    prompts=extract_official_test_prompts(load_json_strict(test))
    report=context_preflight(prompts,OwnBPE.load(a.task/'tokenizer.json'),8192,2048)
    shutil.copyfile(test,a.task/'official_test.json')
    report.update(test_sha256=sha256_file(test),dataset_revision=revision)
    write_json_atomic(a.task/'public_context_preflight.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
