import importlib
import json
from zipfile import ZipFile

import pytest


def api():
    try:
        return importlib.import_module("cs781.submission")
    except ImportError as exc:
        pytest.fail(f"Submission implementation missing: {exc}")


def test_zip_has_only_root_prediction_json(tmp_path):
    s = api()
    output = tmp_path/"submission.zip"
    s.write_submission({"0001":"next "*128},{"0001":"prompt "*512},output)
    with ZipFile(output) as z:
        assert z.namelist()==["prediction.json"]
        assert set(json.loads(z.read("prediction.json")))=={"0001"}


@pytest.mark.parametrize("pred",[{"extra":"next "*128},{"0001":"short"},{"0001":"prompt "*128},{1:"next "*128}])
def test_invalid_ids_words_and_prompt_echo_are_rejected(tmp_path,pred):
    s = api()
    with pytest.raises(ValueError):
        s.write_submission(pred,{"0001":"prompt "*512},tmp_path/"submission.zip")
    assert not (tmp_path/"submission.zip").exists()


def test_duplicate_keys_and_nonstandard_json_are_rejected(tmp_path):
    s = api()
    path = tmp_path/"prediction.json"
    for text in ['{"a":1,"a":2}','{"a":NaN}']:
        path.write_text(text)
        with pytest.raises(ValueError):
            s.load_json_strict(path)
