import importlib

import pytest
import torch


def api():
    try:
        return importlib.import_module("cs781.examples")
    except ImportError as exc:
        pytest.fail(f"Example implementation missing: {exc}")


def test_first_suffix_is_supervised_and_prefix_is_not():
    e = api()
    from cs781.contracts import ContinuationExample
    batch = e.collate_a([ContinuationExample("id", "doc", "acts", "train", [10,11], [20,21])], 0, 1, 2, 8)
    assert batch.input_ids.tolist() == [[1,10,11,20,21]]
    assert batch.targets.tolist() == [[-100,-100,20,21,2]]
    assert batch.target_tokens == 3
    assert batch.input_tokens == 5


def test_right_padding_and_one_token_suffix():
    e = api()
    from cs781.contracts import ContinuationExample as X
    batch = e.collate_a([X("a","a","acts","train",[10],[20]), X("b","b","acts","train",[10,11],[20,21])],0,1,2,8)
    assert batch.targets.tolist()[0] == [-100,20,2,-100,-100]
    assert batch.target_tokens == 5
    assert batch.input_tokens == 8


def test_overlength_and_empty_suffix_are_rejected():
    e = api()
    from cs781.contracts import ContinuationExample as X
    with pytest.raises(ValueError):
        e.collate_a([X("a","a","acts","train",[10]*8,[20])],0,1,2,8)
    with pytest.raises(ValueError):
        e.collate_a([X("a","a","acts","train",[10],[])],0,1,2,8)
