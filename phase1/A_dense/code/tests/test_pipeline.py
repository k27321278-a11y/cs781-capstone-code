import importlib
import os
import subprocess
from pathlib import Path

import pytest


def api():
    try:
        return importlib.import_module("cs781.cli")
    except ImportError as exc:
        pytest.fail(f"CLI implementation missing: {exc}")


def test_incomplete_data_cli_refuses_training(tmp_path):
    c = api()
    output = tmp_path/"run"
    assert c.main(["train-a","--data",str(tmp_path/"partial"),"--run-dir",str(output),"--target-tokens","100"]) != 0
    assert not (output/"checkpoints/latest.pt").exists()


def test_prepare_failure_never_starts_training(tmp_path):
    script = Path(__file__).parents[1]/"scripts/run_a.sh"
    assert script.is_file(), "Fail-fast launcher not implemented"
    trace = tmp_path/"trace"
    fake = tmp_path/"python"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CS781_TRACE"\nexit 9\n')
    fake.chmod(0o755)
    env = {**os.environ,"PYTHON_BIN":str(fake),"CS781_TRACE":str(trace)}
    process = subprocess.run(["bash",str(script),str(tmp_path/"data"),str(tmp_path/"run"),"0","100"],env=env,capture_output=True)
    assert process.returncode != 0
    assert "validate-data" in trace.read_text()
    assert "train-a" not in trace.read_text()


def test_status_does_not_create_a_run(tmp_path):
    c = api()
    output = tmp_path/"absent"
    assert c.main(["status","--run-dir",str(output)]) != 0
    assert not output.exists()
