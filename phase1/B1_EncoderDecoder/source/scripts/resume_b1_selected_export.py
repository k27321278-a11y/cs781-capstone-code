"""Export the already selected B1 checkpoint after a non-selection confirmation failure.

This keeps the completed full-dev selection immutable and generates only the
official Codabench ZIP using its selected checkpoint and decode policy.
"""

from __future__ import annotations

import json
from pathlib import Path

from cs781.b_upgrade import export
from cs781.io import sha256_file, write_json_atomic
from cs781.prepare import load_prepared
from cs781.submission import load_json_strict


ART = Path("/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006")
RUN = ART / "b1_improved"
SELECTION = RUN / "evaluation" / "selection.json"
OUTPUT = ART / "CS781_B1_OwnBPE_Codabench.zip"
REPORT = RUN / "evaluation" / "recovery_official_export.json"


def main() -> None:
    selection = load_json_strict(SELECTION)
    selected = selection["selected"]
    checkpoint = Path(selected["checkpoint"])
    policy = selected["policy"]
    if not checkpoint.is_file() or not isinstance(policy, int):
        raise ValueError("Selected B1 checkpoint or decode policy is malformed")
    original = load_prepared(ART / "prepared")
    aligned = load_prepared(ART / "b_taskaligned")
    submission = export(checkpoint, original, aligned, policy, OUTPUT, SELECTION,
                        ART / "official_test.json")
    record = {
        "reason": "official export recovery after non-selected baseline confirmation generation failure",
        "selection_sha256": sha256_file(SELECTION),
        "selected_checkpoint": str(checkpoint),
        "selected_policy": policy,
        "submission": submission,
    }
    if REPORT.exists() and load_json_strict(REPORT) != record:
        raise ValueError("Existing recovery export report differs")
    write_json_atomic(REPORT, record)
    print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
