"""Score saved LegalSpan intervention generations with the course evaluator.

This is a small descriptive analysis over the fixed generated validation
examples, not model selection or a Codabench prediction.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arms = json.loads(args.generated.read_text(encoding="utf-8"))
    if not isinstance(arms, list) or not arms:
        raise ValueError("Expected non-empty arm list in generated intervention file")
    results = {}
    expected_ids = None
    for arm in arms:
        name = arm["arm"]
        rows = arm["outputs"]
        ids = [row["id"] for row in rows]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Arm {name} does not use the same validation IDs")
        truth = {row["id"]: row["reference_summary"] for row in rows}
        prediction = {row["id"]: row["summary"] for row in rows}
        with tempfile.TemporaryDirectory(prefix="phase6_legalspan_ablation_") as td:
            truth_path = Path(td) / "truth.json"
            prediction_path = Path(td) / "prediction.json"
            truth_path.write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
            prediction_path.write_text(json.dumps(prediction, ensure_ascii=False), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(args.evaluator), "--truth", str(truth_path), "--prediction", str(prediction_path)],
                text=True,
                capture_output=True,
                check=False,
            )
        if proc.returncode:
            raise RuntimeError(f"Course evaluator failed for {name}: {proc.stderr or proc.stdout}")
        text = proc.stdout
        def metric(label):
            match = re.search(rf"^{re.escape(label)}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
            if not match:
                raise ValueError(f"Could not parse {label} from evaluator output: {text}")
            return float(match.group(1))
        results[name] = {
            "count": len(rows),
            "bertscore_f1": metric("BERTScore F1"),
            "rouge1_f1": metric("ROUGE-1"),
            "rouge_l_f1": metric("ROUGE-L"),
        }
        print(json.dumps({"arm": name, **results[name]}), flush=True)

    output = {
        "purpose": "descriptive held-out validation intervention comparison",
        "interpretation_limit": "Only three generated examples per arm; do not treat as a reliable population estimate or leaderboard result.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
