"""Recompute all recorded outcomes from the compact generated-text exports."""
import argparse
import ast
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def helpers():
    notebook = json.loads((ROOT / "math_reasoning.ipynb").read_text(encoding="utf-8"))
    namespace = {"re": re, "Decimal": Decimal, "InvalidOperation": InvalidOperation, "Fraction": Fraction}
    for index in (16, 19):
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        tree.body = [node for node in tree.body if
                     (index == 16 and isinstance(node, (ast.Assign, ast.FunctionDef))) or
                     (index == 19 and isinstance(node, ast.FunctionDef) and node.name == "compute_metrics")]
        exec(compile(tree, f"parser-cell-{index}", "exec"), namespace)
    return namespace


def check_predictions(path, expected, namespace):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert len({row["example_id"] for row in rows}) == len(rows)
    for row in rows:
        answer, value = namespace["extract_final_answer"](row["raw_output"])
        _, gold = namespace["extract_final_answer"](row["gold_answer"])
        assert gold is not None
        assert row["extracted_answer"] == answer
        assert row["extraction_success"] == (value is not None)
        assert row["correct"] == (value is not None and value == gold)
    metrics = namespace["compute_metrics"](rows)
    for key, value in metrics.items():
        assert math.isclose(float(value), float(expected[key]), abs_tol=1e-12), (path.name, key)
    return rows, metrics


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/recorded")
    args = parser.parse_args()
    namespace = helpers()
    main_rows = csv_rows(args.results / "required/scores.csv")
    assert len(main_rows) == 6
    required, n_required = {}, 0
    for row in main_rows:
        stem = f"{row['condition']}_{row['benchmark']}"
        saved = json.loads((args.results / "required" / f"{stem}_metrics.json").read_text(encoding="utf-8"))
        predictions, metrics = check_predictions(args.results / "required" / f"{stem}_predictions.jsonl.gz", saved, namespace)
        for key, value in metrics.items():
            assert math.isclose(float(value), float(row[key]), abs_tol=1e-12)
        required[(row["condition"], row["benchmark"])] = predictions
        n_required += len(predictions)
    assert n_required == 6957
    optional_rows = csv_rows(args.results / "extensions_v2/optional_scores.csv")
    assert len(optional_rows) == 18
    n_optional = 0
    for row in optional_rows:
        stem = f"{row['condition']}_{row['benchmark']}_tokens{row['max_new_tokens']}"
        folders = [args.results / name for name in ("extensions_v1", "extensions_v2")]
        folder = next(folder for folder in folders if (folder / "metrics" / f"{stem}.json").exists())
        saved = json.loads((folder / "metrics" / f"{stem}.json").read_text(encoding="utf-8"))
        predictions, metrics = check_predictions(folder / "predictions" / f"{stem}.jsonl.gz", saved, namespace)
        for key, value in metrics.items():
            assert math.isclose(float(value), float(row[key]), abs_tol=1e-12)
        n_optional += len(predictions)
    for benchmark, expected in {"gsm8k": (534, 53), "svamp": (276, 72)}.items():
        before = required[("sft_answer", benchmark)]
        after = required[("sft_reasoning", benchmark)]
        assert [row["example_id"] for row in before] == [row["example_id"] for row in after]
        gained = sum(not a["correct"] and b["correct"] for a, b in zip(before, after))
        lost = sum(a["correct"] and not b["correct"] for a, b in zip(before, after))
        assert (gained, lost) == expected
    print(f"Verified {n_required} main predictions and {n_optional} optional predictions across 18 additional evaluations.")


if __name__ == "__main__":
    main()
