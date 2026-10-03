"""Recompute the six required scores and paired comparisons without a GPU."""
import argparse
import ast
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

CONDITIONS = ("original", "sft_answer", "sft_reasoning")
BENCHMARKS = {"gsm8k": 1319, "svamp": 1000}
EXPECTED_CODE_HASH = "9f4e2186d2ed57b97994a7bc9bd217699027b0102e2814d61aceebdfca8b8f03"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_of(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def require(test, message):
    if not test:
        raise ValueError(message)


def notebook_helpers(notebook):
    namespace = dict(re=re, Decimal=Decimal, InvalidOperation=InvalidOperation, Fraction=Fraction)
    for index in (16, 19):
        tree = ast.parse(source_of(notebook["cells"][index]))
        tree.body = [node for node in tree.body if (
            index == 16 and isinstance(node, (ast.Assign, ast.FunctionDef))
        ) or (index == 19 and isinstance(node, ast.FunctionDef) and node.name == "compute_metrics")]
        exec(compile(tree, f"required-cell-{index}", "exec"), namespace)
    return namespace


def failure_category(raw, namespace):
    if namespace["extract_final_answer"](raw)[1] is not None:
        return "valid"
    lines = raw.rstrip().splitlines()
    markers = [i for i, line in enumerate(lines) if namespace["MARKER_RE"].match(line)]
    if not markers:
        return "missing_marker"
    if len(markers) > 1:
        return "multiple_markers"
    if markers[0] != len(lines) - 1:
        return "marker_not_final_line"
    return "invalid_final_number_or_line"


def paired_counts(before, after):
    require(len(before) == len(after), "Paired prediction counts differ.")
    require([r["example_id"] for r in before] == [r["example_id"] for r in after],
            "Paired example IDs differ.")
    common = [(a, b) for a, b in zip(before, after)
              if a["extraction_success"] and b["extraction_success"]]
    n = len(before)
    gained = sum(not a["correct"] and b["correct"] for a, b in zip(before, after))
    lost = sum(a["correct"] and not b["correct"] for a, b in zip(before, after))
    return {
        "n_examples": n, "wrong_to_correct": gained, "correct_to_wrong": lost,
        "net_additional_correct": gained - lost,
        "accuracy_difference": (gained - lost) / n if n else 0.0,
        "extraction_failure_to_success": sum(not a["extraction_success"] and b["extraction_success"]
                                             for a, b in zip(before, after)),
        "extraction_success_to_failure": sum(a["extraction_success"] and not b["extraction_success"]
                                             for a, b in zip(before, after)),
        "common_extracted_n": len(common),
        "before_correct_on_common": sum(a["correct"] for a, b in common),
        "after_correct_on_common": sum(b["correct"] for a, b in common),
        "before_accuracy_on_common": sum(a["correct"] for a, b in common) / len(common) if common else 0.0,
        "after_accuracy_on_common": sum(b["correct"] for a, b in common) / len(common) if common else 0.0,
    }


def protected_snapshot(root):
    root = Path(root)
    paths = [root / "math_reasoning.ipynb"]
    paths += sorted((root / "submission").glob("*"))
    paths += [root / "logs" / name for name in (
        "provenance.json", "split_manifest.json", "sft_answer_train_history.json",
        "sft_reasoning_train_history.json",
    )]
    paths += sorted((root / "logs").glob("*_config.json"))
    paths += [root / "ckpt" / f"{c}_lora.pt" for c in CONDITIONS[1:]]
    return {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}


def check_snapshot(root, snapshot):
    for relative, expected in snapshot.items():
        path = Path(root) / relative
        require(path.is_file() and file_hash(path) == expected,
                f"Required file changed: {relative}")


def check_split(manifest):
    train, val = manifest["train_indices"], manifest["val_indices"]
    require(manifest["seed"] == 42 and manifest["val_ratio"] == 0.1, "Required split settings differ.")
    require(len(train) == 6725 and len(val) == 748, "Required train/validation sizes differ.")
    require(all(type(i) is int for i in train + val), "Split indices must be integers.")
    require(len(set(train)) == len(train) and len(set(val)) == len(val), "Duplicate split indices.")
    require(set(train).isdisjoint(val) and set(train) | set(val) == set(range(7473)),
            "Invalid training/validation partition.")
    require(manifest["gsm8k_test_size"] == 1319 and manifest["svamp_total_size"] == 1000,
            "Required benchmark sizes differ.")


def audit_required(root):
    root = Path(root)
    notebook = read_json(root / "math_reasoning.ipynb")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    code_hash = digest([source_of(c) for c in code_cells])
    require(code_hash == EXPECTED_CODE_HASH, "Notebook code differs from the audited required run.")
    provenance = read_json(root / "logs" / "provenance.json")
    require(provenance["code_hash"] == code_hash, "Notebook/provenance hash mismatch.")
    require(provenance["model"] == "Qwen/Qwen2.5-1.5B-Instruct", "Wrong required base model.")
    for cell in code_cells:
        require(type(cell.get("execution_count")) is int, "The required notebook is not fully executed.")
        require(all(o.get("output_type") != "error" for o in cell.get("outputs", [])),
                "Required notebook contains an error output.")
    manifest = read_json(root / "logs" / "split_manifest.json")
    check_split(manifest)
    require(manifest["dataset_revisions"] == {k: provenance[k] for k in ("gsm8k_revision", "svamp_revision")},
            "Dataset revisions differ between the split and provenance.")
    namespace = notebook_helpers(notebook)
    predictions, metrics, histories = {}, [], {}
    for condition in CONDITIONS[1:]:
        history = read_json(root / "logs" / f"{condition}_train_history.json")
        expected = {"condition": condition, "seed": 42, "num_epochs": 2, "max_seq_length": 512,
                    "learning_rate": 1e-5, "effective_batch_size": 2, "optimizer": "AdamW",
                    "optimizer_weight_decay": 0.01, "scheduler": "none", "gradient_accumulation_steps": 1,
                    "n_train_examples": 6725, "n_val_examples": 748, "status": "complete",
                    "code_hash": code_hash, "split_hash": digest(manifest),
                    "model_revision": provenance["model_revision"], "packages": provenance["packages"]}
        require(all(history.get(k) == v for k, v in expected.items()), f"Training configuration differs: {condition}")
        require(len(history["train_loss"]) == len(history["val_loss"]) == 2, "Missing epoch losses.")
        require(all(math.isfinite(x) for x in history["train_loss"] + history["val_loss"]), "Nonfinite losses.")
        require(history["lora"]["rank"] == 8 and history["lora"]["alpha"] == 16
                and history["lora"]["dropout"] == 0.1, "Required LoRA settings differ.")
        require(set(history["lora"]["target_modules"]) == {
            "down_proj", "gate_proj", "k_proj", "lm_head", "o_proj", "q_proj", "up_proj", "v_proj"
        }, "Required LoRA targets differ.")
        adapter = root / "ckpt" / f"{condition}_lora.pt"
        if adapter.exists():
            require(file_hash(adapter) == history["adapter_sha256"], f"Adapter hash mismatch: {condition}")
        histories[condition] = history
    for benchmark, count in BENCHMARKS.items():
        for condition in CONDITIONS:
            rows = read_jsonl(root / "submission" / f"{condition}_{benchmark}_predictions.jsonl")
            require(len(rows) == count and len({r["example_id"] for r in rows}) == count,
                    f"Prediction count or IDs differ: {condition}/{benchmark}")
            for row in rows:
                answer, value = namespace["extract_final_answer"](row["raw_output"])
                _, gold = namespace["extract_final_answer"](row["gold_answer"])
                require(gold is not None, "Invalid gold answer.")
                require(row["condition"] == condition and row["benchmark"] == benchmark, "Wrong prediction labels.")
                require(type(row["correct"]) is bool and type(row["extraction_success"]) is bool,
                        "Prediction flags must be booleans.")
                require(row["extracted_answer"] == answer and row["extraction_success"] == (value is not None)
                        and row["correct"] == (value is not None and value == gold),
                        f"Prediction/parser mismatch: {condition}/{benchmark}/{row['example_id']}")
            predictions[(condition, benchmark)] = rows
            computed = namespace["compute_metrics"](rows)
            saved = read_json(root / "submission" / f"{condition}_{benchmark}_metrics.json")
            require(all(saved[k] == v for k, v in computed.items()), f"Metrics mismatch: {condition}/{benchmark}")
            config = read_json(root / "logs" / f"{condition}_{benchmark}_config.json")
            require(saved["run_hash"] == digest(config), "Evaluation fingerprint mismatch.")
            expected_data = [{("ID" if benchmark == "svamp" else "example_id"): r["example_id"],
                              "source": r["source"], "target": r["gold_answer"]} for r in rows]
            require(config["data_hash"] == digest(expected_data), "Evaluation data hash mismatch.")
            require(config["model_revision"] == provenance["model_revision"] and config["code_hash"] == code_hash
                    and config["packages"] == provenance["packages"] and config["split_hash"] == digest(manifest),
                    "Evaluation provenance differs.")
            require(config["decoding"] == {"do_sample": False, "max_new_tokens": 256, "batch_size": 8, "device": "cuda"},
                    "Required decoding settings differ.")
            if condition != "original":
                require(config["adapter_hash"] == histories[condition]["adapter_sha256"], "Evaluation adapter differs.")
            else:
                require(config["adapter_hash"] is None, "Original evaluation unexpectedly uses an adapter.")
            categories = {}
            for row in rows:
                category = failure_category(row["raw_output"], namespace)
                categories[category] = categories.get(category, 0) + 1
            metrics.append({**saved, "format_categories": categories})
        baseline = predictions[("original", benchmark)]
        for condition in CONDITIONS[1:]:
            current = predictions[(condition, benchmark)]
            require([(r["example_id"], r["source"], r["gold_answer"]) for r in current]
                    == [(r["example_id"], r["source"], r["gold_answer"]) for r in baseline],
                    "Models were not evaluated on identical ordered examples/prompts.")
    with (root / "submission" / "scores.csv").open(encoding="utf-8", newline="") as handle:
        scores = list(csv.DictReader(handle))
    lookup = {(m["condition"], m["benchmark"]): m for m in metrics}
    require(len(scores) == 6 and {(r["condition"], r["benchmark"]) for r in scores} == set(lookup),
            "Score summary must contain exactly the six required combinations.")
    for row in scores:
        saved = lookup[(row["condition"], row["benchmark"])]
        for key in ("n_examples", "n_extracted", "n_correct", "accuracy", "extraction_success_rate",
                    "conditional_accuracy", "wall_clock_seconds", "peak_gpu_memory_gb"):
            require(math.isclose(float(row[key]), saved[key], abs_tol=1e-12, rel_tol=1e-12), "CSV/JSON mismatch.")
        gap = lookup[(row["condition"], "gsm8k")]["accuracy"] - lookup[(row["condition"], "svamp")]["accuracy"]
        delta = saved["accuracy"] - lookup[("original", row["benchmark"])]["accuracy"]
        require(math.isclose(float(row["generalization_gap_svamp"]), gap, abs_tol=1e-12)
                and math.isclose(float(row["accuracy_change_vs_original"]), delta, abs_tol=1e-12),
                "CSV comparison metric mismatch.")
    paired, common_all, candidates = [], [], []
    for benchmark in BENCHMARKS:
        for before, after in itertools.combinations(CONDITIONS, 2):
            paired.append({"benchmark": benchmark, "before": before, "after": after,
                           **paired_counts(predictions[(before, benchmark)], predictions[(after, benchmark)])})
        triples = list(zip(*(predictions[(c, benchmark)] for c in CONDITIONS)))
        common = [triple for triple in triples if all(r["extraction_success"] for r in triple)]
        common_all.append({"benchmark": benchmark, "n_examples": len(common),
                           "n_correct": {c: sum(t[i]["correct"] for t in common) for i, c in enumerate(CONDITIONS)},
                           "note": "Selection-conditioned descriptive comparison, not whole-benchmark performance."})
        category_tests = {
            "reasoning_correct_answer_wrong": lambda o, a, r: r["correct"] and not a["correct"],
            "answer_correct_reasoning_wrong": lambda o, a, r: a["correct"] and not r["correct"],
            "both_finetuned_wrong_with_valid_answers": lambda o, a, r: a["extraction_success"] and r["extraction_success"]
                                                                           and not a["correct"] and not r["correct"],
            "original_correct_reasoning_wrong": lambda o, a, r: o["correct"] and not r["correct"],
        }
        for name, predicate in category_tests.items():
            choices = [t for t in triples if predicate(*t)]
            if choices:
                selected = min(choices, key=lambda t: (len(t[0]["source"]), str(t[0]["example_id"])))
                candidates.append({"benchmark": benchmark, "category": name, "category_size": len(choices),
                                   "selection": "Shortest source in this outcome category, then example ID.",
                                   "models": dict(zip(CONDITIONS, selected))})
    return {
        "audit": {"status": "verified", "code_hash": code_hash, "split_hash": digest(manifest),
                  "executed_code_cells": len(code_cells), "n_prediction_rows": sum(len(v) for v in predictions.values())},
        "provenance": provenance, "training_histories": histories, "metrics": metrics,
        "paired_comparisons": paired, "common_extracted_all_three": common_all,
        "case_candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_required(args.project.resolve())
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps({"audit": report["audit"], "paired_comparisons": report["paired_comparisons"],
                      "common_extracted_all_three": report["common_extracted_all_three"]}, indent=2))


if __name__ == "__main__":
    main()
