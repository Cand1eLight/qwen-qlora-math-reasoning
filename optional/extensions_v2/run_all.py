"""Queue v1 and three independently trained data/preprocessing comparisons."""
import argparse
import csv
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "extensions_v1"))
import run_supplement as core
from verify_results import (atomic_json, audit_required, check_snapshot, digest, file_hash,
                            notebook_helpers, paired_counts, protected_snapshot, read_json, read_jsonl, require)
from data_ops import DATA_SEED, EXTRA_COUNT, SOURCE_SHA256, clean_gsm, filter_external

VERSION = "extensions_v2"
VARIANTS = ("sft_reasoning_clean", "sft_reasoning_repeat_control", "sft_reasoning_mathqa")


def stored_protocol(path, value):
    if path.exists():
        require(read_json(path) == value, f"Stored protocol/input changed: {path}. Preserve this run; use a new versioned directory.")
    else:
        atomic_json(path, value)


def raw_inputs(project, manifest):
    trainpath = project / "data" / "gsm8k" / "train.jsonl"
    require(file_hash(trainpath) == manifest["official_train_sha256"], "GSM8K training snapshot changed.")
    official = read_jsonl(trainpath)
    gsm_test = read_jsonl(project / "data" / "gsm8k" / "test.jsonl")
    svamp = read_jsonl(project / "data" / "svamp" / "train.jsonl") + read_jsonl(project / "data" / "svamp" / "test.jsonl")
    require((len(official), len(gsm_test), len(svamp)) == (7473, 1319, 1000), "Wrong dataset snapshot counts.")
    # Compare raw test snapshots to the already audited saved predictions, not just their counts.
    for rows, raw, kind in ((read_jsonl(project / "submission" / "original_gsm8k_predictions.jsonl"), gsm_test, "gsm8k"),
                            (read_jsonl(project / "submission" / "original_svamp_predictions.jsonl"), svamp, "svamp")):
        helper = notebook_helpers(read_json(project / "math_reasoning.ipynb"))["extract_final_answer"]
        for saved, item in zip(rows, raw):
            question = item["question"] if kind == "gsm8k" else f"{item['Body'].strip()} {item['Question'].strip()}".strip()
            require(question in saved["source"], "Raw test question differs from required prediction source.")
            target = item["answer"] if kind == "gsm8k" else f"#### {item['Answer']}"
            require(helper(target)[1] == helper(saved["gold_answer"])[1], "Raw test gold differs.")
    return official, gsm_test, svamp


def prepare(project, output):
    report = audit_required(project)
    manifest = read_json(project / "logs" / "split_manifest.json")
    for package, version in report["provenance"]["packages"].items():
        require(core.metadata.version(package) == version, f"Package version differs: {package}")
    for condition in ("sft_answer", "sft_reasoning"):
        require(file_hash(project / "ckpt" / f"{condition}_lora.pt") == report["training_histories"][condition]["adapter_sha256"], "Required adapter changed/missing.")
    snapshot = protected_snapshot(project)
    stored_protocol(output / "required_file_hashes.json", snapshot)
    official, gsm_test, svamp = raw_inputs(project, manifest)
    archive = HERE / "assets" / "MathQA.zip"
    require(file_hash(archive) == SOURCE_SHA256, "Bundled official MathQA source checksum differs.")
    basic = {"version": VERSION, "source_hashes": {name: file_hash(HERE / name) for name in ("run_all.py", "data_ops.py")},
             "v1_source_hashes": {name: file_hash(HERE.parent / "extensions_v1" / name) for name in ("run_supplement.py", "verify_results.py")},
             "required_code_hash": report["audit"]["code_hash"], "split_hash": report["audit"]["split_hash"],
             "raw_data_hashes": {"gsm_train": digest(official), "gsm_test": digest(gsm_test), "svamp": digest(svamp)},
             "external_archive_sha256": SOURCE_SHA256, "external_n": EXTRA_COUNT, "data_seed": DATA_SEED,
             "variant_order": list(VARIANTS), "epochs": 2, "batch_size": 2, "lr": 1e-5,
             "rank": 8, "alpha": 16, "dropout": .1, "sequence_length": 512, "train_seed": 42,
             "decoding": {"do_sample": False, "batch_size": 8, "max_new_tokens": 256},
             "validation_offsets": core.sample_offsets(748), "test_policy": "Report all fixed variants on both full benchmarks; no test-based tuning.",
             "control_policy": "MathQA and repeated-GSM variants both have 8,725 examples and two epochs. Base/clean have 6,725. Separate independent adapters on the unchanged frozen base.",
             "scope": "One training seed; appended-data comparison controls sample/update count, not domain or token-length distributions."}
    stored_protocol(output / "input_protocol.json", basic)
    filtered_path, audit_path = output / "data" / "mathqa_selected.json", output / "data" / "data_audit.json"
    if filtered_path.exists() and audit_path.exists():
        selected, data_audit = read_json(filtered_path), read_json(audit_path)
        require(data_audit["input_protocol_hash"] == digest(basic) and data_audit["selected_data_hash"] == digest(selected), "Filtered data cache differs.")
        require(len(selected) == EXTRA_COUNT, "Wrong selected-data count.")
        core.log("Reusing the audited external training subset.")
    else:
        core.log("CPU preparation: checking 29,837 MathQA training questions against all GSM8K and SVAMP questions.")
        protected = [(f"gsm8k-official-train-{i}", r["question"]) for i, r in enumerate(official)]
        protected += [(f"gsm8k-test-{i}", r["question"]) for i, r in enumerate(gsm_test)]
        protected += [(f"svamp-{r['ID']}", f"{r['Body']} {r['Question']}") for r in svamp]
        selected, data_audit = filter_external(archive, protected)
        data_audit.update(input_protocol_hash=digest(basic), selected_data_hash=digest(selected))
        atomic_json(filtered_path, selected)
        atomic_json(audit_path, data_audit)
    train = [official[i] for i in manifest["train_indices"]]
    extract = notebook_helpers(read_json(project / "math_reasoning.ipynb"))["extract_final_answer"]
    cleaned, clean_audit = clean_gsm(train, extract)
    repeat_indices = random.Random(DATA_SEED).sample(range(len(train)), EXTRA_COUNT)
    repeat = train + [train[i] for i in repeat_indices]
    mixed = train + [{"question": r["question"], "answer": r["answer"]} for r in selected]
    variants = dict(zip(VARIANTS, (cleaned, repeat, mixed)))
    counts = [len(variants[c]) for c in VARIANTS]
    require(counts == [6725, 8725, 8725], "Variant sample counts differ.")
    protocol = {**basic, "variant_training_hashes": {c: digest(data) for c, data in variants.items()},
                "n_training_examples": dict(zip(VARIANTS, counts)), "repeat_training_offsets": repeat_indices,
                "external_selected_hash": digest(selected), "cleaning": clean_audit}
    stored_protocol(output / "protocol.json", protocol)
    atomic_json(output / "cleaning_audit.json", clean_audit)
    atomic_json(output / "error_analysis.json", {"case_candidates": report["case_candidates"],
                                                "paired": report["paired_comparisons"],
                                                "full_benchmark_metrics_and_format_failures": report["metrics"]})
    check_snapshot(project, snapshot)
    core.log(f"ALL OPTIONALS PREPARE PASSED: 2,000 external training rows; cleaning changed {clean_audit['changed_targets']} targets. No GPU work started.")
    return report, manifest, official, protocol, variants, snapshot


def finish_v1(project):
    import fcntl
    folder = project / "optional" / "extensions_v1"
    folder.mkdir(parents=True, exist_ok=True)
    announced = False
    with (folder / ".run.lock").open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                break
            except BlockingIOError:
                if not announced:
                    core.log("WAITING FOR V1: its existing job will finish first; no second GPU job is being launched.")
                    announced = True
                time.sleep(15)
    status = read_json(folder / "status.json") if (folder / "status.json").exists() else {}
    # Run --check even on completed v1, validating immutable source/input provenance.
    cmd = [sys.executable, str(folder / "run_supplement.py"), "--project", str(project)]
    subprocess.run(cmd + ["--check"], check=True, cwd=project)
    if status.get("status") != "complete":
        core.log("STAGE V1: starting/resuming capacity and generation-length comparisons.")
        with (folder / "console.log").open("a", encoding="utf-8") as handle:
            subprocess.run(cmd + ["--run"], stdout=handle, stderr=subprocess.STDOUT, check=True, cwd=project)
    status = read_json(folder / "status.json")
    require(status["status"] == "complete" and status["required_files_unchanged"], "V1 did not finish successfully.")
    summary = read_json(folder / "summary.json")
    require(summary["protocol_hash"] == digest(read_json(folder / "protocol.json")), "V1 summary/protocol differs.")
    require(len(summary["evaluations"]) == 9, "Incomplete V1 results.")
    verify_v1_results(project, summary)
    core.log("STAGE V1 COMPLETE: capacity and generation-length comparisons saved.")
    return summary


def verify_v1_results(project, summary):
    folder = project / "optional" / "extensions_v1"
    helpers = notebook_helpers(read_json(project / "math_reasoning.ipynb"))
    history = read_json(folder / "logs" / "sft_reasoning_r16_train_history.json")
    require(history["status"] == "complete" and history["extension_protocol_hash"] == summary["protocol_hash"]
            and history["adapter_sha256"] == file_hash(folder / "ckpt" / "sft_reasoning_r16_lora.pt"), "V1 adapter/history differs.")
    expected = {(c, "gsm8k_validation", n) for c in core.CONDITIONS for n in (256, 512)}
    expected |= {(core.VARIANT, b, 256) for b in ("gsm8k_validation", "gsm8k", "svamp")}
    require({(r["condition"], r["benchmark"], r["max_new_tokens"]) for r in summary["evaluations"]} == expected,
            "V1 evaluation plan differs.")
    for row in summary["evaluations"]:
        c, b, n = row["condition"], row["benchmark"], row["max_new_tokens"]
        result = read_json(folder / "results" / f"{c}_{b}_tokens{n}.json")
        cfg, predictions = result["config"], result["predictions"]
        require(result["fingerprint"] == digest(cfg) and cfg["protocol_hash"] == summary["protocol_hash"], "V1 result fingerprint differs.")
        require((cfg["condition"], cfg["benchmark"], cfg["max_new_tokens"]) == (c, b, n), "V1 labels differ.")
        require(len(predictions) == {"gsm8k_validation": 200, "gsm8k": 1319, "svamp": 1000}[b], "V1 result count differs.")
        for p in predictions:
            answer, value = helpers["extract_final_answer"](p["raw_output"])
            _, gold = helpers["extract_final_answer"](p["gold_answer"])
            require(gold is not None and p["extracted_answer"] == answer and p["extraction_success"] == (value is not None)
                    and p["correct"] == (value is not None and value == gold), "V1 saved scoring differs.")
            meta = core.output_token_metadata(p["generated_token_ids"], cfg["eos_ids"], n)
            require(meta == {key: p[key] for key in meta}, "V1 token metadata differs.")
        metrics = core.decoding_summary(predictions, helpers["compute_metrics"])
        require(result["metrics"] == metrics and all(row[k] == v for k, v in metrics.items()), "V1 saved metrics differ.")


def train_variant(ns, train, val, condition, output, protocol_hash):
    adapter = output / "ckpt" / f"{condition}_lora.pt"
    history_path = output / "logs" / f"{condition}_train_history.json"
    guard = {"protocol_hash": protocol_hash, "training_data_hash": digest(train.data),
             "loss_validation_data_hash": digest(val.data), "condition": condition}
    stored_protocol(output / "logs" / f"{condition}_inputs.json", guard)
    if adapter.exists() and history_path.exists():
        history = read_json(history_path)
        if history.get("status") == "complete":
            require(history["extension_protocol_hash"] == protocol_hash and history["adapter_sha256"] == file_hash(adapter), "Optional adapter/history differs.")
            core.log(f"Reusing completed adapter: {condition}")
            return history
    if adapter.exists():
        require((output / "ckpt" / f"{condition}_resume.pt").is_file(), "Incomplete adapter has no resume checkpoint.")
    ns["seed_all"]()
    hooks = ns["inject_lora"](ns["lm"], rank=8, alpha=16, dropout=.1)
    try:
        core.log(f"TRAIN {condition}: {len(train)} examples, two epochs, same r8/alpha16 settings.")
        history = ns["train_lora"](ns["lm"], ns["tokenizer"], hooks, train, val, num_epochs=2,
                                   batch_size=2, learning_rate=1e-5, device="cuda", max_len=512, condition=condition)
        require(history["n_train_examples"] == len(train) and history["n_val_examples"] == 748
                and history["num_epochs"] == 2 and history["n_trainable_params"] == 10460160
                and history["lora"]["rank"] == 8 and history["lora"]["alpha"] == 16,
                "Training settings differ from the protocol.")
        ns["save_lora_adapter"](hooks, str(adapter), rank=8, alpha=16, dropout=.1, target_modules=None)
        history.update(extension_protocol_hash=protocol_hash, adapter_sha256=file_hash(adapter), **guard)
        atomic_json(history_path, history)
        return history
    finally:
        ns["detach_lora"](hooks)
        gc.collect()
        ns["torch"].cuda.empty_cache()


def run_data_comparisons(project, output, report, manifest, official, protocol, variants, v1):
    core.check_gpu_idle()
    ns = core.load_runtime(project, output, report)
    tokenizer, cls = ns["tokenizer"], ns["GSM8K_Dataset"]
    val = cls([official[i] for i in manifest["val_indices"]], tokenizer, mode="sft_reasoning", max_len=512)
    offsets = protocol["validation_offsets"]
    selected = cls([official[manifest["val_indices"][i]] for i in offsets], tokenizer, mode="sft_reasoning", max_len=512)
    for row, offset in zip(selected.data, offsets):
        row["example_id"] = f"gsm8k-training-validation-{manifest['val_indices'][offset]:05d}"
    tests = {
        "gsm8k": cls(read_jsonl(project / "data" / "gsm8k" / "test.jsonl"), tokenizer, mode="default"),
        "svamp": ns["SVAMP_Dataset"](read_jsonl(project / "data" / "svamp" / "train.jsonl") + read_jsonl(project / "data" / "svamp" / "test.jsonl"), tokenizer)}
    for name, dataset in tests.items():
        require(digest(dataset.data) == read_json(project / "logs" / f"original_{name}_config.json")["data_hash"], "Optional test prompts/data differ.")
    protocol_hash = digest(protocol)
    evaluations, histories, paired = [], {}, []
    # Baseline on exactly the same pre-specified validation subset, already generated by v1.
    baseline_val = read_json(project / "optional" / "extensions_v1" / "results" / "sft_reasoning_gsm8k_validation_tokens256.json")
    require(baseline_val["config"]["data_hash"] == digest(selected.data), "Validation comparison sample differs from v1.")
    for condition in VARIANTS:
        train = cls(variants[condition], tokenizer, mode="sft_reasoning", max_len=512)
        histories[condition] = train_variant(ns, train, val, condition, output, protocol_hash)
        hooks = ns["load_lora_adapter"](ns["lm"], str(output / "ckpt" / f"{condition}_lora.pt"), device="cuda")
        try:
            for benchmark, dataset in [("gsm8k_validation", selected), *tests.items()]:
                core.log(f"EVAL {condition}/{benchmark}: fixed 256-token greedy decoding.")
                result = core.evaluate_optional(ns, dataset, condition, benchmark, 256, output, protocol_hash,
                                                histories[condition]["adapter_sha256"])
                evaluations.append({"condition": condition, "benchmark": benchmark, "max_new_tokens": 256,
                                    **result["metrics"], "wall_clock_seconds": result["wall_clock_seconds"],
                                    "peak_gpu_memory_gb": result["peak_gpu_memory_gb"]})
                base = baseline_val["predictions"] if benchmark == "gsm8k_validation" else read_jsonl(project / "submission" / f"sft_reasoning_{benchmark}_predictions.jsonl")
                paired.append({"condition": condition, "benchmark": benchmark, "reference": "required_sft_reasoning", **paired_counts(base, result["predictions"])})
        finally:
            ns["detach_lora"](hooks)
            gc.collect()
            ns["torch"].cuda.empty_cache()
    for benchmark in ("gsm8k_validation", "gsm8k", "svamp"):
        a = read_json(output / "results" / f"{VARIANTS[1]}_{benchmark}_tokens256.json")
        b = read_json(output / "results" / f"{VARIANTS[2]}_{benchmark}_tokens256.json")
        paired.append({"condition": VARIANTS[2], "benchmark": benchmark, "reference": VARIANTS[1],
                       **paired_counts(a["predictions"], b["predictions"])})
    combined = [dict(row) for row in v1["evaluations"] + evaluations]
    for condition in (core.VARIANT, *VARIANTS):
        full = {row["benchmark"]: row for row in combined if row["condition"] == condition and row["benchmark"] in tests}
        gap = full["gsm8k"]["accuracy"] - full["svamp"]["accuracy"]
        for benchmark, row in full.items():
            row["generalization_gap_svamp"] = gap
            original = ns["compute_metrics"](read_jsonl(project / "submission" / f"original_{benchmark}_predictions.jsonl"))
            row["accuracy_change_vs_original"] = row["accuracy"] - original["accuracy"]
    enriched_v2 = [row for row in combined if row["condition"] in VARIANTS]
    summary = {"protocol_hash": protocol_hash, "training": histories, "evaluations": enriched_v2, "paired_comparisons": paired,
               "v1_summary_hash": file_hash(project / "optional" / "extensions_v1" / "summary.json"),
               "data_audit_hash": file_hash(output / "data" / "data_audit.json"), "all_optional_directions": {
                   "error_analysis": "error_analysis.json and case_notes.txt; original saved predictions remain unchanged.",
                   "hyperparameters": "v1 rank8/alpha16 vs rank16/alpha32; 256/512 generation budget on fixed validation questions.",
                   "additional_training_data": "2,000 numeric, program-checked MathQA training rows plus matched-volume repeated-GSM control.",
                   "cleaning_preprocessing": "GSM8K calculator-annotation removal ablation; MathQA numeric filtering and lexical decontamination."},
               "interpretation_scope": protocol["scope"]}
    atomic_json(output / "summary.json", summary)
    fields = ["condition", "benchmark", "max_new_tokens", "n_examples", "accuracy", "extraction_success_rate", "conditional_accuracy", "generalization_gap_svamp", "accuracy_change_vs_original", "n_extracted", "n_correct", "wall_clock_seconds", "peak_gpu_memory_gb"]
    with (output / "optional_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    core.log("ALL OPTIONALS RUN COMPLETE: four new adapters and 18 optional evaluations across v1/v2; all four optional directions covered.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--overnight", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    require(HERE == project / "optional" / VERSION, "Extract the package inside the existing project first.")
    import fcntl
    with (HERE / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A v2 preparation/overnight job is already running; do not launch another.") from exc
        report, manifest, official, protocol, variants, snapshot = prepare(project, HERE)
        if args.prepare:
            return
        started = time.perf_counter()
        try:
            atomic_json(HERE / "status.json", {"status": "waiting_or_running", "protocol_hash": digest(protocol)})
            v1 = finish_v1(project)
            run_data_comparisons(project, HERE, report, manifest, official, protocol, variants, v1)
            check_snapshot(project, snapshot)
            atomic_json(HERE / "status.json", {"status": "complete", "protocol_hash": digest(protocol),
                                             "required_files_unchanged": True, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                             "this_invocation_wall_clock_seconds": time.perf_counter() - started})
        except Exception as exc:
            atomic_json(HERE / "status.json", {"status": "failed", "error": str(exc), "protocol_hash": digest(protocol)})
            raise
        finally:
            check_snapshot(project, snapshot)


if __name__ == "__main__":
    main()
