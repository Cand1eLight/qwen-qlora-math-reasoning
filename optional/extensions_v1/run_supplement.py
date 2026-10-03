"""Run isolated, pre-specified capacity and generation-budget comparisons."""
import argparse
import ast
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from verify_results import (CONDITIONS, atomic_json, audit_required, check_snapshot,
                            digest, file_hash, paired_counts, protected_snapshot,
                            read_json, read_jsonl, require, source_of)

SAMPLE_SEED = 20260927
SAMPLE_SIZE = 200
VARIANT = "sft_reasoning_r16"
VERSION = "extensions_v1"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sample_offsets(size, count=SAMPLE_SIZE):
    require(size >= count, "Validation set is smaller than the specified sample.")
    return sorted(random.Random(SAMPLE_SEED).sample(range(size), count))


def output_token_metadata(token_ids, eos_ids, budget):
    """Count through the first EOS; exclude subsequent batch padding."""
    ids = [int(v) for v in token_ids]
    require(budget > 0 and len(ids) <= budget, "Invalid generated-token count or budget.")
    first_eos = next((i for i, value in enumerate(ids) if value in eos_ids), None)
    actual = ids if first_eos is None else ids[:first_eos + 1]
    return {"generated_token_ids": actual, "generated_tokens": len(actual),
            "content_tokens": len(actual) if first_eos is None else len(actual) - 1,
            "stopped_by_eos": first_eos is not None,
            "budget_reached_without_eos": first_eos is None and len(actual) == budget}


def decoding_summary(rows, compute_metrics):
    metrics = compute_metrics(rows)
    n = len(rows)
    limited = [r for r in rows if r["budget_reached_without_eos"]]
    metrics.update({
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n if n else 0.0,
        "n_budget_reached_without_eos": len(limited),
        "budget_reached_without_eos_rate": len(limited) / n if n else 0.0,
        "n_extraction_failures_at_budget": sum(not r["extraction_success"] for r in limited),
    })
    return metrics


def budget_comparison(short, long):
    result = paired_counts(short, long)
    matching = [b["generated_token_ids"][:len(a["generated_token_ids"])] == a["generated_token_ids"]
                for a, b in zip(short, long)]
    result.update({
        "n_matching_token_prefixes": sum(matching), "n_changed_token_prefixes": len(matching) - sum(matching),
        "n_short_budget_reached": sum(r["budget_reached_without_eos"] for r in short),
        "n_format_recovered_after_short_budget": sum(
            match and a["budget_reached_without_eos"] and not a["extraction_success"] and b["extraction_success"]
            for a, b, match in zip(short, long, matching)),
        "n_correct_recovered_after_short_budget": sum(
            match and a["budget_reached_without_eos"] and not a["correct"] and b["correct"]
            for a, b, match in zip(short, long, matching)),
        "note": "Reaching a budget without EOS is a measured stopping event, not proof that the answer would be correct later.",
    })
    return result


def load_definitions(notebook, namespace):
    for index in (3, 10, 16, 19, 22, 25):
        tree = ast.parse(source_of(notebook["cells"][index]))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     or (index in (10, 16) and isinstance(node, ast.Assign))]
        exec(compile(tree, f"required-definitions-{index}", "exec"), namespace)


def preflight(project, output):
    log("Checking required predictions, notebook, training histories and cached dataset files.")
    report = audit_required(project)
    manifest = read_json(project / "logs" / "split_manifest.json")
    official_path = project / "data" / "gsm8k" / "train.jsonl"
    require(official_path.is_file(), "Missing cached GSM8K training snapshot. Do not rerun the required notebook.")
    require(file_hash(official_path) == manifest["official_train_sha256"], "Training snapshot changed.")
    official = read_jsonl(official_path)
    require(len(official) == 7473, "Wrong GSM8K training snapshot size.")
    for relative, count in (("gsm8k/test.jsonl", 1319), ("svamp/train.jsonl", 700), ("svamp/test.jsonl", 300)):
        require(len(read_jsonl(project / "data" / relative)) == count, f"Wrong cached dataset size: {relative}")
    for condition in CONDITIONS[1:]:
        adapter = project / "ckpt" / f"{condition}_lora.pt"
        require(adapter.is_file(), f"Missing required adapter: {adapter}")
        require(file_hash(adapter) == report["training_histories"][condition]["adapter_sha256"], "Required adapter changed.")
    for package, version in report["provenance"]["packages"].items():
        require(metadata.version(package) == version, f"Package version differs: {package}")
    offsets = sample_offsets(len(manifest["val_indices"]))
    validation_indices = [manifest["val_indices"][offset] for offset in offsets]
    protocol = {
        "version": VERSION, "required_code_hash": report["audit"]["code_hash"],
        "required_split_hash": report["audit"]["split_hash"], "provenance": report["provenance"],
        "extension_source_hashes": {name: file_hash(Path(__file__).parent / name)
                                    for name in ("run_supplement.py", "verify_results.py")},
        "required_adapter_hashes": {c: report["training_histories"][c]["adapter_sha256"] for c in CONDITIONS[1:]},
        "validation_sample_seed": SAMPLE_SEED, "validation_sample_size": SAMPLE_SIZE,
        "validation_offsets": offsets, "validation_official_training_indices": validation_indices,
        "validation_raw_data_hash": digest([official[i] for i in validation_indices]),
        "capacity_condition": {"condition": VARIANT, "rank": 16, "alpha": 32, "alpha_over_rank": 2.0,
                               "dropout": 0.1, "target_modules": "all nn.Linear including lm_head",
                               "seed": 42, "num_epochs": 2, "learning_rate": 1e-5,
                               "train_batch_size": 2, "max_seq_length": 512,
                               "optimizer": "AdamW", "weight_decay": 0.01, "scheduler": "none",
                               "n_train_examples": 6725, "n_loss_validation_examples": 748,
                               "initialization": "Independent adapter on the unchanged frozen base model."},
        "decoding": {"do_sample": False, "batch_size": 8, "validation_budgets": [256, 512],
                     "rank_comparison_budget": 256, "test_budget": 256,
                     "prompt_and_parser": "Unchanged required notebook definitions."},
        "test_policy": "The single fixed rank-16 variant is reported on both benchmarks regardless of its validation result. No test-based setting selection.",
        "required_result_policy": "Read-only; all extension checkpoints, logs and results stay in optional/extensions_v1.",
        "scope": "Single training seed. Capacity comparison keeps alpha/r fixed; it changes rank and alpha jointly, not rank alone.",
    }
    path = output / "protocol.json"
    if path.exists():
        require(read_json(path) == protocol, "Extension protocol/code changed. Preserve this run and use a new versioned directory.")
    else:
        atomic_json(path, protocol)
    atomic_json(output / "main_analysis.json", report)
    snapshot = protected_snapshot(project)
    snapshot_path = output / "required_file_hashes.json"
    if snapshot_path.exists():
        require(read_json(snapshot_path) == snapshot, "A required source/result/adapter changed since preflight.")
    else:
        atomic_json(snapshot_path, snapshot)
    return report, manifest, official, protocol, snapshot


def check_gpu_idle():
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(all(pid == str(os.getpid()) for pid in pids),
            "GPU 0 has another active compute process. Arrange access before starting; no process will be stopped.")


def load_runtime(project, output, report):
    for key, suffix in {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
                        "HF_DATASETS_CACHE": "huggingface/datasets"}.items():
        os.environ[key] = str(project / "cache" / suffix)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for key, suffix in {"TRITON_CACHE_DIR": "triton", "TORCH_HOME": "torch",
                        "XDG_CACHE_HOME": "misc", "CUDA_CACHE_PATH": "cuda"}.items():
        folder = output / "cache" / suffix
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    (output / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(output / "tmp")
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    require(torch.cuda.is_available(), "CUDA is unavailable in the existing task environment.")
    free, total = torch.cuda.mem_get_info(0)
    require(free >= 12 * 1024**3, "Less than 12 GiB GPU memory is free. Check shared server usage first.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    provenance = report["provenance"]
    namespace = dict(torch=torch, nn=nn, Dataset=Dataset, DataLoader=DataLoader, tqdm=tqdm,
                     os=os, json=json, Path=Path, random=random, time=time, gc=gc, re=re,
                     Decimal=Decimal, InvalidOperation=InvalidOperation, Fraction=Fraction,
                     SEED=42, MODEL_ID=provenance["model"], MODEL_REVISION=provenance["model_revision"],
                     CODE_HASH=report["audit"]["code_hash"], SPLIT_HASH=report["audit"]["split_hash"],
                     PACKAGE_VERSIONS=provenance["packages"], hashlib=__import__("hashlib"),
                     CKPT_DIR=str(output / "ckpt"), LOG_DIR=str(output / "logs"))
    for folder in ("ckpt", "logs", "results"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    load_definitions(read_json(project / "math_reasoning.ipynb"), namespace)
    namespace["seed_all"]()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    log("Loading the existing pinned model from the project cache; network downloads are disabled.")
    model = AutoModelForCausalLM.from_pretrained(
        provenance["model"], revision=provenance["model_revision"], local_files_only=True,
        attn_implementation="sdpa", device_map={"": 0}, quantization_config=quant, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(provenance["model"], revision=provenance["model_revision"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    namespace.update(lm=model, tokenizer=tokenizer)
    log(f"GPU: {torch.cuda.get_device_name(0)}; free before model load: {free / 1024**3:.1f} GiB")
    return namespace


def evaluate_optional(namespace, dataset, condition, benchmark, budget, output, protocol_hash, adapter_hash):
    torch, model, tokenizer = namespace["torch"], namespace["lm"], namespace["tokenizer"]
    model.eval()
    tokenizer.padding_side = "left"
    eos_ids = namespace["get_eos_ids"](model, tokenizer)
    stem = f"{condition}_{benchmark}_tokens{budget}"
    progress_path, result_path = output / "logs" / f"{stem}_progress.json", output / "results" / f"{stem}.json"
    config = {"condition": condition, "benchmark": benchmark, "max_new_tokens": budget,
              "do_sample": False, "batch_size": 8, "protocol_hash": protocol_hash,
              "data_hash": digest(dataset.data), "adapter_hash": adapter_hash, "eos_ids": eos_ids,
              "pad_token_id": tokenizer.pad_token_id, "prompt": namespace["INSTRUCTION"]}
    fingerprint = digest(config)
    previous = read_json(result_path) if result_path.exists() else (read_json(progress_path) if progress_path.exists() else None)
    rows, previous_seconds, previous_peak = [], 0.0, 0.0
    if previous:
        require(previous["fingerprint"] == fingerprint and previous["config"] == config, "Optional evaluation protocol differs.")
        rows = previous["predictions"]
        require(len(rows) <= len(dataset), "Optional checkpoint is longer than the dataset.")
        for index, row in enumerate(rows):
            item = dataset.data[index]
            require((row["example_id"], row["source"], row["gold_answer"])
                    == (item.get("example_id", item.get("ID")), item["source"], item["target"]), "Optional checkpoint data differ.")
            answer, value = namespace["extract_final_answer"](row["raw_output"])
            _, gold = namespace["extract_final_answer"](row["gold_answer"])
            require(row["extracted_answer"] == answer and row["extraction_success"] == (value is not None)
                    and row["correct"] == (value is not None and value == gold), "Optional saved scoring differs.")
            require(row["condition"] == condition and row["benchmark"] == benchmark
                    and row["max_new_tokens"] == budget, "Optional checkpoint labels differ.")
            require(output_token_metadata(row["generated_token_ids"], eos_ids, budget)
                    == {key: row[key] for key in output_token_metadata(row["generated_token_ids"], eos_ids, budget)},
                    "Optional token metadata differ.")
        previous_seconds = previous["wall_clock_seconds"]
        previous_peak = previous["peak_gpu_memory_gb"]
        if result_path.exists():
            require(len(rows) == len(dataset) and previous["metrics"] == decoding_summary(rows, namespace["compute_metrics"]),
                    "Completed optional evaluation is inconsistent.")
            log(f"Reusing completed optional evaluation: {stem}")
            return previous
        log(f"Resuming {stem}: {len(rows)}/{len(dataset)}")
    from torch.utils.data import Subset
    loader = namespace["DataLoader"](Subset(dataset, range(len(rows), len(dataset))), batch_size=8,
                                      shuffle=False, collate_fn=namespace["collate_fn"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for batch in namespace["tqdm"](loader, desc=stem):
        encoded = tokenizer(batch["source"], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                       eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
        tails = generated[:, encoded["input_ids"].shape[1]:].cpu().tolist()
        raw_outputs = tokenizer.batch_decode(tails, skip_special_tokens=True)
        for example_id, source, target, raw, ids in zip(batch["example_id"], batch["source"], batch["target"], raw_outputs, tails):
            answer, value = namespace["extract_final_answer"](raw)
            _, gold = namespace["extract_final_answer"](target)
            require(gold is not None, "Invalid optional gold answer.")
            rows.append({"example_id": example_id, "source": source, "raw_output": raw,
                         "extracted_answer": answer, "extraction_success": value is not None,
                         "gold_answer": target, "correct": value is not None and value == gold,
                         "condition": condition, "benchmark": benchmark, "max_new_tokens": budget,
                         **output_token_metadata(ids, eos_ids, budget)})
        torch.cuda.synchronize()
        elapsed = previous_seconds + time.perf_counter() - started
        peak = max(previous_peak, torch.cuda.max_memory_allocated() / 1024**3)
        state = {"fingerprint": fingerprint, "config": config, "predictions": rows,
                 "wall_clock_seconds": elapsed, "peak_gpu_memory_gb": peak}
        atomic_json(progress_path, state)
    torch.cuda.synchronize()
    state = {"fingerprint": fingerprint, "config": config, "predictions": rows,
             "wall_clock_seconds": previous_seconds + time.perf_counter() - started,
             "peak_gpu_memory_gb": max(previous_peak, torch.cuda.max_memory_allocated() / 1024**3)}
    state["metrics"] = decoding_summary(rows, namespace["compute_metrics"])
    require(len(rows) == len(dataset), "Incomplete optional evaluation.")
    atomic_json(result_path, state)
    log(f"Finished {stem}: {json.dumps(state['metrics'])}")
    return state


def train_rank16(namespace, train, validation, output, protocol_hash):
    adapter = output / "ckpt" / f"{VARIANT}_lora.pt"
    history_path = output / "logs" / f"{VARIANT}_train_history.json"
    if adapter.exists() and history_path.exists():
        history = read_json(history_path)
        if history.get("status") == "complete":
            require(history["extension_protocol_hash"] == protocol_hash
                    and history["adapter_sha256"] == file_hash(adapter), "Optional adapter/history differs.")
            log("Reusing the completed rank-16 adapter.")
            return history
    if adapter.exists():
        require((output / "ckpt" / f"{VARIANT}_resume.pt").is_file(),
                "An incomplete optional adapter has no recoverable training checkpoint.")
    namespace["seed_all"]()
    hooks = namespace["inject_lora"](namespace["lm"], target_modules=None, rank=16, alpha=32, dropout=0.1)
    try:
        log("Training rank 16 from the unchanged base model: two epochs, 6,725 training examples.")
        history = namespace["train_lora"](namespace["lm"], namespace["tokenizer"], hooks, train, validation,
                                         num_epochs=2, batch_size=2, learning_rate=1e-5, device="cuda",
                                         max_len=512, condition=VARIANT)
        require(history["lora"]["rank"] == 16 and history["lora"]["alpha"] == 32
                and history["lora"]["dropout"] == 0.1 and history["num_epochs"] == 2
                and history["n_train_examples"] == 6725 and history["n_val_examples"] == 748,
                "Rank-16 training controls differ from the pre-specified comparison.")
        require(history["n_trainable_params"] == 20920320,
                "Rank-16 adapter parameter count is not twice the required adapter count.")
        namespace["save_lora_adapter"](hooks, str(adapter), rank=16, alpha=32, dropout=0.1, target_modules=None)
        history.update(adapter_sha256=file_hash(adapter), extension_protocol_hash=protocol_hash)
        atomic_json(history_path, history)
        return history
    finally:
        namespace["detach_lora"](hooks)
        gc.collect()
        namespace["torch"].cuda.empty_cache()


def smoke_rank16(namespace, train, validation, output, protocol_hash):
    smoke = output / "smoke"
    path = smoke / "smoke_check.json"
    if path.exists():
        evidence = read_json(path)
        require(evidence["protocol_hash"] == protocol_hash and evidence["status"] == "passed", "Optional smoke protocol differs.")
        return
    from torch.utils.data import Subset
    old_paths = {key: namespace[key] for key in ("CKPT_DIR", "LOG_DIR")}
    namespace["seed_all"]()
    hooks = namespace["inject_lora"](namespace["lm"], rank=16, alpha=32, dropout=0.1)
    try:
        for key, suffix in (("CKPT_DIR", "ckpt"), ("LOG_DIR", "logs")):
            folder = smoke / suffix
            folder.mkdir(parents=True, exist_ok=True)
            namespace[key] = str(folder)
        log("Rank-16 preflight: eight training examples and four held-out validation examples.")
        history = namespace["train_lora"](namespace["lm"], namespace["tokenizer"], hooks,
                                         Subset(train, range(8)), Subset(validation, range(4)),
                                         num_epochs=1, batch_size=2, learning_rate=1e-5, device="cuda",
                                         max_len=512, condition="rank16_smoke")
        _, metrics = namespace["evaluate"](namespace["lm"], namespace["tokenizer"], Subset(validation, range(4)),
                                          "rank16_smoke", "gsm8k_validation", 64, 2, "cuda")
        require(all(not h.dropout.training for h in hooks.values()), "LoRA dropout remains active during inference.")
        atomic_json(path, {"status": "passed", "protocol_hash": protocol_hash, "training": history,
                           "inference": metrics, "note": "Preflight only; excluded from experimental scores."})
    finally:
        namespace.update(old_paths)
        namespace["detach_lora"](hooks)
        gc.collect()
        namespace["torch"].cuda.empty_cache()


def run(project, output, report, manifest, official, protocol):
    check_gpu_idle()
    namespace = load_runtime(project, output, report)
    tokenizer, dataset_class = namespace["tokenizer"], namespace["GSM8K_Dataset"]
    train = dataset_class([official[i] for i in manifest["train_indices"]], tokenizer, mode="sft_reasoning", max_len=512)
    validation = dataset_class([official[i] for i in manifest["val_indices"]], tokenizer, mode="sft_reasoning", max_len=512)
    for item, official_index in zip(validation.data, manifest["val_indices"]):
        item["example_id"] = f"gsm8k-training-validation-{official_index:05d}"
    offsets = protocol["validation_offsets"]
    selected = dataset_class([official[manifest["val_indices"][i]] for i in offsets], tokenizer, mode="sft_reasoning", max_len=512)
    for item, offset in zip(selected.data, offsets):
        item["example_id"] = validation.data[offset]["example_id"]
    protocol_hash = digest(protocol)
    smoke_rank16(namespace, train, validation, output, protocol_hash)
    evaluations = {}
    for condition in CONDITIONS:
        hooks = None
        adapter_hash = None
        if condition != "original":
            adapter = project / "ckpt" / f"{condition}_lora.pt"
            adapter_hash = file_hash(adapter)
            hooks = namespace["load_lora_adapter"](namespace["lm"], str(adapter), device="cuda")
        try:
            for budget in (256, 512):
                log(f"Validation generation-budget comparison: {condition}, {budget} tokens, 200 fixed questions.")
                evaluations[(condition, "gsm8k_validation", budget)] = evaluate_optional(
                    namespace, selected, condition, "gsm8k_validation", budget, output, protocol_hash, adapter_hash)
        finally:
            if hooks:
                namespace["detach_lora"](hooks)
            gc.collect()
            namespace["torch"].cuda.empty_cache()
    history = train_rank16(namespace, train, validation, output, protocol_hash)
    hooks = namespace["load_lora_adapter"](namespace["lm"], str(output / "ckpt" / f"{VARIANT}_lora.pt"), device="cuda")
    try:
        evaluations[(VARIANT, "gsm8k_validation", 256)] = evaluate_optional(
            namespace, selected, VARIANT, "gsm8k_validation", 256, output, protocol_hash, history["adapter_sha256"])
        benchmarks = [
            ("gsm8k", dataset_class(read_jsonl(project / "data" / "gsm8k" / "test.jsonl"), tokenizer, mode="default")),
            ("svamp", namespace["SVAMP_Dataset"](read_jsonl(project / "data" / "svamp" / "train.jsonl")
                                                 + read_jsonl(project / "data" / "svamp" / "test.jsonl"), tokenizer)),
        ]
        for benchmark, dataset in benchmarks:
            original_config = read_json(project / "logs" / f"original_{benchmark}_config.json")
            require(digest(dataset.data) == original_config["data_hash"], "Optional test data/prompts differ from the required run.")
            log(f"Final fixed rank-16 evaluation: {benchmark}. These scores are not used to choose training settings.")
            evaluations[(VARIANT, benchmark, 256)] = evaluate_optional(
                namespace, dataset, VARIANT, benchmark, 256, output, protocol_hash, history["adapter_sha256"])
    finally:
        namespace["detach_lora"](hooks)
    summary = {
        "protocol_hash": protocol_hash, "training": history,
        "evaluations": [{"condition": c, "benchmark": b, "max_new_tokens": budget, **result["metrics"],
                         "wall_clock_seconds": result["wall_clock_seconds"],
                         "peak_gpu_memory_gb": result["peak_gpu_memory_gb"]}
                        for (c, b, budget), result in evaluations.items()],
        "budget_comparisons": [{"condition": c, **budget_comparison(
            evaluations[(c, "gsm8k_validation", 256)]["predictions"],
            evaluations[(c, "gsm8k_validation", 512)]["predictions"])} for c in CONDITIONS],
        "capacity_validation_comparison": paired_counts(
            evaluations[("sft_reasoning", "gsm8k_validation", 256)]["predictions"],
            evaluations[(VARIANT, "gsm8k_validation", 256)]["predictions"]),
        "capacity_test_comparisons": [{"benchmark": benchmark, **paired_counts(
            read_jsonl(project / "submission" / f"sft_reasoning_{benchmark}_predictions.jsonl"),
            evaluations[(VARIANT, benchmark, 256)]["predictions"])} for benchmark in ("gsm8k", "svamp")],
        "interpretation_scope": "One training seed. The 200-question validation sample is fixed before these evaluations; its comparisons are descriptive. Token-prefix differences are reported, not silently treated as pure truncation effects.",
    }
    atomic_json(output / "summary.json", summary)
    namespace["torch"].cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    require((project / "math_reasoning.ipynb").is_file(), "Project notebook not found; check --project.")
    output = project / "optional" / VERSION
    output.mkdir(parents=True, exist_ok=True)
    import fcntl  # Linux server: kernel-managed lock releases even if the process exits.
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("An extension check/run is already active in this directory.") from exc
        report, manifest, official, protocol, snapshot = preflight(project, output)
        try:
            if args.check:
                log("SUPPLEMENT CHECK PASSED. No GPU training or inference has started.")
                log(f"Plan: six validation length evaluations, one rank-16 training run, one validation capacity evaluation and two final test evaluations. Output: {output}")
            else:
                atomic_json(output / "status.json", {"status": "running", "protocol_hash": digest(protocol), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                started = time.perf_counter()
                run(project, output, report, manifest, official, protocol)
                check_snapshot(project, snapshot)
                atomic_json(output / "status.json", {"status": "complete", "protocol_hash": digest(protocol),
                                                     "this_invocation_wall_clock_seconds": time.perf_counter() - started,
                                                     "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "required_files_unchanged": True})
                log("SUPPLEMENT RUN COMPLETE. Nine optional evaluations and rank-16 training saved; required files unchanged.")
        except Exception as exc:
            atomic_json(output / "status.json", {"status": "failed", "error": str(exc), "protocol_hash": digest(protocol)})
            raise
        finally:
            check_snapshot(project, snapshot)


if __name__ == "__main__":
    main()
