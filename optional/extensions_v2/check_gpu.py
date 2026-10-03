"""Server readiness diagnostics; required files and formal runs stay unchanged."""
import argparse
import gc
from pathlib import Path
import shutil
import time

import run_all
from verify_results import atomic_json, check_snapshot, digest, file_hash, require


def scan_dataset(dataset, tokenizer, chunk_size=128):
    """Check the actual tokenizer boundary and every training target before GPU work."""
    lengths = []
    for start in range(0, len(dataset), chunk_size):
        rows = dataset.data[start:start + chunk_size]
        sources = [r["source"] for r in rows]
        texts = [r["source"] + r["target"] + tokenizer.eos_token for r in rows]
        source_ids = tokenizer(sources, padding=False, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(texts, padding=False, add_special_tokens=False)["input_ids"]
        for offset, (source, full) in enumerate(zip(source_ids, full_ids)):
            require(full[:len(source)] == source, f"Tokenizer source/target boundary mismatch at row {start + offset}.")
            require(len(full) - len(source) >= 2, f"No substantive supervised target at row {start + offset}.")
            lengths.append(len(full))
    require(len(lengths) == len(dataset), "Tokenizer scan skipped rows.")
    longest = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))[:4]
    # Include a second set drawn from different positions, retaining a fixed order.
    choices = longest + [i for i in range(len(dataset)) if i not in longest][:4]
    require(len(choices) == 8, "Readiness test requires at least eight examples.")
    for i in choices:
        item = dataset[i]
        require(len(item["input_ids"]) <= dataset.max_len, "Training sequence exceeds its cap.")
        require(len(item["input_ids"]) == len(item["attention_mask"]) == len(item["labels"]), "Training mask lengths differ.")
        require(sum(value != -100 for value in item["labels"]) >= 2, "Training item has no substantive labels.")
    return choices, {"n_examples_checked": len(lengths), "max_full_tokens_before_truncation": max(lengths),
                     "n_over_sequence_cap": sum(v > dataset.max_len for v in lengths), "sample_indices": choices}


def smoke_variant(ns, dataset, choices, name, rank, alpha, output, expected_params=None, device="cuda"):
    torch, model = ns["torch"], ns["lm"]
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ns["seed_all"]()
    hooks = ns["inject_lora"](model, rank=rank, alpha=alpha, dropout=.1)
    optimizer = None
    old_cache = model.config.use_cache
    losses = []
    started = time.perf_counter()
    try:
        params = ns["freeze_and_collect_trainable_params"](model, hooks)
        n_params = sum(p.numel() for p in params)
        if expected_params is not None:
            require(n_params == expected_params, "Unexpected real-model LoRA parameter count.")
        optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
        model.config.use_cache = False
        model.train()
        for start in range(0, len(choices), 2):
            batch = ns["collate_fn"]([dataset[i] for i in choices[start:start + 2]])
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            loss = outputs.loss
            require(bool(torch.isfinite(loss).item()), f"Non-finite readiness loss: {name}")
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in params),
                    f"Missing/non-finite LoRA gradients: {name}")
            optimizer.step()
            require(all(bool(torch.isfinite(p).all().item()) for p in params), f"Non-finite adapter after update: {name}")
            losses.append(float(loss.item()))
        optimizer.zero_grad(set_to_none=True)
        model.config.use_cache = old_cache
        model.eval()
        with torch.inference_mode():
            encoded = ns["tokenizer"]([dataset.data[choices[0]]["source"]], padding=True,
                                      return_tensors="pt", add_special_tokens=False).to(device)
            generated = model.generate(**encoded, max_new_tokens=32, do_sample=False,
                                       eos_token_id=ns["get_eos_ids"](model, ns["tokenizer"]),
                                       pad_token_id=ns["tokenizer"].pad_token_id)
            require(generated.shape[0] == 1 and generated.shape[1] > encoded["input_ids"].shape[1], "Readiness generation returned no new token.")
        require(all(not h.dropout.training for h in hooks.values()), "LoRA dropout remains active during inference.")
        path = output / f"diagnostic_{name}_adapter.pt"
        ns["save_lora_adapter"](hooks, str(path), rank=rank, alpha=alpha, dropout=.1, target_modules=None)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(saved["config"] == {"rank": rank, "alpha": alpha, "dropout": .1, "target_modules": None}, "Saved adapter configuration differs.")
        for layer, hook in hooks.items():
            for field in ("lora_A", "lora_B"):
                require(torch.equal(getattr(hook, field).detach().cpu(), saved["weights"][layer][field]), "Adapter save/read roundtrip differs.")
        peak = torch.cuda.max_memory_allocated() / 1024**3 if use_cuda else 0.0
        return {"status": "passed", "rank": rank, "alpha": alpha, "n_trainable_params": n_params,
                "n_diagnostic_examples": len(choices), "n_updates": len(losses), "finite_batch_losses": losses,
                "adapter_roundtrip_passed": True, "generation_passed": True, "peak_gpu_memory_gb": peak,
                "wall_clock_seconds": time.perf_counter() - started,
                "diagnostic_adapter_sha256": file_hash(path)}
    finally:
        model.config.use_cache = old_cache
        ns["detach_lora"](hooks)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = project / "optional" / "extensions_v2"
    require(Path(__file__).resolve().parent == output, "Extract this diagnostic into optional/extensions_v2 first.")
    require(shutil.disk_usage(project).free >= 5 * 1024**3, "Less than 5 GiB disk space is free on the project filesystem.")
    import fcntl
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("V2 is already running. This diagnostic will not run alongside it.") from exc
        run_all.core.check_gpu_idle()
        report, manifest, official, protocol, variants, snapshot = run_all.prepare(project, output)
        diagnostic = output / "gpu_readiness"
        diagnostic.mkdir(parents=True, exist_ok=True)
        stage = "load_runtime"
        try:
            ns = run_all.core.load_runtime(project, diagnostic, report)
            cls, tok = ns["GSM8K_Dataset"], ns["tokenizer"]
            specs = [("rank16_raw", [official[i] for i in manifest["train_indices"]], 16, 32, 20920320),
                     ("rank8_clean", variants[run_all.VARIANTS[0]], 8, 16, 10460160),
                     ("rank8_mathqa", variants[run_all.VARIANTS[2]][-2000:], 8, 16, 10460160)]
            evidence = []
            for name, raw, rank, alpha, params in specs:
                stage = name
                dataset = cls(raw, tok, mode="sft_reasoning", max_len=512)
                run_all.core.log(f"READINESS {name}: checking all {len(dataset)} training examples with the real tokenizer.")
                choices, token_audit = scan_dataset(dataset, tok)
                result = smoke_variant(ns, dataset, choices, name, rank, alpha, diagnostic, expected_params=params)
                evidence.append({"condition": name, "raw_data_hash": digest(raw), "token_audit": token_audit, **result})
                run_all.core.log(f"READINESS {name} PASSED: finite loss/gradients; adapter roundtrip; peak {result['peak_gpu_memory_gb']:.2f} GiB.")
            check_snapshot(project, snapshot)
            atomic_json(diagnostic / "status.json", {"status": "passed", "protocol_hash": digest(protocol),
                                                     "diagnostic_code_hash": file_hash(Path(__file__)),
                                                     "gpu": ns["torch"].cuda.get_device_name(0), "checks": evidence,
                                                     "required_files_unchanged": True,
                                                     "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "scope": "Readiness only: 12 updates across three disposable adapters, separate from formal experiments. The repeated-GSM control reuses the checked original examples. This is not an accuracy estimate or an unattended-runtime guarantee."})
            run_all.core.log("GPU READINESS CHECK PASSED. Required results unchanged; formal supplementary training has not started.")
        except Exception as exc:
            atomic_json(diagnostic / "status.json", {"status": "failed", "stage": stage, "error": str(exc)})
            raise
        finally:
            check_snapshot(project, snapshot)


if __name__ == "__main__":
    main()
