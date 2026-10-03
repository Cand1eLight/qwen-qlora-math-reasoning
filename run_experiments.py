"""Run a validation-only smoke check or execute the complete course notebook."""
import argparse
import ast
import json
import os
from pathlib import Path
import time


def configure_paths(root):
    os.environ["MATH_WORK_DIR"] = str(root)
    for name, suffix in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HF_DATASETS_CACHE": "huggingface/datasets", "PIP_CACHE_DIR": "pip",
        "TMPDIR": "tmp", "JUPYTER_RUNTIME_DIR": "jupyter/runtime",
        "JUPYTER_DATA_DIR": "jupyter/data", "JUPYTER_CONFIG_DIR": "jupyter/config",
    }.items():
        folder = root / "cache" / suffix
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)


def smoke(root, notebook):
    namespace = {"__name__": "__smoke__"}
    cells = [1, 2, 3, 5, 6, 8, 10, 12, 13, 14, 16, 17, 19, 22, 23, 25, 26]
    for index in cells:
        source = "".join(notebook["cells"][index]["source"])
        print(f"Smoke setup: cell {index}", flush=True)
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
    import torch
    from torch.utils.data import Subset
    # Both the training steps and inference checks use train/validation only.
    namespace["seed_all"]()
    model, tokenizer = namespace["lm"], namespace["tokenizer"]
    hooks = namespace["inject_lora"](model, rank=8, alpha=16, dropout=0.1)
    try:
        namespace["CKPT_DIR"] = str(root / "smoke" / "ckpt")
        namespace["LOG_DIR"] = str(root / "smoke" / "logs")
        Path(namespace["CKPT_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(namespace["LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        train = Subset(namespace["gsm8k_train_dataset_reasoning"], range(8))
        val = Subset(namespace["gsm8k_val_dataset_reasoning"], range(4))
        history = namespace["train_lora"](
            model, tokenizer, hooks, train, val, num_epochs=1, batch_size=2,
            learning_rate=1e-5, device="cuda", max_len=512, condition="smoke",
        )
        predictions, metrics = namespace["evaluate"](
            model, tokenizer, val, "smoke", "gsm8k_validation",
            max_new_tokens=256, batch_size=2, device="cuda",
        )
        assert all(not hook.dropout.training for hook in hooks.values())
        evidence = {
            "status": "passed", "code_hash": namespace["CODE_HASH"],
            "packages": namespace["PACKAGE_VERSIONS"],
            "model_revision": namespace["MODEL_REVISION"],
            "training": history, "validation_inference": metrics,
            "validation_outputs": predictions,
            "note": "Preflight on eight training and four validation examples; not required benchmark results.",
        }
        namespace["atomic_json"](root / "smoke" / "smoke_check.json", evidence)
        print("SMOKE CHECK PASSED. Formal benchmark evaluations have not started.", flush=True)
    finally:
        namespace["detach_lora"](hooks)
        torch.cuda.empty_cache()


def full(root):
    import nbformat
    from nbclient import NotebookClient
    path = root / "math_reasoning.ipynb"
    smoke_path = root / "smoke" / "smoke_check.json"
    if not smoke_path.exists():
        raise RuntimeError("Run --smoke and check its output before starting the full experiment.")
    notebook = nbformat.read(path, as_version=4)
    from hashlib import sha256
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    actual_hash = sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    evidence = json.loads(smoke_path.read_text(encoding="utf-8"))
    if evidence["status"] != "passed" or evidence["code_hash"] != actual_hash:
        raise RuntimeError("Smoke check does not match this notebook. Repeat --smoke.")

    def save(**kwargs):
        temporary = path.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        os.replace(temporary, path)

    def announce(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running code cell {cell_index}", flush=True)

    def finished(cell, cell_index, **kwargs):
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text[-5000:], flush=True)
        save()

    client = NotebookClient(
        notebook, timeout=None, kernel_name="math-reasoning", allow_errors=False,
        resources={"metadata": {"path": str(root)}}, store_widget_state=False,
        on_cell_execute=announce, on_cell_executed=finished,
    )
    try:
        client.execute()
        print("FULL RUN COMPLETE: notebook outputs and required results saved.", flush=True)
    finally:
        save()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    configure_paths(root)
    notebook = json.loads((root / "math_reasoning.ipynb").read_text(encoding="utf-8"))
    # Catch syntax errors before downloading data or allocating GPU memory.
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    smoke(root, notebook) if args.smoke else full(root)


if __name__ == "__main__":
    main()
