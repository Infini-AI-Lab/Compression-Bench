#!/usr/bin/env python3
"""Measure a baseline's decode latency (and hidden PPL and correctness gate) with the task's runtime.

    baseline_latency.py <task> --kind bf16|gptq --gpu N[,M] [--tensor-parallel K] [--record-root DIR]

Decode latency is taken from streaming receipt timestamps: one warm-up request, then three rounds of
three 2048-token prompts, 128 output tokens each; the statistic is the median of the round means of the
last-token minus first-token receipt time. Host entry point; source env.sh before running. Results never
update frozen task baselines. Multi-GPU references use the same protocol with an explicit TP size. Records
go to `evaluation_progress/baselines/latency/<kind>/<setting>/` unless --record-root says otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from contextlib import contextmanager

HERE = Path(__file__).resolve()
if "--inside" not in sys.argv:
    REPO = HERE.parents[3]
    sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "deploy")]
    import gptq_baseline
    import measure_bf16_baselines as bf16


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def cards_for(args, run_id):
    deadline = time.monotonic() + args.wait_for_gpus
    candidates = args.gpu_pool.split(',') if args.gpu_pool else [args.gpu]
    while True:
        last_error = None
        snapshot = (gptq_baseline.Cards(candidates[0], run_id).gpus.sample()
                    if args.gpu_pool else {})
        for selected in candidates:
            cards = gptq_baseline.Cards(selected, run_id)
            observed = snapshot.get(int(selected), {}) if args.gpu_pool else {}
            if observed.get('pids') or observed.get('memory_mib', 0) > cards.gpus.BUSY_MEMORY_MIB:
                last_error = SystemExit(f"GPU {selected} is busy: {observed}")
                print(f"Waiting: {last_error}", flush=True)
                continue
            try:
                cards.__enter__()
                break
            except SystemExit as exc:
                print(f"Waiting for GPU {selected}: {exc}", flush=True)
                last_error = exc
        else:
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(15)
            continue
        break
    try:
        yield selected
    finally:
        cards.release()


def inside(work, timeout):
    sys.path.insert(0, str(work / "evaluator"))
    import decode_benchmark
    from inference_runtime import SubmittedRuntime
    from transformers import AutoTokenizer
    from importlib.metadata import version

    cfg = json.loads((work / "evaluator/evaluator-assets/decode_config.json").read_text())
    prompts = json.loads((work / "evaluator/evaluator-assets/decode_inputs.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(work / "tokenizer", local_files_only=True)
    assert version("vllm") == cfg["vllm_version"]
    assert (cfg["input_tokens"], cfg["output_tokens"], cfg["rounds"], cfg["requests_per_round"]) == (2048, 128, 3, 3)
    assert len(prompts["measured"]) == 3
    assert all(len(ids) == 2048 for ids in [prompts["warmup"], *prompts["measured"]])
    rows = []
    configuration = json.loads((work / "configuration.json").read_text())
    if configuration.get("validate_kernels"):
        with (work / "results/kernel-validation.log").open("w") as log:
            subprocess.run([sys.executable, str(work / "evaluator/validate_packed_kernel.py"),
                            str(work / "results/kernel-validation.json")], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        started = time.perf_counter()
        with SubmittedRuntime(work / "submission", max_model_len=cfg["max_model_len"],
                              max_num_seqs=1, gpu_memory_utilization=cfg["gpu_memory_utilization"],
                              timeout=timeout) as engine:
            startup = time.perf_counter() - started
            warmup = engine.measure_decode(prompts["warmup"], max_new_tokens=128, vocab_size=len(tokenizer))
            write(work / "results/warmup.json", warmup)
            with (work / "results/decode-samples.jsonl").open("x") as log:
                for r in range(3):
                    for p, ids in enumerate(prompts["measured"]):
                        sample = engine.measure_decode(ids, max_new_tokens=128, vocab_size=len(tokenizer))
                        row = {"round": r, "request": p, "input_tokens": len(ids),
                               "output_tokens": len(sample["token_ids"]),
                               **{k: v for k, v in sample.items() if k != "token_ids"}}
                        rows.append(row)
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(f"round {r + 1} request {p + 1}: {row['decode_seconds']:.6f}s", flush=True)
        summary = decode_benchmark.summarize(rows, cfg)
        write(work / "results/decode.json", {"status": "measured", "decode_seconds": summary["median_round_mean_seconds"],
              "engine_startup_seconds_excluded": startup, "vllm_version": version("vllm"),
              "measurement_protocol": cfg["measurement_protocol"], **summary})
    except Exception as exc:
        write(work / "results/decode.json", {"status": "execution_failed", "decode_seconds": None,
              "exception": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
              "completed_requests": len(rows)})
    for filename in ("inference-worker.log", "packed-load.json", "packed-residency.json"):
        path = work / "results" / filename
        if path.exists():
            shutil.copy2(path, path.with_name("decode-" + filename))
    # Decode is already persisted. A failing correctness gate never prevents
    # measuring latency, and every quality stage uses the very same adapter.
    import ppl_benchmark
    import accuracy_benchmark
    for name, measure in (
        ("ppl-hidden", lambda: ppl_benchmark.measure(work / "submission", work / "assets/hidden", work / "tokenizer", SubmittedRuntime, timeout=timeout, expected_split="hidden")),
        ("gate-hidden", lambda: accuracy_benchmark.measure(work / "submission", work / "assets/hidden", SubmittedRuntime, tokenizer_dir=work / "tokenizer", timeout=timeout, expected_split="hidden")),
    ):
        try:
            value = measure()
            value.pop("model_memory", None)
            write(work / "results" / f"{name}.json", value)
        except Exception as exc:
            write(work / "results" / f"{name}.json", {"status": "execution_failed", "error": str(exc), "traceback": traceback.format_exc()})
        for filename in ("inference-worker.log", "packed-load.json", "packed-residency.json"):
            path = work / "results" / filename
            if path.exists():
                shutil.copy2(path, path.with_name(name + "-" + filename))


def host(args):
    task = gptq_baseline.task_dir(args.task)
    contract = gptq_baseline.contract_of(task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = REPO / "compression_bench/evolve_space/baselines/latency" / f"{args.kind}-{task.name}-{stamp}"
    record = (args.record_root or REPO / "evaluation_progress/baselines/latency") / args.kind / task.name.removesuffix("_standard_1xH200NVL")
    for sub in ("evaluator/evaluator-assets", "submission", "results"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    configuration = {"schema": "compression-bench-baseline-latency-v1", "kind": args.kind, "task": task.name,
                     "model": contract["model"], "timestamp_utc": stamp, "gpu_indices": args.gpu,
                     "tensor_parallel_size": args.tensor_parallel, "cpu_offload_gb": args.cpu_offload_gb,
                     "image": args.image, "batch_size": 1, "input_tokens": 2048, "output_tokens": 128,
                     "warmup_requests": 1, "rounds": 3, "requests_per_round": 3,
                     "statistic": "median of three round means of last-token minus first-token receipt time",
                     "excludes": ["prefill", "TTFT", "loading", "quantization"],
                     "enforce_eager": False, "enable_prefix_caching": False, "speculative_config": None,
                     "runtime_source_sha256": digest(REPO / "compression_bench/harness/final_validate/inference_runtime.py"),
                     "validate_kernels": args.validate_kernels,
                     "vllm_enable_v1_multiprocessing": 1,
                     "work_dir": str(work.relative_to(REPO)), "timeout_seconds_per_operation": args.timeout}
    result = {"status": "execution_failed", "decode_seconds": None,
              "ppl_verification": {"status": "not_run", "reason": "checkpoint/runtime unavailable"},
              "gate_verification": {"status": "not_run", "reason": "checkpoint/runtime unavailable"}}
    try:
        configuration["host_gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True).splitlines()
        configuration["image_id"] = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
        ppl_cfg = json.loads((task / "assets/hidden/ppl_config.json").read_text())
        gate_cfg = json.loads((task / "assets/hidden/accuracy_config.json").read_text())
        configuration["quality_reference"] = {"bf16_ppl": ppl_cfg["bf16_ppl"],
            "bf16_accuracy": gate_cfg["bf16"]["accuracy"], "chance": gate_cfg["bf16"]["chance"],
            "max_accuracy_drop": json.loads((task / "tests/evaluator-assets/task_policy.json").read_text())["quality"]["max_accuracy_drop"]}
        if args.model_dir:
            model = Path(args.model_dir).resolve()
        elif args.kind == "gptq":
            model = gptq_baseline.out_dir(task) / "model"
        else:
            lookup = subprocess.run([sys.executable, str(REPO / "compression_bench/scripts/model_for_task.py"), str(task)], capture_output=True, text=True)
            configuration["checkpoint_lookup"] = {"returncode": lookup.returncode, "stdout": lookup.stdout, "stderr": lookup.stderr}
            if lookup.returncode:
                facts = json.loads((task / "assets/model_facts.json").read_text())
                configuration["source_checkpoint_facts"] = {k: v for k, v in facts.items() if k not in ("tensors", "tensor_groups", "config")}
                configuration["filesystem_free_bytes"] = shutil.disk_usage(REPO).free
                raise FileNotFoundError(lookup.stderr.strip())
            model = Path(lookup.stdout.strip()).resolve()
        configuration["model_dir"] = str(model)
        configuration["source_config_sha256"] = digest(model / "config.json")
        if args.kind == "gptq":
            recipe = gptq_baseline.out_dir(task) / "quantize-spec.json"
            configuration["packed_execution"] = {"decode": "packed Triton GEMM; no full BF16 checkpoint",
                "prefill": "above 128 tokens, a temporary per-layer BF16 workspace decoded from packed GPU codes and discarded after GEMM",
                "embedding": "packed row lookup", "persistent_weights": "packed codes, float16 scales; no cached dequantized matrices"}
            configuration["quantization_recipe"] = json.loads(recipe.read_text())
            configuration["quantization_recipe_sha256"] = digest(recipe)
            configuration["packed_format_sha256"] = digest(model / "gptq_meta.json")
        index = model / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            missing = sorted({f for f in mapping.values() if not (model / f).is_file()})
            configuration["missing_indexed_shards"] = missing
            if missing:
                raise FileNotFoundError(f"Checkpoint is incomplete; missing indexed shards: {missing}")
        for name in bf16.EVALUATOR + ("decode_benchmark.py",):
            shutil.copy2(bf16.FINAL_VALIDATE / name, work / "evaluator" / name)
        shutil.copy2(task / "tests/evaluator-assets/task_policy.json", work / "evaluator/evaluator-assets/task_policy.json")
        # A BF16 reference is exempt from the compressed-storage census. This also
        # permits TP references; no change to any compute or streaming operation.
        runtime = work / "evaluator/inference_runtime.py"
        old = "        self.memory_policy = model_memory.read_policy(Path(__file__).resolve().parent)\n"
        source = runtime.read_text()
        assert source.count(old) == 1
        runtime.write_text(source.replace(old, "        self.memory_policy = None\n").replace(
            "self.log = tempfile.TemporaryFile()", "self.log = open('/work/results/inference-worker.log', 'w+b')"))
        starter = REPO / "compression_tasks/_template/starters" / f"inference.{contract['starter']}.py"
        inference = bf16.engine_source(starter, args.tensor_parallel)
        old_executor = 'os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"'
        assert inference.count(old_executor) == 1
        inference = inference.replace(old_executor, 'os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"')
        if args.kind == "gptq":
            # A general plugin installs the adapter in every spawned vLLM worker.
            # It keeps codes/scales on the GPU and decodes only GEMM tiles.
            assert inference.count("        from vllm import LLM\n") == 1
            inference = inference.replace("        from vllm import LLM\n", "        import gptq_packed_runtime\n        gptq_packed_runtime.install()\n        from vllm import LLM\n")
            inference = inference.replace("generation_config='vllm',", "quantization='compression_bench_packed_gptq', generation_config='vllm',")
            end = "cpu_offload_gb=_OFFLOAD_GB, **kwargs)"
            assert inference.count(end) == 1
            inference = inference.replace(end, end + "\n        gptq_packed_runtime.record_residency(self.model)")
            for name in ("gptq_packed_runtime.py", "gptq_codec.py"):
                shutil.copy2(HERE.parent / name, work / "submission" / name)
                shutil.copy2(HERE.parent / name, work / "evaluator" / name)
                configuration[name + "_sha256"] = digest(HERE.parent / name)
            shutil.copy2(HERE.parent / "validate_packed_kernel.py", work / "evaluator/validate_packed_kernel.py")
            plugin = work / "evaluator/compression_bench_baseline_packed-1.0.dist-info"
            plugin.mkdir()
            (plugin / "METADATA").write_text("Metadata-Version: 2.1\nName: compression-bench-baseline-packed\nVersion: 1.0\n")
            (plugin / "entry_points.txt").write_text("[vllm.general_plugins]\ncompression_bench_baseline_packed = gptq_packed_runtime:install\n")
        shutil.copytree(task / "assets/hidden", work / "assets/hidden")
        (work / "submission/inference.py").write_text(inference)
        (work / "submission/model").symlink_to(model)
        shutil.copytree(task / "assets/tokenizer", work / "tokenizer")
        for name in ("decode_config.json", "decode_inputs.json"):
            source = task / "assets/hidden" / name
            shutil.copy2(source, work / "evaluator/evaluator-assets" / name)
            configuration[name + "_sha256"] = digest(source)
        configuration["inference_sha256"] = digest(work / "submission/inference.py")
        shutil.copy2(HERE, work / "runner.py")
        command = ["docker", "run", "--rm", "--name", f"baseline-latency-{args.kind}-{stamp.lower()}-{args.gpu.replace(',', '-')}",
                   "--gpus", f'"device={args.gpu}"', "--ipc=host", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-e", "HOME=/work/cache", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=1",
                   "-e", f"COMPRESSION_BENCH_BF16_CPU_OFFLOAD_GB={args.cpu_offload_gb}", "-v", f"{work}:/work",
                   *gptq_baseline.mounts_for(model), "--entrypoint", "python3", args.image,
                   "/work/runner.py", "--inside", "/work", "--timeout", str(args.timeout)]
        if args.kind == "gptq":
            at = command.index("--entrypoint")
            command[at:at] = ["-e", f"COMPRESSION_BENCH_PACKED_META={model}/gptq_meta.json", "-e", "COMPRESSION_BENCH_PACKED_LOAD_REPORT=/work/results/packed-load.json", "-e", "COMPRESSION_BENCH_PACKED_RESIDENCY_REPORT=/work/results/packed-residency.json"]
        at = command.index("--entrypoint")
        for target in sorted({Path(os.path.realpath(p)).parent for p in model.iterdir() if p.is_symlink()}):
            binding = f"{target}:{target}:ro"
            if binding not in command:
                command[at:at] = ["-v", binding]
        configuration["command"] = command
        write(work / "configuration.json", configuration)
        with cards_for(args, f"baseline-latency-{args.kind}-{stamp}") as selected:
            configuration['gpu_indices'] = selected
            if args.gpu_pool:
                configuration['requested_gpu_pool'] = args.gpu_pool
            command[command.index('--gpus') + 1] = f'"device={selected}"'
            command[command.index('--name') + 1] = f'baseline-latency-{args.kind}-{stamp.lower()}-{selected.replace(",", "-")}'
            write(work / 'configuration.json', configuration)
            with (work / "execution.log").open("w") as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout * 12).returncode
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "stop", "-t", "5", command[command.index("--name") + 1]], stdout=log, stderr=subprocess.STDOUT, timeout=60)
                    raise
        path = work / "results/decode.json"
        if path.exists():
            result = json.loads(path.read_text())
            for key, name in (("ppl_verification", "ppl-hidden"), ("gate_verification", "gate-hidden")):
                p = work / "results" / f"{name}.json"
                result[key] = {"file": p.name, "status": "measured" if p.exists() and json.loads(p.read_text()).get("valid") else "execution_failed"}
                if p.exists():
                    value = json.loads(p.read_text())
                    if value.get("valid"):
                        if name == "ppl-hidden":
                            result[key]["ppl"] = value["ppl"]
                        else:
                            reference = configuration["quality_reference"]
                            retention = (value["accuracy"] - reference["chance"]) / (reference["bf16_accuracy"] - reference["chance"])
                            result[key].update(accuracy=value["accuracy"], retention=retention,
                                               accuracy_gate_pass=retention >= 1 - reference["max_accuracy_drop"])
        else:
            result.update(error=f"container exited {code} before producing timing results", returncode=code)
    except (Exception, SystemExit) as exc:
        result.update(exception=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    write(work / "configuration.json", configuration)
    record.mkdir(parents=True, exist_ok=True)
    for path in [work / "configuration.json", *sorted((work / "results").glob("*")), work / "execution.log", work / "submission/inference.py"]:
        if path.is_file():
            if path.name == "gate-hidden.json":
                value = json.loads(path.read_text())
                value.pop("generations", None)
                write(record / path.name, value)
            else:
                shutil.copy2(path, record / path.name)
    write(record / "SUMMARY.json", {**result, "configuration": "configuration.json"})
    print(json.dumps({"record": str(record), **result}, indent=2), flush=True)
    return 0 if result["status"] == "measured" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--kind", choices=("bf16", "gptq"), default="bf16")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-pool", help="For TP=1, wait for any idle GPU in this comma-separated pool")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--image", default="compression_bench/platform-gguf:v1")
    parser.add_argument("--model-dir")
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--validate-kernels", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--wait-for-gpus", type=float, default=0)
    parser.add_argument("--inside", type=Path)
    args = parser.parse_args()
    if args.gpu_pool and (args.tensor_parallel != 1 or not all(g.isdigit() for g in args.gpu_pool.split(','))):
        parser.error('a GPU pool requires tensor parallelism 1 and GPU indices')
    if args.inside:
        inside(args.inside, args.timeout)
        return 0
    if not args.task or len(args.gpu.split(",")) != args.tensor_parallel:
        parser.error("a task and exactly --tensor-parallel GPUs are required")
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
