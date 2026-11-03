#!/usr/bin/env python3
"""Released quantized models (and the regenerated GPTQ baseline) through the task's own verdict pipeline.

    python3 released_baseline.py fetch <task> --release <name> [--reuse DIR]
    python3 released_baseline.py run <task> --release <name> --stage public|hidden --gpu-pool 0,1
    python3 released_baseline.py quantize-gptq <task> [--staging DIR]; move-gptq <task> --staging DIR
    python3 released_baseline.py collect
    python3 released_baseline.py resummarize <record dir>

`fetch` downloads the release's files at the commit `released_models.py` pins (or hard-links
an identical copy found under --reuse) and checks every byte against the release's LFS
sha256. `run` builds the package -- the published files unchanged, the checkpoint's own
config.json and generation_config.json beside a GGUF, and the runtime's inference.py --
and runs the task's evaluator on it in a no-network container on one idle, claimed GPU,
with the Verifier's container limits (8 CPUs, 128 GiB, 16 GiB shm):

  hidden  the hidden verdict, as `evaluate.py run`: check, smoke, PPL, correctness gate,
          decode, score, on the task's tests/ inputs
  public  the public reward check, as `evaluate_public.py`: check, smoke, PPL, decode, score
          on the public inputs

The evaluator is the task's own, file for file, with two gates recorded instead of
enforced, so that every measurement of a model that fails one is still taken: the
own-compression gate (a released quantization is by definition not the submission's
own), and the resident-memory census's limit and foreign-format checks, extended by
`released_memory.py` to the native runtime it cannot read (llama.cpp).
The patch is `CENSUS_PATCH` below; each run records it and every evaluator file's sha256.
The protocol's own verdict (0 when a gate fails) and the measurements are both reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
INSIDE = "--inside" in sys.argv
if not INSIDE:
    REPO = HERE.parents[3]
    sys.path.insert(0, str(HERE.parent))
    import released_models  # noqa: E402

    ROOT = REPO / "compression_bench" / "evolve_space" / "baselines" / "released"
    MODELS = ROOT / "models"
    RECORD = REPO / "evaluation_progress" / "baselines" / "released"
IMAGES = {"llamacpp": "compression_bench/released-llamacpp:0.3.36",
          "vllm-gptq": "compression_bench/platform-gguf:v1"}
HF = "https://huggingface.co"
GIB = 1 << 30

#: The baseline's only change to the evaluator: model_memory.py learns the native runtimes'
#: adapters (released_memory.py) and records, instead of raising, a foreign format and a
#: resident size over the limit. Everything else, and every other file, is the task's own.
CENSUS_PATCH = [
    ('def install_vllm_adapter():\n    """Use a named worker method; do not enable pickle-based RPC serialization."""\n',
     'def _released_memory():\n'
     '    """Baseline adapters for native runtimes (released_memory.py), loaded by absolute path."""\n'
     '    import importlib.util, sys as _sys\n'
     '    name = "compression_bench_released_memory"\n'
     '    if name not in _sys.modules:\n'
     '        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name("released_memory.py"))\n'
     '        module = importlib.util.module_from_spec(spec)\n'
     '        _sys.modules[name] = module\n'
     '        spec.loader.exec_module(module)\n'
     '    return _sys.modules[name]\n\n\n'
     'def install_vllm_adapter():\n    """Use a named worker method; do not enable pickle-based RPC serialization."""\n'
     '    _released_memory().install()\n'),
    ('    if len(llms) != 1:\n        raise ValueError("model-memory unsupported: expose one vLLM LLM on the engine; "\n',
     '    if len(llms) != 1:\n'
     '        native = _released_memory().collect(engine)\n'
     '        if native is not None:\n'
     '            return native\n'
     '        raise ValueError("model-memory unsupported: expose one vLLM LLM on the engine; "\n'),
    ('        foreign = worker.get("foreign_quant_methods") or []\n        if foreign:\n'
     '            raise ValueError("the compression is not the submission\'s own: weights are served by "\n'
     '                             + ", ".join(foreign[:6]))\n',
     '        foreign = worker.get("foreign_quant_methods") or []\n'
     '        # Baseline: recorded, not enforced (a released quantization is never the submission\'s own).\n'),
    ('    result = dict(snapshot, phase=phase, resident_bytes=total,\n'
     '                  resident_gib=total / 2**30, limit_bytes=limit, passed=total <= limit)\n'
     '    if total > limit:\n'
     '        raise ValueError(f"model resident memory exceeds limit after {phase}: "\n'
     '                         f"{total} bytes > {limit} bytes")\n',
     '    foreign = sorted({m for w in workers for m in (w.get("foreign_quant_methods") or [])})\n'
     '    host = sum(int(w.get("host_model_bytes") or 0) for w in workers)\n'
     '    result = dict(snapshot, phase=phase, resident_bytes=total,\n'
     '                  resident_gib=total / 2**30, limit_bytes=limit, passed=total <= limit,\n'
     '                  foreign_quant_methods=foreign, own_compression_passed=not foreign,\n'
     '                  host_model_bytes=host, recorded_not_enforced=True)\n'),
]


def sha256(path: Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str) + "\n")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------- fetch
def release_dir(spec: dict) -> Path:
    return MODELS / spec["repo"].replace("/", "--") / spec["revision"]


def lfs_digests(spec: dict) -> dict:
    sys.path.insert(0, str(HERE.parent))
    import released_identity
    return released_identity.tree(spec["repo"], spec["revision"])


def fetch(task: str, name: str, reuse: list[Path]) -> dict:
    spec = released_models.release(task, name)
    tree = lfs_digests(spec)
    out = release_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in spec["files"]:
        info = tree[path]
        target = out / Path(path).name
        origin = "present"
        if not (target.is_file() and target.stat().st_size == info["size"]):
            target.unlink(missing_ok=True)
            found = [p for d in reuse for p in Path(d).rglob(Path(path).name)
                     if p.is_file() and p.stat().st_size == info["size"]]
            for candidate in found:
                if info["sha256"] is None or sha256(candidate) == info["sha256"]:
                    os.link(candidate, target)
                    origin = f"hard link of a byte-identical copy at {candidate}"
                    break
            else:
                from huggingface_hub import hf_hub_download
                got = hf_hub_download(spec["repo"], path, revision=spec["revision"], local_dir=out / ".download")
                os.replace(got, target)
                origin = "downloaded"
        digest = sha256(target)
        if info["sha256"] is not None and digest != info["sha256"]:
            raise SystemExit(f"{path}: sha256 {digest} is not the release's {info['sha256']}")
        files[path] = {"bytes": target.stat().st_size, "sha256": digest, "lfs_sha256": info["sha256"],
                       "git_oid": info["git_oid"], "origin": origin, "local": str(target.relative_to(REPO))}
        print(f"{path}: {origin}, sha256 verified", flush=True)
    shutil.rmtree(out / ".download", ignore_errors=True)
    provenance = {"schema": "compression-bench-released-provenance-v1", "release": spec, "files": files,
                  "verified_utc": stamp(), "unchanged": all(f["sha256"] == f["lfs_sha256"] for f in files.values()
                                                            if f["lfs_sha256"])}
    write(out / f"PROVENANCE-{name}.json", provenance)
    return provenance


def source_files(task: str) -> Path:
    """The pinned BF16 checkpoint's config.json and generation_config.json (shipped beside a GGUF)."""
    source = released_models.source_of(task)
    out = MODELS / "source" / source["repo"].replace("/", "--") / source["revision"]
    out.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for name in ("config.json", "generation_config.json"):
        if not (out / name).is_file():
            url = f"{HF}/{source['repo']}/resolve/{source['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=120) as r:
                (out / name).write_bytes(r.read())
    return out


# ---------------------------------------------------------------- run
def task_dirs(task: str, stage: str) -> tuple[Path, Path]:
    """(evaluator sources, evaluator assets) of the stage: the hidden Verifier's or the public Checker's."""
    root = released_models.task_dir(task)
    evaluator = root / ("tests" if stage == "hidden" else "environment/public-checker")
    return evaluator, evaluator / "evaluator-assets"


def build_evaluator(task: str, stage: str, dest: Path, engine: str | None = None) -> dict:
    evaluator, _ = task_dirs(task, stage)
    shutil.copytree(evaluator, dest, ignore=shutil.ignore_patterns("__pycache__", "Dockerfile", "docker-compose.yaml",
                                                                    "platform", "test.sh"))
    original = {p.name: sha256(p) for p in sorted(evaluator.glob("*.py"))}
    memory = dest / "model_memory.py"
    text = memory.read_text()
    for old, new in CENSUS_PATCH:
        if text.count(old) != 1:
            raise SystemExit(f"census patch does not apply to {memory}: {old[:60]!r}")
        text = text.replace(old, new)
    memory.write_text(text)
    shutil.copy2(HERE.parent / "released_memory.py", dest / "released_memory.py")
    shutil.copy2(HERE, dest / "released_baseline.py")
    added = ["released_memory.py", "released_baseline.py (driver)"]
    if engine == "vllm-gptq":
        # As baseline_latency.py runs it: the packed plugin installs in every spawned vLLM worker.
        for name in ("gptq_packed_runtime.py", "gptq_codec.py"):
            shutil.copy2(HERE.parent / name, dest / name)
        plugin = dest / "compression_bench_baseline_packed-1.0.dist-info"
        plugin.mkdir()
        (plugin / "METADATA").write_text("Metadata-Version: 2.1\nName: compression-bench-baseline-packed\nVersion: 1.0\n")
        (plugin / "entry_points.txt").write_text("[vllm.general_plugins]\ncompression_bench_baseline_packed = gptq_packed_runtime:install\n")
        added += ["gptq_packed_runtime.py", "gptq_codec.py", "compression_bench_baseline_packed-1.0.dist-info (vLLM plugin entry point)"]
    return {"source": str(evaluator.relative_to(REPO)), "original_sha256": original,
            "used_sha256": {p.name: sha256(p) for p in sorted(dest.glob("*.py"))},
            "changed": ["model_memory.py (CENSUS_PATCH)"], "added": added,
            "assets_sha256": {p.name: sha256(p) for p in sorted((dest / "evaluator-assets").glob("*.json"))}}


def build_submission(task: str, spec: dict, dest: Path) -> dict:
    model = dest / "model"
    model.mkdir(parents=True)
    if spec["engine"] == "vllm-gptq":
        source = REPO / spec["model_dir"]
        for path in sorted(source.rglob("*")):
            if path.is_file():
                (model / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
                os.link(path, model / path.relative_to(source))
        shutil.copy2(REPO / spec["inference"], dest / "inference.py")
        for name in ("gptq_packed_runtime.py", "gptq_codec.py"):
            shutil.copy2(HERE.parent / name, dest / name)
        return json.loads((source.parent / "PROVENANCE-gptq.json").read_text())
    provenance = json.loads((release_dir(spec) / f"PROVENANCE-{spec['name']}.json").read_text())
    for path in spec["files"]:
        os.link(REPO / provenance["files"][path]["local"], model / Path(path).name)
    if spec["format"] == "gguf":
        src = source_files(task)
        for name in ("config.json", "generation_config.json"):
            shutil.copy2(src / name, model / name)
    shutil.copy2(HERE.parent / "released_engines" / f"inference.{spec['engine']}.py", dest / "inference.py")
    return provenance


def cards(pool: list[str], run_id: str, wait_s: float):
    import gptq_baseline
    deadline = time.monotonic() + wait_s
    while True:
        for gpu in pool:
            held = gptq_baseline.Cards(gpu, run_id)
            try:
                held.__enter__()
                return gpu, held
            except SystemExit as exc:
                print(f"GPU {gpu}: {exc}", flush=True)
        if time.monotonic() > deadline:
            raise SystemExit(f"no idle GPU in {pool}")
        time.sleep(60)


def run(task: str, name: str, stage: str, pool: list[str], wait_s: float, timeout_s: float) -> int:
    spec = released_models.release(task, name)
    short = released_models.short(task)
    when = stamp()
    work = ROOT / "work" / f"{short}-{name}-{stage}-{when}"
    record = RECORD / short / name / stage
    work.mkdir(parents=True)
    configuration = {"schema": "compression-bench-released-run-v1", "task": released_models.task_dir(task).name,
                     "release": spec, "stage": stage, "started_utc": when, "work_dir": str(work.relative_to(REPO))}
    summary = {"status": "execution_failed"}
    try:
        configuration["evaluator"] = build_evaluator(task, stage, work / "evaluator", spec["engine"])
        provenance = build_submission(task, spec, work / "submission")
        configuration["provenance"] = provenance
        image = IMAGES[spec["engine"]]
        configuration["image"] = {"tag": image, "id": subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip(),
            "labels": json.loads(subprocess.check_output(["docker", "image", "inspect", image, "--format",
                                                          "{{json .Config.Labels}}"], text=True))}
        (work / "out").mkdir()
        gpu, held = cards(pool, f"released-{short}-{name}-{stage}-{when}", wait_s)
        try:
            container = f"released-{short}-{name}-{stage}-{when}".lower().replace("_", "-").replace(".", "-")
            command = ["docker", "run", "--rm", "--name", container, "--gpus", f'"device={gpu}"', "--network=none",
                       "--cpus", "8", "--memory", "128g", "--shm-size", "16g", "--user", f"{os.getuid()}:{os.getgid()}",
                       "-e", "HOME=/tmp", "-e", "TORCH_DISABLE_NATIVE_JIT=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                       *(["-e", "COMPRESSION_BENCH_BF16_CPU_OFFLOAD_GB=0",
                          "-e", "COMPRESSION_BENCH_PACKED_META=/app/submission/model/gptq_meta.json",
                          "-e", "COMPRESSION_BENCH_PACKED_LOAD_REPORT=/logs/verifier/packed-load.json",
                          "-e", "COMPRESSION_BENCH_PACKED_RESIDENCY_REPORT=/logs/verifier/packed-residency.json"]
                         if spec["engine"] == "vllm-gptq" else []),
                       "-v", f"{work / 'evaluator'}:/tests:ro", "-v", f"{work / 'submission'}:/app/submission:ro",
                       "-v", f"{work / 'out'}:/logs/verifier", "--entrypoint", "python3", image,
                       "/tests/released_baseline.py", "--inside", "--stage", stage]
            configuration.update(gpu_index=gpu, container=container, command=command,
                                 host_gpus=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                                                    "--format=csv,noheader"], text=True).splitlines(),
                                 compute_apps_before=subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                                                              "--format=csv,noheader"], text=True).splitlines())
            write(work / "configuration.json", configuration)
            with (work / "execution.log").open("w") as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "stop", "-t", "10", container], stdout=log, stderr=subprocess.STDOUT)
                    code = "timeout"
        finally:
            held.release()
        configuration.update(finished_utc=stamp(), container_exit=code)
        got = work / "out" / "released-summary.json"
        summary = json.loads(got.read_text()) if got.is_file() else {"status": "execution_failed", "container_exit": code}
    except (Exception, SystemExit) as exc:
        summary.update(error=str(exc), traceback=traceback.format_exc())
    write(work / "configuration.json", configuration)
    archive(work, record, configuration, summary)
    print(json.dumps({"record": str(record.relative_to(REPO)), **{k: summary.get(k) for k in
                      ("status", "measurements", "gates", "protocol_reward", "reward_without_own_compression_gate")}},
                     indent=1, default=str), flush=True)
    return 0 if summary.get("status") == "measured" else 1


def archive(work: Path, record: Path, configuration: dict, summary: dict) -> None:
    """Every stage result and log into evaluation_progress; hidden generations stay out."""
    if record.exists():
        shutil.move(str(record), str(record.with_name(record.name + "-superseded-" + stamp())))
    record.mkdir(parents=True)
    write(record / "configuration.json", configuration)
    for path in sorted((work / "out").glob("*")) if (work / "out").is_dir() else []:
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                value.pop("generations", None)
            write(record / path.name, value)
        elif path.is_file():
            shutil.copy2(path, record / path.name)
    if (work / "execution.log").is_file():
        shutil.copy2(work / "execution.log", record / "execution.log")
    if (work / "submission" / "inference.py").is_file():
        shutil.copy2(work / "submission" / "inference.py", record / "inference.py")
    write(record / "SUMMARY.json", summary)


# ---------------------------------------------------------------- inside the container
def inside(stage: str) -> None:
    os.chdir("/tests")
    sys.path.insert(0, "/tests")
    out = Path("/logs/verifier")
    import own_compression
    import evaluate
    recorded = {}
    real_scan = own_compression.scan

    def scan(submission):
        found = real_scan(submission)
        recorded["own_compression"] = found
        return {**found, "passed": True, "recorded_not_enforced": True}

    own_compression.scan = scan
    assets = Path("/tests/evaluator-assets")
    if stage == "hidden":
        profile = dict(tokenizer=assets / "tokenizer", size_limit_gib=evaluate.DEFAULT_LIMIT_GIB, score=True,
                       ppl_assets=assets, ppl_split="hidden", decode_assets=assets,
                       accuracy_assets=assets)
        public = False
    else:
        import evaluate_public
        profile, public = evaluate_public.load_profile("formal"), True
    summary = {"stage": stage, "status": "execution_failed", "runtime": runtime_versions()}
    try:
        evaluate.evaluate_submission(Path("/app/submission"), profile, public=public, output_dir=out)   # stage files in out
        summary["status"] = "measured"
    except Exception as exc:
        summary.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    stages = {}
    for name in ("check", "smoke", "ppl", "accuracy", "decode", "score"):
        path = out / f"{name}.json"
        if path.is_file():
            stages[name] = json.loads(path.read_text())
    write(out / "own_compression.json", recorded.get("own_compression"))
    summary.update(summarize(stages, recorded.get("own_compression"), stage, float(profile["size_limit_gib"])))
    write(out / "released-summary.json", summary)


def runtime_versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    found = {}
    for package in ("vllm", "torch", "transformers", "llama_cpp_python", "numpy", "safetensors"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    try:
        import torch
        found["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        found["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                                       "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    for path in ("/opt/released/freeze.txt", "/opt/released/base-freeze.txt"):
        if Path(path).is_file():
            found[Path(path).name] = Path(path).read_text().splitlines()
    return found


def summarize(stages: dict, own: dict | None, stage: str, limit_gib: float) -> dict:
    check, ppl, acc, decode, score = (stages.get(k) or {} for k in ("check", "ppl", "accuracy", "decode", "score"))
    memory = [s for r in (ppl, acc, decode) for s in ((r.get("model_memory") or {}).get("snapshots") or [])]
    resident = max((s.get("resident_bytes") or 0 for s in memory), default=None)
    host = max((s.get("host_model_bytes") or 0 for s in memory), default=None)
    limit = limit_gib
    measurements = {
        "package_bytes": check.get("total_bytes"), "package_gib": check.get("total_gib"),
        "resident_model_bytes_max": resident, "resident_model_gib_max": resident / GIB if resident else None,
        "host_model_bytes_max": host, "census_adapter": sorted({w.get("adapter", "vllm-torch-storage-audited-v1")
                                                               for s in memory for w in s.get("workers", [])}),
        "smoke_text": (stages.get("smoke") or {}).get("generated_text"),
        "ppl": ppl.get("ppl"), "bf16_ppl": score.get("bf16_ppl"), "ppl_increase_percent": score.get("ppl_increase_percent"),
        "accuracy": acc.get("accuracy"),
        "accuracy_by_benchmark": {k: {"correct": v.get("correct"), "items": v.get("items"), "accuracy": v.get("accuracy")}
                                  for k, v in (acc.get("benchmarks") or {}).items()},
        "bf16_accuracy": score.get("bf16_accuracy"), "accuracy_retention": score.get("accuracy_retention"),
        "decode_latency_seconds": decode.get("median_round_mean_seconds"),
        "decode_tokens_per_second": decode.get("decode_tokens_per_second"),
        "reward_formula_value": score.get("reward"),
    }
    # Own compression: the package scan's findings and the census's record of who serves the weights
    # (the scan's module list does not name every runtime, e.g. llama_cpp; the census sees the layers).
    served_by = sorted({m for s in memory for m in (s.get("foreign_quant_methods") or [])})
    measurements["own_compression"] = {"package_scan_findings": (own or {}).get("findings"), "weights_served_by": served_by}
    gates = {
        "package_capacity": None if check.get("total_bytes") is None else check["total_bytes"] <= int(limit * GIB),
        "own_compression": None if own is None and not memory else bool(own and own.get("passed")) and not served_by,
        "resident_capacity": None if resident is None else all(s.get("passed") for s in memory),
        "no_cpu_resident_weights": None if host is None else host == 0,
        "smoke": (stages.get("smoke") or {}).get("valid"),
        "ppl_valid": ppl.get("valid"),
        "correctness_gate": score.get("accuracy_gate_pass") if stage == "hidden" else "not part of the public check",
        "decode_complete": decode.get("complete"),
    }
    failed = [k for k, v in gates.items() if v is False]
    missing = [k for k, v in gates.items() if v is None]
    reasons = []
    if gates["own_compression"] is False:
        why = [f"weights served by {', '.join(served_by)}"] if served_by else []
        why += [f"package scan: {f}" for f in ((own or {}).get("findings") or [])[:4]]
        reasons.append("own-compression gate (not the submission's own compression): " + "; ".join(why))
    if gates["resident_capacity"] is False:
        reasons.append(f"resident model storage {resident} bytes > limit {int(limit * GIB)} bytes")
    if gates["no_cpu_resident_weights"] is False:
        reasons.append(f"{host} bytes of weights resident in host memory")
    if gates["correctness_gate"] is False:
        reasons.append(f"correctness gate: retention {score.get('accuracy_retention')} < floor {score.get('accuracy_gate_floor')}")
    return {"measurements": measurements, "gates": gates, "failed_gates": failed, "unmeasured_gates": missing,
            "fail_reasons": reasons,
            "protocol_reward": 0.0 if failed or missing else score.get("reward"),
            "reward_without_own_compression_gate": (0.0 if [k for k in failed if k != "own_compression"] or missing
                                                    else score.get("reward"))}


def quantize_gptq(task: str, pool: list[str], wait_s: float, staging: Path | None = None) -> dict:
    """Regenerate the GPTQ baseline with its recorded recipe and check it against the recorded run.

    With `staging` (e.g. a directory in /dev/shm), the checkpoint is written there while the BF16 source
    occupies the disk, and `move-gptq` brings it to its place later."""
    spec = released_models.gptq(task)
    short = released_models.short(task)
    out_root = staging or released_models.GPTQ_ROOT
    model = out_root / short / "model"
    if not (model / "gptq_meta.json").is_file():
        gpu, held = cards(pool, f"released-gptq-pick-{short}", wait_s)
        held.release()                       # gptq_baseline.py claims the card itself
        log = out_root / short / "quantize.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            # The harness interpreter (env.sh), which gptq_baseline.py needs to resolve the checkpoint.
            subprocess.run([os.environ.get("COMPRESSION_BENCH_PY") or sys.exit("source compression_bench/scripts/env.sh first"),
                            str(HERE.parent / "gptq_baseline.py"), "quantize", task, "--gpu", gpu,
                            "--output-root", str(out_root)], stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not (model.parent / "compact-kept-report.json").is_file():   # also a quantized checkpoint whose compaction failed
        # A parent that imported numpy leaves mkl-service's MKL_THREADING_LAYER=INTEL in the C environment; compact imports
        # torch (libgomp) first and MKL then refuses to load. It only copies bytes, so the threading layer is immaterial.
        subprocess.run([sys.executable, str(HERE.parent / "compact_gptq_kept.py"), str(model)], check=True,
                       env={**os.environ, "MKL_THREADING_LAYER": "GNU"})
    recorded = json.loads((released_models.LATENCY_GPTQ / short / "quantize-report.json").read_text())
    regenerated = json.loads((model.parent / "quantize-report.json").read_text())
    same = {key: recorded.get(key) == regenerated.get(key) for key in ("recipe", "lm_head", "dead_columns", "experts_without_tokens")}
    same["calibration_sha256"] = (recorded.get("calibration") or {}).get("sha256") == (regenerated.get("calibration") or {}).get("sha256")
    for key in ("gptq_shards_bytes", "kept_text_tensors", "kept_text_tensor_bytes"):
        same["package." + key] = (recorded.get("package") or {}).get(key) == (regenerated.get("package") or {}).get(key)
    files = {str(p.relative_to(model)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(model.rglob("*")) if p.is_file()}
    provenance = {"schema": "compression-bench-released-provenance-v1", "release": spec, "files": files,
                  "regenerated_utc": stamp(), "matches_recorded_run": same, "unchanged": all(same.values()),
                  "recorded_quantize_report": str((released_models.LATENCY_GPTQ / short / "quantize-report.json").relative_to(REPO))}
    write(model.parent / "PROVENANCE-gptq.json", provenance)
    return provenance


TABLE_ORDER = ["nemotron3_5_super_vl_120b_a12b_32g", "nemotron3_5_super_vl_120b_a12b_48g", "qwen3_5_122b_a10b_32g",
               "qwen3_5_122b_a10b_48g", "qwen3_8_27b_8g", "qwen3_8_27b_12g"]


def audit_latency_gptq(task: str) -> dict | None:
    """baseline_latency.py's recorded GPTQ run against the task's verdict protocol, without weights: its inputs and
    scoring code byte for byte, the package-stage own-compression scan on its recorded code, its package and
    residency. None when the task has no archived record (archive_baseline_setting.py writes provenance.json)."""
    import importlib.util
    import tempfile
    task_root = released_models.task_dir(task)
    record = released_models.LATENCY_GPTQ / task
    if not (record / "provenance.json").is_file():
        return None
    provenance = json.loads((record / "provenance.json").read_text())
    inputs = {k: sha256(task_root / k) == v for k, v in provenance["benchmark_input_sha256"].items()}
    code = {k: provenance["runtime_sha256"].get(k) == sha256(task_root / "tests" / k)
            for k in ("ppl_benchmark.py", "accuracy_benchmark.py", "decode_benchmark.py", "inference_runtime.py", "model_memory.py")}
    spec = importlib.util.spec_from_file_location("audit_own_compression", task_root / "tests" / "own_compression.py")
    own = importlib.util.module_from_spec(spec)
    writes, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # a task's generated files stay as rendered
    try:
        spec.loader.exec_module(own)
    finally:
        sys.dont_write_bytecode = writes
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        sub = Path(tmp) / "submission"
        (sub / "model").mkdir(parents=True)
        shutil.copy2(record / "inference.py", sub / "inference.py")
        for name in ("gptq_packed_runtime.py", "gptq_codec.py"):
            shutil.copy2(record / "runtime" / name if (record / "runtime" / name).is_file() else HERE.parent / name, sub / name)
        shutil.copy2(record / "model-config.json", sub / "model" / "config.json")
        scan = own.scan(sub)
    package = json.loads((record / "quantize-report.json").read_text())["package"]
    residency = json.loads((record / "packed-residency.json").read_text())
    stored = sum(sum((r.get("parameter_storage_bytes_by_dtype") or {}).values()) for r in residency) if isinstance(residency, list) else None
    rec = json.loads((record / "SUMMARY.json").read_text())
    return {"record": str(record.relative_to(REPO)),
            "inputs_identical_to_current_task": inputs, "scoring_code_identical": code,
            "differences": ["inference_runtime.py: the census off (memory_policy = None), the worker log kept",
                            "stages run: decode, hidden PPL, gate; no package check, own-compression scan or smoke stage"],
            "own_compression_scan_on_recorded_code": scan, "package_total_gib": package.get("total_gib"),
            "packed_residency_parameter_bytes": stored, "packed_residency_parameter_gib": stored / GIB if stored else None,
            "recorded": {"ppl": (rec.get("ppl_verification") or {}).get("ppl"), "accuracy": (rec.get("gate_verification") or {}).get("accuracy"),
                         "retention": (rec.get("gate_verification") or {}).get("retention"),
                         "accuracy_gate_pass": (rec.get("gate_verification") or {}).get("accuracy_gate_pass"),
                         "decode_seconds": rec.get("decode_seconds")}}


def collect() -> dict:
    """RESULTS.json, RESULTS.csv and TABLE.md over every archived hidden verdict, plus the GPTQ checks. Every release
    of a task is a row."""
    import csv
    rows = []
    for task in TABLE_ORDER:
        source = released_models.source_of(task)
        names = list(released_models.RELEASES[task])
        if task in released_models.NO_FIT:
            rows.append({"task": task, "release": None, "kind": "released", "status": "no fitting release",
                         "note": released_models.NO_FIT[task], "source": source})
        for name in names + ["gptq"]:
            record = RECORD / task / name / "hidden"
            audit = audit_latency_gptq(task) if name == "gptq" else None
            if audit:
                write(RECORD / task / "gptq" / "LATENCY_RECORD_AUDIT.json", audit)
            if not (record / "SUMMARY.json").is_file():
                rows.append({"task": task, "release": name, "kind": "gptq" if name == "gptq" else "released",
                             "status": "not measured", "source": source})
                continue
            summary = json.loads((record / "SUMMARY.json").read_text())
            configuration = json.loads((record / "configuration.json").read_text())
            spec = configuration["release"]
            identity = RECORD / "identity" / f"{task}-{name}.json"   # released_identity.py --out
            row = {"task": task, "release": name, "kind": "gptq" if name == "gptq" else "released",
                   "repo": spec.get("repo"), "revision": spec.get("revision"), "files": spec.get("files"),
                   "format": spec.get("format"), "engine": spec.get("engine"), "source": source,
                   "identity": (json.loads(identity.read_text())["identity"] if identity.is_file() else None),
                   "status": summary.get("status"), **summary.get("measurements", {}),
                   "gates": summary.get("gates"), "failed_gates": summary.get("failed_gates"),
                   "fail_reasons": summary.get("fail_reasons"), "protocol_reward": summary.get("protocol_reward"),
                   "reward_without_own_compression_gate": summary.get("reward_without_own_compression_gate"),
                   "image": configuration.get("image"), "gpu_index": configuration.get("gpu_index"),
                   "runtime": summary.get("runtime"), "record": str(record.relative_to(REPO))}
            if name == "gptq":
                row["latency_record_audit"] = audit
                latency = released_models.LATENCY_GPTQ / task / "SUMMARY.json"
                if latency.is_file():
                    t = json.loads(latency.read_text())
                    row["latency_record"] = {"ppl": (t.get("ppl_verification") or {}).get("ppl"),
                                             "accuracy": (t.get("gate_verification") or {}).get("accuracy"),
                                             "decode_seconds": t.get("decode_seconds"),
                                             "record": str(latency.parent.relative_to(REPO))}
                row["regeneration"] = configuration.get("provenance", {}).get("matches_recorded_run")
            rows.append(row)
    result = {"schema": "compression-bench-released-baselines-v1", "collected_utc": stamp(),
              "protocol": "the task's hidden verdict (evaluate.py run), own-compression and census gates recorded, not enforced",
              "rows": rows}
    write(RECORD / "RESULTS.json", result)
    columns = ["task", "kind", "release", "status", "repo", "revision", "package_gib", "resident_model_gib_max",
               "host_model_bytes_max", "ppl", "bf16_ppl", "ppl_increase_percent", "accuracy", "bf16_accuracy",
               "accuracy_retention", "code_correct", "knowledge_correct", "math_correct", "decode_latency_seconds",
               "failed_gates", "protocol_reward", "reward_without_own_compression_gate", "fail_reasons"]
    with (RECORD / "RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            by = row.get("accuracy_by_benchmark") or {}
            writer.writerow({**{k: row.get(k) for k in columns},
                             "code_correct": (by.get("code") or {}).get("correct"),
                             "knowledge_correct": (by.get("knowledge") or {}).get("correct"),
                             "math_correct": (by.get("math") or {}).get("correct"),
                             "failed_gates": ";".join(row.get("failed_gates") or []),
                             "fail_reasons": " | ".join(row.get("fail_reasons") or [])})
    write_table(rows)
    return result


LABEL = {"nemotron3_5_super_vl_120b_a12b_32g": "Nemotron-3.5 Super VL 120B-A12B, 32 GiB",
         "nemotron3_5_super_vl_120b_a12b_48g": "Nemotron-3.5 Super VL 120B-A12B, 48 GiB",
         "qwen3_5_122b_a10b_32g": "Qwen3.5-122B-A10B, 32 GiB", "qwen3_5_122b_a10b_48g": "Qwen3.5-122B-A10B, 48 GiB",
         "qwen3_8_27b_8g": "Qwen3.8-27B, 8 GiB", "qwen3_8_27b_12g": "Qwen3.8-27B, 12 GiB"}


def write_table(rows: list[dict]) -> None:
    def f(x, fmt):
        return "–" if x is None else format(x, fmt)
    lines = ["| Task | Baseline | Package GiB | GPU-resident GiB | Hidden PPL (vs BF16) | Accuracy (code / knowledge / math, of 50) "
             "| Retention | Correctness gate | Decode L (s) | Failed gates | Protocol reward | Reward without the own-compression gate |",
             "|---|---|--:|--:|--:|--:|--:|---|--:|---|--:|--:|"]
    for r in rows:
        if r.get("status") != "measured":
            lines.append(f"| {LABEL[r['task']]} | {r.get('release') or '–'} | | | | | | | | {r.get('note') or r.get('status')} | | |")
            continue
        by = r.get("accuracy_by_benchmark") or {}
        counts = " / ".join(str((by.get(k) or {}).get("correct", "–")) for k in ("code", "knowledge", "math"))
        gate = {True: "pass", False: "fail"}.get((r.get("gates") or {}).get("correctness_gate"), "–")
        name = r["release"] if r["kind"] == "released" else "GPTQ (repo baseline, regenerated)"
        lines.append(f"| {LABEL[r['task']]} | {name} | {f(r.get('package_gib'), '.3f')} | {f(r.get('resident_model_gib_max'), '.3f')} "
                     f"| {f(r.get('ppl'), '.4f')} ({f(r.get('ppl_increase_percent'), '+.2f')}%) | {f(r.get('accuracy'), '.4f')} ({counts}) "
                     f"| {f(r.get('accuracy_retention'), '.3f')} | {gate} | {f(r.get('decode_latency_seconds'), '.3f')} "
                     f"| {', '.join(r.get('failed_gates') or []) or 'none'} | {f(r.get('protocol_reward'), '.3f')} "
                     f"| {f(r.get('reward_without_own_compression_gate'), '.3f')} |")
    (RECORD / "TABLE.md").write_text("# Released and GPTQ baselines: hidden verdicts\n\n" + "\n".join(lines) + "\n\n"
                                     + "The protocol and the gates: released_baseline.py; every field: RESULTS.json.\n")


def resummarize(record: Path) -> dict:
    """SUMMARY.json again from an archived record's stage results (no GPU)."""
    previous = json.loads((record / "SUMMARY.json").read_text())
    stages = {n: json.loads((record / f"{n}.json").read_text())
              for n in ("check", "smoke", "ppl", "accuracy", "decode", "score") if (record / f"{n}.json").is_file()}
    own = json.loads((record / "own_compression.json").read_text()) if (record / "own_compression.json").is_file() else None
    configuration = json.loads((record / "configuration.json").read_text())
    limit = float(released_models.contract_capacity(configuration["task"]))
    summary = {**previous, **summarize(stages, own, previous.get("stage") or configuration["stage"], limit)}
    write(record / "SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("fetch", "run", "resummarize", "quantize-gptq", "move-gptq", "collect"))
    parser.add_argument("--staging", type=Path, help="quantize-gptq/move-gptq: the staging root (e.g. /dev/shm/...)")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--release")
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    parser.add_argument("--stage", choices=("public", "hidden"))
    parser.add_argument("--gpu-pool", default="0")
    parser.add_argument("--wait-for-gpu", type=float, default=6 * 3600)
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    if args.inside:
        inside(args.stage)
        return 0
    if args.command == "fetch":
        fetch(args.task, args.release, args.reuse)
        return 0
    if args.command == "collect":
        got = collect()
        print(json.dumps([{k: r.get(k) for k in ("task", "release", "status", "ppl", "accuracy", "protocol_reward")}
                          for r in got["rows"]], indent=1))
        return 0
    if args.command == "resummarize":
        got = resummarize(Path(args.task))
        print(json.dumps({k: got.get(k) for k in ("gates", "fail_reasons", "protocol_reward",
                                                  "reward_without_own_compression_gate")}, indent=1))
        return 0
    pool = [g for g in args.gpu_pool.split(",") if g]
    if args.command == "move-gptq":
        short = released_models.short(args.task)
        target = released_models.GPTQ_ROOT / short
        if target.exists():
            raise SystemExit(f"{target} exists")
        shutil.move(str(args.staging / short), str(target))
        print(target)
        return 0
    if args.command == "quantize-gptq":
        got = quantize_gptq(args.task, pool, args.wait_for_gpu, args.staging)
        print(json.dumps(got["matches_recorded_run"], indent=1))
        return 0 if got["unchanged"] else 1
    return run(args.task, args.release, args.stage, pool, args.wait_for_gpu, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
