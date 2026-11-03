#!/usr/bin/env python3
"""Three cold-process peak-memory repetitions per unchanged released baseline artifact.

    memory_study.py released     # profile every artifact of BASELINES that has fewer than three repetitions
    memory_study.py summarize    # memory_results.json from the recorded profiles

Released profiles use the pinned published GGUF bytes (released_baseline.py builds the package); this script never
quantizes a checkpoint. Each repetition runs memory_probe.py under Nsight Systems ($NSYS_ROOT, default
/opt/nvidia/nsight-systems/2025.6.3) in the released llama.cpp runtime on its own idle card; the trace's CUDA
allocation lifetimes give each phase's traced peak and llama.cpp's load report its KV / recurrent-state / compute
buffers. Records go to the evidence study's output directory (evidence_studies.py: memory/<task>/<artifact>/rep<N>/),
full traces to its work directory.
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import common
from common import REPO, read_json, utc_now, write_json
from evidence_studies import OUT, WORK
NSYS = Path(os.environ.get("NSYS_ROOT", "/opt/nvidia/nsight-systems/2025.6.3"))
BASELINES = [("qwen3_8_27b_12g", "ista-GSQ-RCO-IQ3_S"), ("qwen3_5_122b_a10b_48g", "unsloth-UD-IQ3_S")]


def trace_summary(sqlite, out):
    c = sqlite3.connect(sqlite)
    phases = c.execute("SELECT e.start,e.end,COALESCE(e.text,s.value) FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id WHERE COALESCE(e.text,s.value) LIKE 'study:%'").fetchall()
    assert len(phases) == 3, phases
    # Only physical device allocations, arrays and device-static allocations.
    # Managed allocations would need residency tracking and are not silently added.
    raw = c.execute("SELECT start,globalPid,deviceId,address,bytes,memKind,memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start").fetchall()
    assert raw, "Nsight produced no memory-allocation events; do not report a peak"
    assert not any(r[5] in (4, 6) for r in raw), "managed memory requires a separate residency analysis"
    live, total = {}, 0
    peaks = {name.removeprefix("study:"): {"cuda_traced_peak_live_bytes": 0, "cuda_traced_live_at_phase_end_bytes": 0} for _, _, name in phases}
    events = []
    for timestamp, pid, device, addr, size, kind, operation in raw:
        if kind not in (2, 3, 5):
            continue
        key = pid, device, addr
        if operation == 0:
            assert key not in live, ("duplicate live allocation", key)
            live[key] = size
            total += size
        elif operation == 1:
            assert key in live, ("unmatched deallocation; incomplete trace", key)
            total -= live.pop(key)
        else:
            raise ValueError(f"unknown memory operation: {operation}")
        events.append({"timestamp_ns": timestamp, "pid": pid, "device": device, "bytes": size, "operation": operation, "live_bytes": total})
        for start, end, name in phases:
            row = peaks[name.removeprefix("study:")]
            if timestamp <= end:
                row["cuda_traced_live_at_phase_end_bytes"] = total
            if start <= timestamp <= end:
                row["cuda_traced_peak_live_bytes"] = max(row["cuda_traced_peak_live_bytes"], total)
    # A phase can allocate nothing: include its persistent starting footprint.
    for start, end, name in phases:
        before = [e["live_bytes"] for e in events if e["timestamp_ns"] < start]
        row = peaks[name.removeprefix("study:")]
        row["cuda_traced_peak_live_bytes"] = max(row["cuda_traced_peak_live_bytes"], before[-1] if before else 0)
    with (out / "cuda_allocation_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(events)
    return {"phases": peaks, "events": len(events), "source": str(sqlite),
            "scope": "live CUDA allocations across all traced processes; includes model/cache/runtime pools and temporary reconstruction, excludes driver/context overhead (NVML sampled separately)"}


def parse_cache(log):
    text = log.read_text(errors="replace")
    entries = re.findall(r"(CUDA\d+) (KV|RS|compute) buffer size\s*=\s*([\d.]+) MiB", text)
    result = {}
    for device, kind, mib in entries:
        result[f"{device}_{kind}_bytes"] = round(float(mib) * 2**20)
    assert any("_KV_" in k for k in result), "native cache allocation was not captured"
    return result


def one(task, name, repetition, package, image):
    destination = OUT / "memory" / task / name / f"rep{repetition}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "profile_summary.json").exists():
        raise RuntimeError("repetition already recorded; refusing accidental extra profile")
    tmp = WORK / "memory" / task / name / f"rep{repetition}"
    tmp.mkdir(parents=True, exist_ok=True)
    gpu, held = common.claim_gpu("study-peak-memory", wait_s=30)
    configuration = {"task": task, "artifact": name, "repetition": repetition, "gpu": gpu,
                     "started_utc": utc_now(), "package": str(package), "kind": "llamacpp",
                     "image": subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip(),
                     "probe_sha256": common.file_sha256(Path(__file__).with_name("memory_probe.py")),
                     "nsys_version": subprocess.check_output(["nsys", "--version"], text=True).strip()}
    uuid = subprocess.check_output(["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    evaluator = REPO / "compression_tasks" / f"{task}_standard_1xH200NVL/tests"
    argv = ["docker", "run", "--rm", "--name", f"study-memory-{gpu}-{os.getpid()}", "--gpus", f"device={gpu}",
            "--network=none", "--cpus", "8", "--memory", "128g", "--shm-size", "16g", "--cap-add=SYS_PTRACE",
            "--security-opt=seccomp=unconfined", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", f"STUDY_GPU_UUID={uuid}", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/study:/evaluator:/submission:/baselines",
            "-v", f"{NSYS}:/nsight:ro", "-v", f"{package}:/submission:ro", "-v", f"{evaluator}:/evaluator:ro",
            "-v", f"{Path(__file__).parent}:/study:ro", "-v", f"{REPO / 'compression_bench/scripts/baselines'}:/baselines:ro",
            "-v", f"{destination}:/result", "-v", f"{tmp}:/traces", "--entrypoint", "/nsight/target-linux-x64/nsys", image,
            "profile", "--trace=cuda,nvtx", "--cuda-memory-usage=true", "--sample=none", "--cpuctxsw=none",
            "--trace-fork-before-exec=true", "--force-overwrite=true", "--output=/traces/profile",
            "python3", "/study/memory_probe.py", "--submission", "/submission", "--evaluator", "/evaluator",
            "--out", "/result"]
    configuration["argv"] = argv
    write_json(destination / "configuration.json", configuration)
    try:
        with (destination / "execution.log").open("w") as log:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        configuration["returncode"] = proc.returncode
        measurement = read_json(destination / "measurement.json") if (destination / "measurement.json").exists() else {"status": "execution_failed"}
        assert proc.returncode == 0 and measurement["status"] == "measured", measurement
        with (tmp / "export.log").open("w") as log:
            subprocess.run(["nsys", "export", "--type=sqlite", "--lazy=false", "--force-overwrite=true", f"--output={tmp / 'profile.sqlite'}", str(tmp / "profile.nsys-rep")], stdout=log, stderr=subprocess.STDOUT, check=True)
        trace = trace_summary(tmp / "profile.sqlite", destination)
        result = {"status": "measured", "task": task, "artifact": name, "repetition": repetition,
                  "measurement": measurement, "cuda_trace": trace,
                  "native_cache_and_compute_buffers": parse_cache(destination / "execution.log")}
        write_json(destination / "profile_summary.json", result)
    except Exception as e:
        write_json(destination / "profile_summary.json", {"status": "execution_failed", "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        held.release()
        configuration["finished_utc"] = utc_now()
        write_json(destination / "configuration.json", configuration)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("released", "summarize"))
    a = p.parse_args()
    sys.path[:0] = [str(REPO / "compression_bench/harness"), str(REPO / "compression_bench/scripts/baselines")]
    if a.command == "released":
        import released_baseline as rb
        import released_models as rm
        for task, name in BASELINES:
            package = WORK / "packages" / task / name
            if not package.exists():
                rb.build_submission(task, rm.release(task, name), package)
            for rep in range(1, 4):
                if not (OUT / "memory" / task / name / f"rep{rep}/profile_summary.json").exists():
                    one(task, name, rep, package, rb.IMAGES["llamacpp"])
    summaries = [read_json(p) for p in sorted((OUT / "memory").glob("*/*/rep*/profile_summary.json"))]
    write_json(OUT / "memory_results.json", {"collected_utc": utc_now(), "profiles": summaries})


if __name__ == "__main__":
    main()
