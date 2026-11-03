#!/usr/bin/env python3
"""Validate the raw streamed samples of every baseline_latency.py record and collect them.

    collect_baseline_latency.py [--root DIR]

Reads `<root>/<kind>/<setting>/` for kind bf16 and gptq and every setting in SETTINGS, checks each record's
samples, quality stages and runtime files, and writes RESULTS.json, VALIDATION.json, RESULTS.csv and
RESULTS.md into the root (default `evaluation_progress/baselines/latency`).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "evaluation_progress/baselines/latency"
SETTINGS = (
    ("Nemotron-3.5 Super VL 120B-A12B", "nemotron3_5_super_vl_120b_a12b", (32, 48)),
    ("Qwen3.5 122B-A10B", "qwen3_5_122b_a10b", (32, 48)),
    ("Qwen3.8 27B", "qwen3_8_27b", (8, 12)),
)


def read(path):
    return json.loads(path.read_text())


def validate(path, summary):
    if summary["status"] != "measured":
        assert summary.get("decode_seconds") is None and summary.get("error"), path
        return {"status": "failure_record_verified", "setting": str(path.relative_to(ROOT))}
    samples = [json.loads(line) for line in (path / "decode-samples.jsonl").read_text().splitlines()]
    assert len(samples) == 9 and {(s["round"], s["request"]) for s in samples} == {(r, p) for r in range(3) for p in range(3)}
    for sample in samples:
        arrivals = sample["arrival_seconds"]
        assert len(arrivals) == 128 and sample["input_tokens"] == 2048
        assert sample["output_tokens"] == 128 and sample["decode_intervals"] == 127
        assert sample["chunk_sizes"] == [1] * 128
        assert arrivals == sample["chunk_arrival_seconds"]
        assert all(math.isfinite(t) and t >= 0 for t in arrivals)
        assert all(a <= b for a, b in zip(arrivals, arrivals[1:]))
        assert math.isclose(sample["decode_seconds"], arrivals[-1] - arrivals[0], abs_tol=1e-9, rel_tol=1e-9)
    means = [statistics.mean(s["decode_seconds"] for s in samples if s["round"] == r) for r in range(3)]
    assert math.isclose(summary["decode_seconds"], statistics.median(means), abs_tol=1e-9, rel_tol=1e-9)
    warmup = read(path / "warmup.json")
    assert len(warmup["token_ids"]) == 128
    return {"status": "samples_verified", "setting": str(path.relative_to(ROOT)),
            "requests": 9, "tokens_per_request": 128, "round_mean_seconds": means,
            "decode_seconds": statistics.median(means), "warmup_tokens": len(warmup["token_ids"])}


def validate_quality(path, summary, config):
    checks = {}
    for key, filename in [('ppl_verification', 'ppl-hidden.json'), ('gate_verification', 'gate-hidden.json')]:
        verification = summary[key]
        checks[key] = verification['status']
        if verification['status'] != 'measured':
            assert (path / filename).is_file(), 'Missing quality execution evidence'
            continue
        value = read(path / filename)
        assert value['valid']
        if key == 'ppl_verification':
            assert value['blocks'] == 256 and value['predicted_tokens'] == 393216
            assert value['ppl'] == verification['ppl']
            assert math.isclose(value['ppl'], math.exp(value['nll'] / value['predicted_tokens']), rel_tol=1e-12)
        else:
            assert value['items'] == 150 and value['accuracy'] == verification['accuracy']
            ref = config['quality_reference']
            retention = (value['accuracy'] - ref['chance']) / (ref['bf16_accuracy'] - ref['chance'])
            assert math.isclose(retention, verification['retention'], abs_tol=1e-12)
            assert verification['accuracy_gate_pass'] == (retention >= 1 - ref['max_accuracy_drop'])
    return checks


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="the records' root (baseline_latency.py --record-root)")
    ROOT = parser.parse_args().root.resolve()
    rows, checks = [], []
    for label, stem, capacities in SETTINGS:
        for capacity in capacities:
            setting = f"{stem}_{capacity}g"
            row = {"model": label, "capacity_gib": capacity, "setting": setting}
            for kind in ("bf16", "gptq"):
                path = ROOT / kind / setting
                summary = read(path / "SUMMARY.json")
                config = read(path / "configuration.json")
                work = REPO / config["work_dir"]
                runtime = path / "runtime"
                runtime.mkdir(exist_ok=True)
                for source in (work / "submission").glob("*.py"):
                    shutil.copy2(source, runtime / source.name)
                for name in ("inference_runtime.py", "ppl_benchmark.py", "accuracy_benchmark.py", "decode_benchmark.py"):
                    source = work / "evaluator" / name
                    if source.is_file():
                        shutil.copy2(source, runtime / name)
                assert config["vllm_enable_v1_multiprocessing"] == 1, "only the standard multiprocess worker execution is collected"
                check = validate(path, summary)
                if summary['status'] == 'measured':
                    check['quality'] = validate_quality(path, summary, config)
                    assert hashlib.sha256((runtime / 'inference.py').read_bytes()).hexdigest() == config['inference_sha256']
                    for name in ('ppl_benchmark.py', 'accuracy_benchmark.py', 'decode_benchmark.py'):
                        assert (runtime / name).read_bytes() == (REPO / 'compression_bench/harness/final_validate' / name).read_bytes()
                    if kind == 'gptq':
                        for name in ('gptq_packed_runtime.py', 'gptq_codec.py'):
                            assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == config[name + '_sha256']
                        for stage in ('decode', 'ppl-hidden', 'gate-hidden'):
                            load = read(path / f'{stage}-packed-load.json')
                            assert load['dequantized_tensors_at_load'] == 0 and load['packed_tensors'] > 0
                            residency = read(path / f'{stage}-packed-residency.json')[0]
                            assert residency['packed_modules'] and not residency['fully_dequantized_checkpoint']
                        check['packed_load_verified'] = True
                if config.get('reused_from'):
                    check['reused_from'] = config['reused_from']
                    proof = read(path / 'reuse-proof.json')
                    assert proof['source'] == config['reused_from']
                checks.append(check)
                row[kind] = {"status": summary["status"], "decode_seconds": summary.get("decode_seconds"),
                             "summary": str((path / "SUMMARY.json").relative_to(ROOT)),
                             "tensor_parallel_size": config["tensor_parallel_size"],
                             "cpu_offload_gb": config["cpu_offload_gb"]}
                row[kind]["ppl_verification"] = summary.get("ppl_verification", {"status": "not_run"})
                row[kind]["gate_verification"] = summary.get("gate_verification", {"status": "not_run"})
                if summary["status"] != "measured":
                    row[kind]["error"] = summary["error"]
                if config.get('reused_from'):
                    row[kind]['reused_from'] = config['reused_from']
                if kind == "gptq":
                    missing = config.get("missing_indexed_shards")
                    if missing:
                        row[kind]["missing_indexed_shards"] = len(missing)
            rows.append(row)
    (ROOT / "RESULTS.json").write_text(json.dumps({"units": "seconds", "settings": rows}, indent=2) + "\n")
    (ROOT / "VALIDATION.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
    with (ROOT / "RESULTS.csv").open("w") as stream:
        writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=("model", "capacity_gib", "bf16_decode_seconds", "bf16_tensor_parallel", "bf16_cpu_offload_gb", "bf16_ppl", "bf16_gate_accuracy", "packed_decode_seconds", "packed_ppl", "packed_gate_accuracy", "packed_gate_pass", "packed_status"))
        writer.writeheader()
        for row in rows:
            native = row["gptq"]
            writer.writerow({"model": row["model"], "capacity_gib": row["capacity_gib"],
                             "bf16_decode_seconds": row["bf16"]["decode_seconds"],
                             "bf16_tensor_parallel": row["bf16"]["tensor_parallel_size"],
                             "bf16_cpu_offload_gb": row["bf16"]["cpu_offload_gb"],
                             "bf16_ppl": row["bf16"]["ppl_verification"].get("ppl"),
                             "bf16_gate_accuracy": row["bf16"]["gate_verification"].get("accuracy"),
                             "packed_decode_seconds": native["decode_seconds"],
                             "packed_ppl": native["ppl_verification"].get("ppl"),
                             "packed_gate_accuracy": native["gate_verification"].get("accuracy"),
                             "packed_gate_pass": native["gate_verification"].get("accuracy_gate_pass"),
                             "packed_status": native["status"]})
    lines = ['# Baseline latency measurements', '',
             'L is the median round-mean first-to-last output-token interval, in seconds. '
             'Each record\'s tensor-parallel size and CPU offload are in RESULTS.json.', '',
             '| Model | GiB | BF16 L | Packed L | BF16 PPL | Packed PPL | BF16 accuracy | Packed accuracy | Accuracy gate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    def number(value, percent=False):
        if value is None:
            return 'execution failure'
        return f'{100 * value:.2f}%' if percent else f'{value:.6f}'
    for row in rows:
        a, b = row['bf16'], row['gptq']
        gate = b['gate_verification'].get('accuracy_gate_pass')
        cells = [row['model'], str(row['capacity_gib']), number(a['decode_seconds']), number(b['decode_seconds']),
                 number(a['ppl_verification'].get('ppl')), number(b['ppl_verification'].get('ppl')),
                 number(a['gate_verification'].get('accuracy'), True), number(b['gate_verification'].get('accuracy'), True),
                 'pass' if gate is True else 'fail' if gate is False else 'execution failure']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    print(f"Validated {len(checks)} setting/reference records; wrote RESULTS.json, RESULTS.csv and RESULTS.md")


if __name__ == "__main__":
    main()
