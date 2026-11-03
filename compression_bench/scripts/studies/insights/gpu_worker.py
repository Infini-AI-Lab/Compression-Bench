"""A persistent single-card worker for quant_interventions.py.

    python3 gpu_worker.py --gpu N --queue DIR

Claims card N with the harness's own claim (compression_bench/harness/lifecycle/gpus.py, an flock held for
this process's lifetime), refuses a card with any compute process on it or excluded by the harness, then executes
job files (`DIR/jobs/*.py`, in name order) in one persistent namespace (GPU_INDEX, GPU_UUID, QUEUE, REPO defined), so
the first job loads Qwen3.8-27B BF16 onto the card (quant_interventions.MODEL) and calls quant_interventions.setup,
and later jobs reuse the loaded model. A job's stdout/stderr goes to `DIR/logs/<job>.log`; finished jobs move to
`DIR/done/`. `DIR/STOP` ends the worker (claim released). Nothing else is signalled or touched.
"""
try:                       # pyarrow first: it needs anaconda's libstdc++ before torch loads the system one
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass
import argparse
import contextlib
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, gpus as harness_gpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--queue", required=True)
    a = ap.parse_args()
    gpus = harness_gpus()
    why = gpus.why_busy(a.gpu)
    if why:
        raise SystemExit(why)
    claim = gpus.Claim(a.gpu, run_id=f"analysis-quant-error-{os.getpid()}")
    if not claim.acquire():
        raise SystemExit(f"GPU {a.gpu} is claimed by another run")
    uuid = gpus.sample()[a.gpu]["uuid"]
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    q = Path(a.queue)
    for d in ("jobs", "done", "logs"):
        (q / d).mkdir(parents=True, exist_ok=True)
    (q / "worker.pid").write_text(f"{os.getpid()} gpu={a.gpu} uuid={uuid}\n")
    ns = {"__name__": "worker", "GPU_INDEX": a.gpu, "GPU_UUID": uuid, "QUEUE": q, "REPO": REPO}
    try:
        while not (q / "STOP").exists():
            jobs = sorted((q / "jobs").glob("*.py"))
            if not jobs:
                time.sleep(5)
                continue
            job = jobs[0]
            log = q / "logs" / (job.stem + ".log")
            t0 = time.time()
            with open(log, "w", buffering=1) as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                print(f"[worker] start {job.name} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
                try:
                    code = compile(job.read_text(), str(job), "exec")
                    exec(code, ns)
                    print(f"[worker] OK {job.name} in {time.time() - t0:.1f}s", flush=True)
                except BaseException:  # noqa: BLE001 - a failed job must not end the worker
                    traceback.print_exc()
                    print(f"[worker] FAILED {job.name} after {time.time() - t0:.1f}s", flush=True)
            job.rename(q / "done" / job.name)
    finally:
        claim.release()


if __name__ == "__main__":
    main()
