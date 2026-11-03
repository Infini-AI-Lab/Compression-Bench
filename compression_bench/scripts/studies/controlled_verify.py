#!/usr/bin/env python3
"""One fresh final verification of a frozen study artifact: the unchanged evaluator, plus device-memory sampling
and per-question correctness kept for pairing (never the generated answers).

Run by controlled_parallel.py and moe_gptq_study.py in the runtime image with the artifact at /app/submission, the
task's tests/ at /tests, this directory at /study and the output at /result; STUDY_GPU_UUID names the card to sample.
Writes the evaluator's stage records plus verification_execution.json and question_correctness.csv to /result.
"""
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path[:0]=['/tests','/study']
import evaluate
from memory_probe import NvmlSampler

os.chdir('/tests')
out=Path('/result');out.mkdir(parents=True,exist_ok=True)
assets=Path('/tests/evaluator-assets')
profile=dict(tokenizer=assets/'tokenizer',size_limit_gib=evaluate.DEFAULT_LIMIT_GIB,score=True,
             ppl_assets=assets,ppl_split='hidden',decode_assets=assets,accuracy_assets=assets)
sampler=NvmlSampler(out/'device_memory_samples.jsonl');sampler.phase='final_verification'
started=time.monotonic();failure=None
try:
    evaluate.evaluate_submission(Path('/app/submission'),profile,public=False,output_dir=out)
except Exception as exc:
    failure={'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
finally:
    sampler.close()
    (out/'verification_execution.json').write_text(json.dumps({'seconds':time.monotonic()-started,'failure':failure,
        'sampled_peak_device_bytes':max((r['device_used_bytes'] for r in sampler.rows),default=None),
        'sampling_interval_seconds':.01,'peak_caveat':'Observed total-device peak at 10 ms cadence; sub-sample transients may be missed. Not the official resident-storage gate.'},indent=2))
# The unmodified scorer already records per-question boolean verdicts.
# Save those for pairing; omit hidden generated answer text from portable output.
p=out/'accuracy.json'
if p.exists():
    measured=json.loads(p.read_text());rows=[]
    for bench,verdicts in measured['correct_items'].items():
        assert sum(verdicts)==measured['benchmarks'][bench]['correct']
        for i,ok in enumerate(verdicts):rows.append({'benchmark':bench,'question_index':i,'correct':ok})
    with (out/'question_correctness.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['benchmark','question_index','correct'],lineterminator='\n');w.writeheader();w.writerows(rows)
    measured.pop('generations',None);measured.pop('predictions',None)
    p.write_text(json.dumps(measured,indent=2)+'\n')
if failure:raise RuntimeError(failure['error'])
