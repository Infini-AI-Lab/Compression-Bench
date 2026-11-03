#!/usr/bin/env python3
"""Controlled single-decision ablation: the study's shared metadata and the CPU-only paired analysis of its frozen
verifications.

Two pairs of executable artifacts, each built from one selected submission: A the selected setting, B with one
decision ablated. Dense (controlled_build.py): per-group MSE range search disabled in the 3-bit MLPs. MoE
(controlled_moe_build.py): expert-intermediate activation ordering disabled. Pipeline:

    controlled_dense_launcher.py --source RUN_DIR      # build dense_A / dense_B
    controlled_moe_launcher.py --source RUN_DIR        # build moe_A / moe_B (+ source reproduction audit)
    controlled_parallel.py                             # freeze each pair, verify all four on idle cards
    controlled_ablation.py collect                     # paired results, bootstrap intervals, REPORT.md
    controlled_finalize.py                             # completion checks, portable archives

Artifacts live in WORK (compression_bench/evolve_space/studies/controlled_ablation), records and results in OUT
(evaluation_progress/controlled_ablation/decisions), which must hold `preregistration.json` (the design record
written before any verification) before the pairs are frozen. No agent calls, searches, sweeps, or result-dependent
model edits.
"""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import random

from common import REPO, available, read_json, utc_now, write_json

WORK = REPO / 'compression_bench/evolve_space/studies/controlled_ablation'
OUT = REPO / 'evaluation_progress/controlled_ablation/decisions'
NAMES = ['dense_A', 'dense_B', 'moe_A', 'moe_B']
TASK = {'dense': 'qwen3_8_27b_12g_standard_1xH200NVL', 'moe': 'qwen3_5_122b_a10b_48g_standard_1xH200NVL'}
STUDY = Path(__file__).resolve().parent                 # mounted read-only at /study in every study container
IMAGE = 'compression_bench/platform-gguf:v1'            # builds, audits and verifications
OBSERVER = STUDY / 'controlled_verify.py'               # the verification wrapper, hash-pinned at freeze
BENCHMARK = {'math': 'GSM8K', 'code': 'HumanEval', 'knowledge': 'TriviaQA'}


def percentile(v,p):
    x=sorted(v);q=(len(x)-1)*p;i=int(q);f=q-i;return x[i]*(1-f)+x[min(i+1,len(x)-1)]*f


def intervals(pair,seed=20261030,n=10000):
    """Paired percentile bootstrap of B minus A: PPL over the fixed evaluation sequences, accuracy over each
    benchmark's questions (Python's own RNG, no GPU)."""
    rng=random.Random(seed)
    a,b=pair
    wa=a['ppl']['windows'];wb=b['ppl']['windows'];assert [w['block'] for w in wa]==[w['block'] for w in wb]
    assert all(x['scored_tokens']==y['scored_tokens'] for x,y in zip(wa,wb))
    ppl=[];acc=[];per_benchmark={k:[] for k in a['accuracy']['correct_items']}
    ca=a['accuracy']['correct_items'];cb=b['accuracy']['correct_items'];benches=sorted(ca)
    for _ in range(n):
        ix=[rng.randrange(len(wa)) for _ in wa]
        tokens=sum(wa[i]['scored_tokens'] for i in ix)
        ppl.append(math.exp(sum(wb[i]['nll'] for i in ix)/tokens)-math.exp(sum(wa[i]['nll'] for i in ix)/tokens))
        d=[]
        for bench in benches:
            vals=[int(y)-int(x) for x,y in zip(ca[bench],cb[bench])]
            value=sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals)
            d.append(value);per_benchmark[bench].append(100*value)
        acc.append(100*sum(d)/len(d))
    return {'seed':seed,'resamples':n,'sign':'B minus A','ppl_delta_95pct':[percentile(ppl,.025),percentile(ppl,.975)],
            'mean_accuracy_change_pp_95pct':[percentile(acc,.025),percentile(acc,.975)],
            'per_benchmark_accuracy_change_pp_95pct':{k:[percentile(v,.025),percentile(v,.975)] for k,v in per_benchmark.items()},
            'scope':'Paired percentile bootstrap over fixed evaluation sequences and questions, stratified by benchmark; not training-run uncertainty; no additional model calls.'}


def collect():
    """Paired results of the four verifications; refused until all four artifacts are frozen."""
    assert (OUT/'all_four_frozen.json').exists(), 'fresh outcome access embargo: wait for all four artifacts to freeze'
    if not (OUT/'analysis_access.json').exists():write_json(OUT/'analysis_access.json',{'first_collect_utc':utc_now(),'all_four_already_frozen':True})
    collect_pairs(OUT,NAMES,TASK,report)


def collect_pairs(out,names,tasks,write_report):
    """results.json/.csv, paired_results.json/.csv and paired_questions.csv in out, from the verifications of names
    (out/verification/<name>, `<kind>_A` / `<kind>_B` for each kind of tasks), then write_report(rows, pairs)."""
    rows=[];raw={}
    for name in names:
        dest=out/'verification'/name
        if not available(dest/'score.json'):continue
        d={k:read_json(dest/(k+'.json')) for k in ('check','ppl','accuracy','decode','score')};raw[name]=d
        ex=read_json(dest/'verification_execution.json');score=d['score'];acc=d['accuracy'];decode=d['decode']
        resident=max(int(d[k].get('model_memory',{}).get('max_resident_bytes') or 0) for k in ('ppl','accuracy','decode')) or None
        latency=decode['median_round_mean_seconds']
        row={'artifact':name,'task':name.split('_')[0],'verification_ppl':d['ppl']['ppl'],
             'GSM8K_accuracy':acc['benchmarks']['math']['accuracy'],'HumanEval_accuracy':acc['benchmarks']['code']['accuracy'],
             'TriviaQA_accuracy':acc['benchmarks']['knowledge']['accuracy'],'mean_accuracy':acc['accuracy'],
             'package_bytes':d['check']['total_bytes'],'resident_model_bytes':resident,
             'sampled_peak_device_bytes':ex['sampled_peak_device_bytes'],'decode_seconds':latency,'tps':127/latency,
             'accuracy_gate_pass':score.get('accuracy_gate_pass'),'reward':score.get('reward'),
             'verification_valid':score.get('valid'),'quality_pass':score.get('quality_pass'),
             'resident_gate_pass':score.get('model_memory',{}).get('passed'),
             'audit_status':score.get('audit_status'),'score_status':score.get('score_status'),
             'round_tps_min':min(127/x for x in decode['round_mean_seconds']),
             'round_tps_max':max(127/x for x in decode['round_mean_seconds'])}
        rows.append(row)
    write_json(out/'results.json',rows)
    if rows:
        with (out/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    paired=[];questions=[]
    for kind in tasks:
        if not all(kind+'_'+c in raw for c in 'AB'):continue
        a,b=[raw[kind+'_'+c] for c in 'AB'];ra,rb=[next(r for r in rows if r['artifact']==kind+'_'+c) for c in 'AB']
        counts={}
        for bench in a['accuracy']['correct_items']:
            aa=a['accuracy']['correct_items'][bench];bb=b['accuracy']['correct_items'][bench];assert len(aa)==len(bb)==50
            corrected=broken=0
            for i,(x,y) in enumerate(zip(aa,bb)):
                change='corrected_by_ablation' if y and not x else 'newly_broken_by_ablation' if x and not y else 'both_correct' if x else 'both_wrong'
                corrected+=int(y and not x);broken+=int(x and not y)
                questions.append({'task':kind,'benchmark':bench,'question_index':i,'A_correct':x,'B_correct':y,'change':change})
            counts[bench]={'A_correct':sum(aa),'B_correct':sum(bb),'corrected_by_B':corrected,'newly_broken_by_B':broken,'change_pp':100*(sum(bb)-sum(aa))/50,
                'corrected_question_indices':[i for i,(x,y) in enumerate(zip(aa,bb)) if y and not x],
                'newly_broken_question_indices':[i for i,(x,y) in enumerate(zip(aa,bb)) if x and not y]}
        paired.append({'task':kind,'sign':'B ablated minus A selected','ppl_delta':rb['verification_ppl']-ra['verification_ppl'],
            'ppl_relative_change_percent':100*(rb['verification_ppl']/ra['verification_ppl']-1),
            'A_metrics':ra,'B_metrics':rb,
            'mean_accuracy_change_pp':100*(rb['mean_accuracy']-ra['mean_accuracy']),'per_benchmark':counts,
            'package_relative_difference':abs(rb['package_bytes']-ra['package_bytes'])/ra['package_bytes'],
            'decode_seconds_delta':rb['decode_seconds']-ra['decode_seconds'],'tps_delta':rb['tps']-ra['tps'],
            'bootstrap':intervals((a,b))})
    write_json(out/'paired_results.json',paired)
    if paired:
        flat=[]
        for pair in paired:
            r={'task':pair['task'],'ppl_A':pair['A_metrics']['verification_ppl'],'ppl_B':pair['B_metrics']['verification_ppl'],
                'ppl_delta_B_minus_A':pair['ppl_delta'],'ppl_relative_change_percent':pair['ppl_relative_change_percent'],
                'mean_accuracy_change_pp':pair['mean_accuracy_change_pp'],
                'ppl_ci_low':pair['bootstrap']['ppl_delta_95pct'][0],'ppl_ci_high':pair['bootstrap']['ppl_delta_95pct'][1],
                'accuracy_pp_ci_low':pair['bootstrap']['mean_accuracy_change_pp_95pct'][0],
                'accuracy_pp_ci_high':pair['bootstrap']['mean_accuracy_change_pp_95pct'][1],
                'package_relative_difference':pair['package_relative_difference']}
            for bench,v in pair['per_benchmark'].items():
                for key in ('A_correct','B_correct','corrected_by_B','newly_broken_by_B','change_pp'):r[bench+'_'+key]=v[key]
            flat.append(r)
        with (out/'paired_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    if questions:
        with (out/'paired_questions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(questions[0]),lineterminator='\n');w.writeheader();w.writerows(questions)
    write_report(rows,paired)


def report(rows,pairs):
    """OUT/REPORT.md of the decision ablation."""
    lines=['# Controlled single-decision ablation','',
        'Four frozen executable artifacts; no agents, sweeps, or post-verification model selection. Dense: per-group MSE range search disabled only in the 3-bit MLPs. MoE: intermediate-space activation ordering disabled; all routed experts remain 3-bit, so precision allocation is not varied.','',
        'A is the agent-selected setting; B ablates one decision. Final verification is fresh and distinct from design-time reward feedback. The same source weights, calibration tokens, cached/shared Hessians, group sizes, parameter count, runtime code and evaluation assets are fixed within each pair. Each pair is frozen before its verification; independent verifications run on separate idle cards, and all four artifacts are frozen before any outcome is inspected. The interventions are fixed in the preregistration before any verification.','',
        '| Artifact | PPL | GSM8K (%) | HumanEval (%) | TriviaQA (%) | Mean (%) | Package (GiB) | Resident (GiB) | Sampled peak (GiB) | Decode (s) | TPS |',
        '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        gib=lambda k:'unavailable' if r[k] is None else f"{r[k]/2**30:.3f}"
        lines.append(f"| {r['artifact']} | {r['verification_ppl']:.5f} | {100*r['GSM8K_accuracy']:.2f} | {100*r['HumanEval_accuracy']:.2f} | {100*r['TriviaQA_accuracy']:.2f} | {100*r['mean_accuracy']:.2f} | {gib('package_bytes')} | {gib('resident_model_bytes')} | {gib('sampled_peak_device_bytes')} | {r['decode_seconds']:.4f} | {r['tps']:.2f} |")
    audit_path=OUT/'moe_source_reproduction_audit.json'
    if available(audit_path):
        audit=read_json(audit_path)
        lines+=['',f"Selected-design provenance: dense A is the selected submission itself. The reproduced MoE A matches {audit['exact_equal_count']}/{audit['tensor_count']} of the selected submission's weight tensors bit for bit (all equal: {audit['all_exact_equal']})."]
    lines+=['','The raw evaluator marks inference audits pending and scores provisional; passing numeric gates does not substitute for that audit. Resident storage is measured by the official storage census. The total-device peak also includes vLLM’s 0.9 GPU-memory-utilization cache reservation; it is not compressed-weight storage.','',
        '| Artifact | Valid stages | Accuracy gate | Resident gate | Audit / score | TPS range across three rounds |',
        '|---|:--:|:--:|:--:|---|--:|']
    for r in rows:
        lines.append(f"| {r['artifact']} | {r['verification_valid']} | {r['accuracy_gate_pass']} | {r['resident_gate_pass']} | {r['audit_status']} / {r['score_status']} | {r['round_tps_min']:.4f}–{r['round_tps_max']:.4f} |")
    for p in pairs:
        lines+=['',f"## {p['task']}: paired change (B − A)",'',f"PPL: {p['ppl_delta']:+.5f}; mean accuracy: {p['mean_accuracy_change_pp']:+.2f} percentage points. Package difference: {100*p['package_relative_difference']:.4f}%. Bootstrap intervals: PPL [{p['bootstrap']['ppl_delta_95pct'][0]:+.6f}, {p['bootstrap']['ppl_delta_95pct'][1]:+.6f}]; mean accuracy change [{p['bootstrap']['mean_accuracy_change_pp_95pct'][0]:+.2f}, {p['bootstrap']['mean_accuracy_change_pp_95pct'][1]:+.2f}] percentage points.",'',
                '| Benchmark | A correct | B correct | Corrected by B | Newly broken by B |','|---|--:|--:|--:|--:|']
        for bench,c in p['per_benchmark'].items():
            lines.append(f"| {BENCHMARK[bench]} | {c['A_correct']}/50 | {c['B_correct']}/50 | {c['corrected_by_B']} | {c['newly_broken_by_B']} |")
        ci=p['bootstrap']['ppl_delta_95pct']
        if ci[0]>0:likelihood='The paired sequence interval is above zero: the selected setting lowers verification PPL in this fixed comparison.'
        elif ci[1]<0:likelihood='The paired sequence interval is below zero: the ablated setting lowers verification PPL; these data do not support a likelihood benefit from the selected decision.'
        else:likelihood='The paired sequence interval includes zero; these data do not resolve a verification-PPL benefit.'
        aci=p['bootstrap']['mean_accuracy_change_pp_95pct']
        if aci[1]<0:accuracy_finding='The paired accuracy interval is below zero, supporting a selected-setting accuracy benefit on these fixed questions.'
        elif aci[0]>0:accuracy_finding='The paired accuracy interval is above zero, supporting an ablated-setting accuracy benefit on these fixed questions.'
        else:accuracy_finding='The paired accuracy interval includes zero; the observed accuracy difference does not establish a reliable benefit.'
        mechanism=('The intervention isolates per-group MSE range search in low-bit MLPs, with activation-weighted GPTQ compensation and allocation fixed.' if p['task']=='dense' else
                   'The intervention isolates expert-intermediate activation ordering, including its grouping and error-compensation consequences, with precision fixed. It does not measure a precision-allocation effect.')
        lines+=['',mechanism+' '+likelihood+' '+accuracy_finding+' Accuracy and corrected/newly broken questions must be interpreted separately from likelihood. One frozen design/calibration state per condition does not establish an agent-wide or model-family-wide effect.']
    lines+=['','## Measurement limits','',
        'Peak memory is observed total-device usage sampled every 10 ms during the complete verification; brief transients can be missed. It is separate from the official resident-model-storage gate and includes runtime/cache/context overhead. Decode uses batch one, 2048 input tokens and 128 output tokens, the evaluator\'s warmup and 3×3 protocol; TPS counts 127 first-to-last token intervals. Bootstrap intervals resample the fixed 256 verification sequences and questions within each benchmark, not independent quantization seeds. No extra model calls are used for these intervals. Parallel verification uses different physical cards; timing differences also contain between-card/session variation and are not isolated speed effects.','',
        f"Completed full-result artifacts: {len(rows)}/4. Missing measurements are not filled from design-time rewards or other verifications."]
    if available(OUT/'parallel_state.json'):
        state=read_json(OUT/'parallel_state.json')
        if state.get('finished_utc'):
            build_seconds=sum(read_json(OUT/(kind+'_build.json'))['seconds'] for kind in TASK)
            verification_seconds=sum(j['gpu_seconds'] for j in state['jobs'])
            start=read_json(OUT/'dense_build_launch.json')['started_utc']
            elapsed=(datetime.fromisoformat(state['finished_utc'])-datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()/60
            lines+=['',f"Execution: four verifications ran concurrently on separate cards. From the recorded dense-builder launch to the last verifier completion: {elapsed:.1f} minutes. Successful builds plus verification claims account for {(build_seconds+verification_seconds)/3600:.2f} GPU-hours; this excludes CPU audits and staging."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['collect'])
    p.parse_args()
    collect()
