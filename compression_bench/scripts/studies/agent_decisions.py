#!/usr/bin/env python3
"""Agent-decision audit of recorded runs and its two-panel figure.

    agent_decisions.py --annotations FILE [--runs FILE] [--ledger FILE] [--pairs FILE] [--out DIR] [--figure-only]

Reads only recorded traces (lineage TRAJECTORY/<run>/steps.jsonl.gz) and public checks; no model execution or search.
Inputs (defaults under evaluation_progress/trajectory_analysis/):
  --runs         the run selection, {"runs": [{"run", "agent", "task", ...}]} (selected_runs.json)
  --ledger       version_ledger.json (compression_bench/scripts/version_ledger.py)
  --pairs        local-vs-public comparison pairs (insights/make_pairs.py output)
  --annotations  the manual review of those runs: run_screen, feedback_exclusions, primary_step_overrides,
                 feedback_actions, failure_exclusions, failure_actions, generation_events, silent_execution_events,
                 and optionally reference_provenance (pair id -> the local validator's reference model) and
                 retry_agents (agents whose checker-failure responses are warmup retries)
Missing observations stay missing. Writes decision_events.json/.csv, trace_evidence.json, run_coverage.json,
snapshot_measurements.json, source_manifest.json, summary.json and agent_decisions.{pdf,svg,png} to --out.
"""
import argparse, csv, gzip, json, re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from common import REPO as ROOT, TRAJECTORY_ANALYSIS as BASE, file_sha256 as sha, read_json as read, run_dir, write_json

OUT = BASE / 'agent_decisions'
AGENT_ORDER = ['opus','sonnet','deepseek','glm','minimax','kimi']
FIGURE_W = 6.75   # inches: full text width, so font sizes are the printed sizes
ACTIONS = ['retain','revert','refine quantization','reallocate precision','change representation','repair execution']

def save(name, obj): write_json(OUT/name, obj, ensure_ascii=False)
def agents_of(runs):
    """Agents in a fixed display order, then any others alphabetically."""
    present = {r['agent'] for r in runs}
    return [a for a in AGENT_ORDER if a in present] + sorted(present - set(AGENT_ORDER))
def stamp(t): return datetime.fromisoformat(t.replace('Z','+00:00'))

def completed(c):
    z=c['response']
    return z.get('valid') is True and isinstance(z.get('public_reward'),(int,float)) and z.get('model_memory',{}).get('passed') is True

def compact_check(c):
    if c is None: return None
    z=c['response']; memory=z.get('model_memory',{})
    fields=('valid','public_only','score_status','public_scoring_version','public_ppl','public_bf16_ppl','public_reward','ppl_increase_percent','total_bytes','total_gib','limit_gib','decode_latency_seconds','error','model_memory_status','size_basis','audit_status')
    return {'utc_start':c['utc_start'],'request':c.get('request'),'seconds':c['seconds'],
      'submission':c.get('submission'), 'response':{k:z[k] for k in fields if k in z} |
      {'model_memory':{k:memory[k] for k in ('protocol','limit_gib','passed','max_resident_bytes') if k in memory}}}
CACHE={}
def load(run):
    if run in CACHE: return CACHE[run]
    p=run_dir(run)
    ident=read(p/'identity.json')
    trace=ROOT/'compression_output'/ident['lineage_id']/'TRAJECTORY'/run/'steps.jsonl.gz'
    with gzip.open(trace,'rt') as f: steps=[json.loads(x) for x in f]
    checks=[json.loads(x) for x in (p/'public-checks/checks.jsonl').read_text().splitlines()]
    CACHE[run]=(p,trace,steps,checks)
    return CACHE[run]

def excerpt(s):
    # Keep original wording and test provenance; never copy hidden generated answers.
    return {k:s[k] for k in ('step','t','kind','text','input') if k in s} | ({'output':s.get('output')} if s.get('kind')=='tool_result' else {})

def cited_ids(source):
    ids=set()
    for m in re.finditer(r'\bsteps?\s+([0-9][0-9/ ,\-]*)',source):
        ids.update(map(int,re.findall(r'\d+',m.group(1))))
    return ids

def build(args):
    selected=read(args.runs)['runs']; main={x['run']:x for x in selected}
    agents=agents_of(selected)
    pairs=read(args.pairs)['pairs']
    ledger={x['run']:x for x in read(args.ledger)['runs']}
    ann=read(args.annotations)
    events=[]; evidence=[]; provenance=[]; coverage=[]
    for run,r in main.items():
        p,tr,steps,checks=load(run)
        hits=[]
        for s in steps:
            t=s.get('text','')
            if s.get('kind') not in ('thinking','text'): continue
            if (re.search(r'math|accuracy|reasoning|generation|humaneval|gsm8k|trivia',t,re.I)
                and re.search(r'bf16|baseline|teacher|unpruned|full.precision',t,re.I)
                and re.search(r'degrad|regress|destroy|worse|drop|wrong|broken|fail|loss|gap|\d+/\d+',t,re.I)
                and re.search(r'\d+\s*[/%%]|\d+\.\d+',t)):
                hits.append(s['step'])
        provenance.append({'run':run,'trace':str(tr.relative_to(ROOT)),'trace_sha256':sha(tr),
                           'checks':str((p/'public-checks/checks.jsonl').relative_to(ROOT)),
                           'checks_sha256':sha(p/'public-checks/checks.jsonl')})
        coverage.append({**r,'generation_screen_hits':len(hits),'screen_steps':hits,
            'generation_screen_disposition':ann['run_screen'].get(r['agent']+' / '+r['task'],'No qualifying paired local deficit found; absolute scores, plans, expectations, and unmatched protocols do not qualify.'),
            'final_verification_context_only':ledger[run]['final']})
    # Full fixed comparison frame, including exclusions and both retained/rejected alternatives.
    for pair in pairs:
        run=Path(pair['run_dir']).name
        p,tr,steps,checks=load(run); look={x['utc_start']:x for x in checks}
        a=look.get(pair['public_utc_A']); b=look.get(pair['public_utc_B'])
        row={'event_id':f"feedback-{pair['id']:02d}",'kind':'local_reward_comparison',
             'run':run,'agent':main[run]['agent'],'task':main[run]['task'],
             'alternatives':{'A':pair['candidate_A'],'B':pair['candidate_B']},
             'local_validation':{k:v for k,v in pair.items() if k.startswith('local_') or k in ('validator_text','change_type','change_type_reason','note')},
             'reward_feedback':{'A':compact_check(a),'B':compact_check(b)},'selection_status':'included',
             'ranking':'reversed' if pair['reversed'] else 'confirmed',
             'candidate_post_design_measurements':{},
             'measurement_warning':'Candidate verification is linked only by exact feedback timestamp and same_bytes=True; run-final context is separate.'}
        metric=pair['local_metric']
        ref=(ann.get('reference_provenance',{}).get(str(pair['id'])) or
             ('Original weights; reconstruction proxy, no output-distribution reference' if pair['validator_text']=='weights_only' else
              'No BF16 reference evaluated' if 'no BF16 reference' in metric else
              'BF16 reference explicitly stated in metric' if 'BF16' in metric else
              'Candidate-only loss/reconstruction proxy, or reference not identified in curated metric; use the trace, not an assumed BF16 teacher'))
        row['local_validation']['reference_model_provenance']=ref
        row['local_validation']['original_metric_description_unmodified']=metric
        ids=cited_ids(pair['local_source'])
        cited=[s for s in steps if s['step'] in ids]
        ex=ann['feedback_exclusions'].get(str(pair['id']))
        if ex: row.update(selection_status='excluded',exclusion_reason=ex)
        if a and b:
            end=max(stamp(x['utc_start'])+timedelta(seconds=x['seconds']) for x in (a,b))
            laststart=max(stamp(x['utc_start']) for x in (a,b))
            nxt=min([stamp(x['utc_start']) for x in checks if stamp(x['utc_start'])>laststart]+[end+timedelta(minutes=30)])
            responses=[s for s in steps if s.get('kind') in ('thinking','text') and s.get('t') and end<=stamp(s['t'])<nxt]
            row['feedback_available_not_before_utc']=end.isoformat()
            row['local_primary_steps']=ann['primary_step_overrides'].get(str(pair['id']),[s['step'] for s in cited])
            primary=[s for s in steps if s['step'] in row['local_primary_steps']]
            row['local_precedes_feedback']=bool(primary) and all(stamp(s['t'])<=end for s in primary)
            if row['selection_status']=='included': assert row['local_precedes_feedback'],pair['id']
            # Equal package bytes are NOT content identity. Timestamp/size associations are
            # exposed as unresolved candidates, never copied as decision measurements.
            row['unresolved_snapshot_associations']={letter:[v['version'] for v in ledger[run]['versions']
                if v.get('reward_feedback') and v['reward_feedback']['utc']==check['utc_start']]
                for letter,check in [('A',a),('B',b)]}
            row['candidate_post_design_measurements']={'A':None,'B':None}
            row['measurement_warning']='Public checks retain names/sizes, not content hashes. Equal byte counts and nearest timestamps do not establish snapshot identity; verification is separately indexed by its actual snapshot.'
            code=ann['feedback_actions'].get(str(pair['id']),{})
            row['subsequent_action']=code
            source_steps=set(code.get('source_steps',[]))
            evidence.append({'event_id':row['event_id'],'trace':str(tr.relative_to(ROOT)),
                'local_source':pair['local_source'],'local_excerpts':[excerpt(s) for s in steps if s['step'] in (ids | set(row['local_primary_steps']))],
                'response_excerpts':[excerpt(s) for s in responses[:4]],
                'annotated_action_excerpts':[excerpt(s) for s in steps if s['step'] in source_steps]})
        events.append(row)
    # Deduplicate the same unordered candidate pair; keep mixed local tests visible.
    groups=defaultdict(list)
    for e in events:
        if e['selection_status']=='included':
            c=e['reward_feedback']; groups[(e['run'],tuple(sorted(c[k]['utc_start'] for k in ('A','B'))))].append(e)
    episodes=[]
    for key,es in groups.items():
        if len({e['ranking'] for e in es})>1:
            for e in es: e.update(selection_status='excluded',exclusion_reason='Conflicting prior local tests for the same candidate pair; retained in audit, omitted from binary ranking counts.')
            continue
        representative=es[0]
        for duplicate in es[1:]: duplicate.update(selection_status='duplicate',exclusion_reason='Same candidate pair; multiple local tests retained but counted once.')
        representative['counted_comparison']=True
        episodes.append(representative)
    # Census of checker failures: bootstrap and incomplete packaging are not agent repairs.
    for run,r in main.items():
        p,tr,steps,checks=load(run); previous=None
        for index,c in enumerate(checks):
            if c['response'].get('valid') is not False: continue
            error=str(c['response'].get('error',''))
            e={'event_id':f'checker-{run}-{index}','kind':'checker_failure', 'run':run,'agent':r['agent'],'task':r['task'],
               'check_index':index,'failure_measurement':compact_check(c),'selection_status':'included'}
            override=ann['failure_exclusions'].get(r['agent']+' / '+r['task']+' / '+str(index))
            if index==0: e.update(selection_status='excluded',exclusion_reason='Initial empty-submission scaffold probe.')
            elif error.startswith(('missing submission/','submission/model/ contains no')): e.update(selection_status='excluded',exclusion_reason='Incomplete package scaffold, before a loadable candidate.')
            elif 'Free memory on device' in error.splitlines()[0] or override: e.update(selection_status='excluded',exclusion_reason=override or 'GPU contention / infrastructure memory availability, not a compression defect.')
            else:
                # Coalesce consecutive invalid checks of the same failure class until a valid check.
                failure_class='storage' if ('too large' in error or 'resident memory exceeds' in error) else 'execution'
                e['failure_class']=failure_class
                if previous is not None and previous['failure_class']==failure_class and not any(completed(x) for x in checks[previous['check_index']+1:index]):
                    e.update(selection_status='duplicate',exclusion_reason='Same unresolved failure episode as '+previous['event_id'])
                else: previous=e
                nxt=next((x for x in checks[index+1:] if completed(x)),None)
                end=stamp(c['utc_start'])+timedelta(seconds=c['seconds'])
                nextstart=stamp(nxt['utc_start']) if nxt else end+timedelta(hours=1)
                response=[s for s in steps if s.get('kind') in ('thinking','text') and s.get('t') and end<=stamp(s['t'])<nextstart]
                code=ann['failure_actions'].get(r['agent']+' / '+r['task'],{})
                e.update(subsequent_action=code,retest=compact_check(nxt),
                  attempted=bool(code),feasible_recovery=bool(nxt and nxt['response'].get('model_memory',{}).get('passed') is True),
                  interpretation='Later successful retest establishes feasible execution after the recorded response, not a single-factor causal effect or a generation-accuracy repair.')
                evidence.append({'event_id':e['event_id'],'trace':str(tr.relative_to(ROOT)),
                    'response_excerpts':[excerpt(s) for s in response[:10]]})
            events.append(e)
    # Pre-specified manually reviewed paired local generation deficits, first per run.
    for spec in ann['generation_events']:
        r=next(x for x in selected if x['agent']==spec['agent'] and x['task']==spec['task'])
        p,tr,steps,checks=load(r['run']); ids=set(spec['source_steps'])
        e={'event_id':'generation-'+r['run'],'kind':'paired_generation_deficit',**spec,'run':r['run'],'selection_status':spec.get('selection_status','included'),
           'post_design_verification_effect':None,'causal_warning':'Local paired candidate scores are observational. Later verification is not the effect of this decision.'}
        if spec.get('gate_after_step'):
            s=next(s for s in steps if s['step']==spec['gate_after_step'])
            e['candidate_feasibility_retest']=next((x for x in checks if x['utc_start']==spec['gate_utc'] and completed(x)),None)
            gate=e['candidate_feasibility_retest']
            e['feasible_improvement']=bool(spec.get('matched_retest_improved') and gate and gate['response'].get('model_memory',{}).get('passed') is True)
            e['candidate_feasibility_retest']=compact_check(gate)
        else: e['feasible_improvement']=False
        evidence.append({'event_id':e['event_id'],'trace':str(tr.relative_to(ROOT)),
                         'local_excerpts':[excerpt(s) for s in steps if s['step'] in ids]})
        events.append(e)
    # One known silent numerical-execution defect: a valid checker response is not correctness proof.
    for spec in ann['silent_execution_events']:
        r=next(x for x in selected if x['agent']==spec['agent'] and x['task']==spec['task'])
        p,tr,steps,checks=load(r['run'])
        e={'event_id':'numerical-'+r['run'],'kind':'numerical_execution_failure',**spec,'run':r['run'],'selection_status':'included'}
        e['candidate_checks']=[compact_check(x) for x in checks if x['utc_start'] in spec['public_utcs']]
        evidence.append({'event_id':e['event_id'],'trace':str(tr.relative_to(ROOT)),
                         'local_excerpts':[excerpt(s) for s in steps if s['step'] in spec['source_steps']]})
        events.append(e)
    matrix={a:{k:{s:0 for s in ('confirmed','reversed')} for k in ACTIONS} for a in agents}
    for e in episodes:
        for action in e.get('subsequent_action',{}).get('actions',[]): matrix[e['agent']][action][e['ranking']]+=1
    gen=[e for e in events if e['kind']=='paired_generation_deficit' and e['selection_status']=='included']
    failures=[e for e in events if e['kind'] in ('checker_failure','numerical_execution_failure') and e['selection_status']=='included']
    response_counts={k:{'attempted':0,'confirmed':0} for k in ('precision','quantization','representation','runtime','storage','retry')}
    for e in gen:
        for category in e['response_categories']:
            response_counts[category]['attempted']+=1
            response_counts[category]['confirmed']+=int(e['feasible_improvement'])
    for e in failures:
        category='storage' if e.get('failure_class')=='storage' else ('retry' if e['agent'] in ann.get('retry_agents',[]) else 'runtime')
        response_counts[category]['attempted']+=int(e['attempted'])
        response_counts[category]['confirmed']+=int(e['feasible_recovery'])
    for r in coverage:
        r['counted_feedback_comparisons']=sum(e['run']==r['run'] for e in episodes)
        r['generation_events']=sum(e['run']==r['run'] for e in gen)
        r['execution_episodes']=sum(e['run']==r['run'] for e in failures)
    summary={'runs':len(main),'agents':agents,'tasks':len({r['task'] for r in selected}),'comparison_frame':len(pairs),
             'comparison_frame_runs':len({Path(p['run_dir']).name for p in pairs}),
             'feedback_comparisons':len(episodes),'feedback_runs':len({e['run'] for e in episodes}),
             'feedback_rank_counts':dict(Counter(e['ranking'] for e in episodes)),
             'feedback_selection_counts':dict(Counter(e['selection_status'] for e in events if e['kind']=='local_reward_comparison')),
             'generation_screen_statements':sum(r['generation_screen_hits'] for r in coverage),
             'paired_generation_observations_reviewed':sum(e['kind']=='paired_generation_deficit' for e in events),
             'generation_deficits':len(gen),'generation_attempted':sum(e['attempted'] for e in gen),
             'generation_matched_improved':sum(e['matched_retest_improved'] for e in gen),
             'generation_feasible_improved':sum(e['feasible_improvement'] for e in gen),
             'generation_closed_deficit':sum(e.get('deficit_closed',False) for e in gen),
             'failure_response_counts':response_counts,
             'execution_episodes':len(failures),'execution_runs':len({e['run'] for e in failures}),
             'execution_attempted':sum(e['attempted'] for e in failures),
             'execution_feasible_recovery':sum(e['feasible_recovery'] for e in failures),
             'matrix':matrix,'by_agent':{},'by_task':{}}
    for dimension,values in [('agent',agents),('task',sorted({r['task'] for r in selected}))]:
        for value in values:
            fs=[e for e in episodes if e[dimension]==value];gs=[e for e in gen if e[dimension]==value];xs=[e for e in failures if e[dimension]==value]
            summary['by_'+dimension][value]={'screened_runs':sum(r[dimension]==value for r in selected),'feedback_pairs':len(fs),'feedback_runs':len({e['run'] for e in fs}),'confirmed':sum(e['ranking']=='confirmed' for e in fs),'reversed':sum(e['ranking']=='reversed' for e in fs),'generation_deficits':len(gs),'execution_episodes':len(xs)}
    save('snapshot_measurements.json',{'about':'Measurements belong to the explicitly named snapshot. Public-check associations based on equal package size do not establish candidate identity; no effect of an individual decision is inferred.', 'runs':[{'run':r['run'],'agent':r['agent'],'task':r['task'],'versions':ledger[r['run']]['versions']} for r in selected]})
    save('decision_events.json',events);save('trace_evidence.json',evidence);save('run_coverage.json',coverage);save('source_manifest.json',provenance);save('summary.json',summary)
    with (OUT/'decision_events.csv').open('w',newline='') as f:
        first=['event_id','kind','run','agent','task','selection_status','exclusion_reason']
        fields=first+sorted(set().union(*(e.keys() for e in events))-set(first))
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for e in events: w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in e.items()})
    return summary

def figure(s):
    """White-background data figure in the shared analysis-figure style (analysis_figures.py, findings_figures.py)."""
    import sys
    sys.path.insert(0,str(ROOT/'compression_bench/scripts'))
    import analysis_figures as af
    import findings_figures as findings
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    findings.style()
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
    fig=plt.figure(figsize=(FIGURE_W,3.05))
    ax=fig.add_axes([.14,.30,.36,.57]);ax2=fig.add_axes([.785,.30,.19,.57])
    colors=['#0F4D92','#3775BA','#8DB3DD','#7B5EB2','#E4DAF2']
    order=s['agents'];actions=ACTIONS[:5]
    n=len(order)
    xmax=max([1]+[sum(s['matrix'][a][k][r] for k in actions) for a in order for r in ('confirmed','reversed')])
    for i,agent in enumerate(order):
        for ranking,dy in [('confirmed',-.16),('reversed',.16)]:
            left=0
            for j,action in enumerate(actions):
                count=s['matrix'][agent][action][ranking]
                if count:
                    ax.barh(i+dy,count,left=left,height=.25,color=colors[j],
                            edgecolor='white' if ranking=='confirmed' else af.C['ink'],
                            linewidth=.35,hatch='///' if ranking=='reversed' else None)
                left+=count
            ax.text(left+.5,i+dy,str(left),va='center',fontsize=6)
    ticks=[af.AGENT.get(a,a)+f"\nn={s['by_agent'][a]['feedback_pairs']}; {s['by_agent'][a]['feedback_runs']}/{s['by_agent'][a]['screened_runs']} runs" for a in order]
    ax.set_yticks(range(n),ticks,fontsize=6)
    ax.set_ylim(n-.4,-.6);ax.set_xlim(0,xmax*1.15+1)
    ax.set_xlabel('coded responses (counts)');ax.grid(axis='y',visible=False)
    ax.tick_params(axis='y',length=0,pad=4)
    findings.panel(ax,'a')
    ax.text(.10,1.045,f"{s['feedback_comparisons']} pairs; {s['feedback_runs']}/{s['runs']} runs",transform=ax.transAxes,
            fontsize=6,color=af.C['gray'],va='bottom')
    categories=['precision','quantization','representation','runtime','storage','retry']
    names=['precision allocation','quantization / distillation','representation','runtime / interface',
           'storage-budget repair','warmup retry']
    ys=[0,1,2,3.4,4.4,5.4]
    x2max=max([1]+[s['failure_response_counts'][k]['attempted'] for k in categories])
    for y,key in zip(ys,categories):
        row=s['failure_response_counts'][key];attempt=row['attempted'];confirmed=row['confirmed']
        ax2.plot([confirmed,attempt],[y,y],color=af.C['gray'],lw=.9,zorder=2)
        ax2.scatter(attempt,y,marker='o',s=26,facecolor='white',edgecolor=af.C['ink'],linewidth=.8,zorder=4)
        ax2.scatter(confirmed,y,marker='D',s=14,color=af.C['indigo'],edgecolor='white',linewidth=.3,zorder=5)
        if attempt==confirmed:
            ax2.text(attempt+.25,y,str(attempt),va='center',fontsize=6)
        else:
            ax2.text(attempt+.24,y-.12,str(attempt),va='center',fontsize=6)
            ax2.text(confirmed-.24,y+.20,str(confirmed),ha='right',va='center',fontsize=6,color=af.C['indigo'])
    ax2.set_yticks(ys,names);ax2.set_ylim(6,-.6)
    ax2.set_xlim(-.2,x2max+1);ax2.set_xlabel('response episodes (counts)')
    ax2.tick_params(axis='y',length=0,pad=4);ax2.grid(axis='y',visible=False)
    findings.panel(ax2,'b')
    ax2.text(.16,1.045,f"{s['generation_deficits']} generation / {s['execution_episodes']} execution",transform=ax2.transAxes,
             fontsize=6,color=af.C['gray'],va='bottom')
    ranks=s['feedback_rank_counts']
    ax.legend(handles=[Patch(facecolor=af.C['light'],edgecolor='white',label=f"ranking confirmed ({ranks.get('confirmed',0)})"),
                       Patch(facecolor='white',edgecolor=af.C['ink'],hatch='///',label=f"ranking reversed ({ranks.get('reversed',0)})")],
              loc='upper left',bbox_to_anchor=(-.01,-.25),fontsize=6,handlelength=1.3,borderaxespad=0,labelspacing=.35)
    ax2.legend(handles=[Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor=af.C['ink'],
                                color='none',markersize=4,label='correction attempted'),
                        Line2D([],[],marker='D',markerfacecolor=af.C['indigo'],markeredgecolor='white',
                                color='none',markersize=4,label='retested gain / feasible recovery')],
               loc='upper left',bbox_to_anchor=(-.63,-.25),fontsize=6,handlelength=1.1,borderaxespad=0,labelspacing=.35)
    fig.legend(handles=[Patch(facecolor=c,edgecolor='none',label=t) for c,t in zip(colors,
                    ['retain','revert','refine quantization','reallocate precision','change representation'])],
               loc='lower center',bbox_to_anchor=(.50,-.015),ncol=5,fontsize=6,
               columnspacing=1.3,handlelength=1.2,handletextpad=.35)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'agent_decisions.{ext}',dpi=300,bbox_inches='tight',pad_inches=.02)
    svg=OUT/'agent_decisions.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs',default=str(BASE/'selected_runs.json'))
    parser.add_argument('--ledger',default=str(BASE/'version_ledger.json'))
    parser.add_argument('--pairs',help='local-vs-public comparison pairs (insights/make_pairs.py output)')
    parser.add_argument('--annotations',help='the manual review of the runs (format above)')
    parser.add_argument('--out',default=str(OUT))
    parser.add_argument('--figure-only',action='store_true')
    args=parser.parse_args()
    OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
    if not args.figure_only and not (args.annotations and args.pairs):
        parser.error('--pairs and --annotations are required to build the audit')
    s=read(OUT/'summary.json') if args.figure_only else build(args);figure(s)
    print(json.dumps({k:v for k,v in s.items() if k not in ('matrix','by_agent','by_task')},indent=2))
