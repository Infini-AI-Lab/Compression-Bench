#!/usr/bin/env bash
# Keep a sweep alive through disk pressure and GPU contention, until its queue has nothing queued or running.
#
#   setsid -f nohup bash compression_bench/scripts/sweep_guardian.sh compression_bench/evolve_space/scheduler/queue.json AGENT \
#       [SCHEDULER ARGS ...] < /dev/null >> compression_bench/evolve_space/scheduler/guardian.out 2>&1
#
# SCHEDULER ARGS (default: --launch-gib 50 --low-gib 100 --critical-gib 30) go to the scheduler whenever the watchdog has
# to be started again, e.g. --exclude-gpus 0,1 to keep the sweep off GPUs 0 and 1.
#
# Every 2 minutes, for the queue's items of AGENT:
#   scheduler   the watchdog (which restarts the scheduler, which relaunches a run a system fault ended -- a killed
#               driver, a GPU taken, a full disk -- seeded from its own workspace, on a free GPU) must be alive;
#               a dead watchdog is started again with the sweep's arguments.
#   disk        below 150 GiB free on the repository's filesystem: harvest_runs.py --all (finished runs' measured
#               weights; never logs, metrics or unmeasured checkpoints), at most every 20 minutes. Below 60 GiB:
#               the measured checkpoint weights kept under compression_output/.case_studies (copies made by
#               `case_study_keep.py`, every version already verdicted) go too, once, with an ALERT.
#   alerts      ALERT lines for what needs a person: an item given up, a final verdict that is no measurement
#               (NOT_RUN: a GPU taken during the verdict), unmeasured checkpoints left by a run, a run held for
#               credit (lifecycle/agent.py credit_hold), the Docker root under 62 GiB (a launch needs 60).
# Never signals a process it did not start; pid files only. Log: guardian.log beside the queue.
set -u
queue="$1"; agent="$2"; shift 2
# the scheduler's arguments when the watchdog has to be started again (default: the sweep's disk tiers)
sched_args=("$@"); [ ${#sched_args[@]} -gt 0 ] || sched_args=(--launch-gib 50 --low-gib 100 --critical-gib 30)
dir="$(dirname "$queue")"; log="$dir/guardian.log"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
. compression_bench/scripts/env.sh >/dev/null 2>&1
echo $$ > "$dir/guardian.pid"
docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
last_harvest=0; case_cleaned=0
say "guardian started for $agent (pid $$)"
while true; do
    # the sweep's items for this agent, and the runs that need attention
    state="$("$COMPRESSION_BENCH_PY" - "$queue" "$agent" <<'PY'
import json, sys
from pathlib import Path
queue, agent = sys.argv[1], sys.argv[2]
items = [i for i in json.loads(Path(queue).read_text())["items"] if i["agent"] == agent]
live = sum(i["status"] in ("queued", "running") for i in items)
alerts = [f"{i['id']} gave up after {i.get('attempts')} attempts" for i in items if i["status"] == "gave_up"]
for i in items:
    run = Path(i.get("run_dir") or "/nonexistent")
    kept = run / "KEPT_AFTER_FAILURE.json"
    if i["status"] == "running" and kept.exists():
        alerts.append(f"{i['id']}: its driver died and the run is kept as it was ({run.name}, --keep-on-failure): resume or end it by hand")
    hold = run / "CREDIT_HOLD.json"
    if i["status"] == "running" and hold.exists():
        try:
            since = json.loads(hold.read_text()).get("since_utc")
        except (OSError, json.JSONDecodeError):
            since = "?"
        alerts.append(f"{i['id']} holds for credit since {since} ({run.name}, container paused, nothing deleted): "
                      f"add credit and it resumes, or touch {run}/END_CREDIT_HOLD to verdict it as it is")
    summary = run / "SUMMARY.json"
    if i["status"] == "done" and summary.exists():
        if json.loads(summary.read_text()).get("verdict") == "NOT_RUN":
            alerts.append(f"{i['id']} final verdict NOT_RUN ({run.name}): re-run it on a free GPU")
        metrics = run / "METRICS.json"
        final = (json.loads(metrics.read_text()).get("verdicts") or {}).get("final") or {} if metrics.exists() else {}
        text = (final.get("error") or "") + json.dumps(final.get("detail") or {})[:200000]
        if any(t in text for t in ("Free memory on device", "CUDA out of memory", "No space left on device")):
            alerts.append(f"{i['id']} final verdict {final.get('status')} was lost to the host ({run.name}): re-run it on a free GPU")
        log = run / "checkpoint_verdicts.log"
        lines = log.read_text().splitlines() if log.exists() else []
        try:
            last = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            last = {}
        if last.get("kept_unmeasured"):
            alerts.append(f"{i['id']} left unmeasured checkpoints {last['kept_unmeasured']} ({run.name})")
print(live)
for a in alerts:
    print(a)
PY
)"
    live="$(echo "$state" | head -1)"
    echo "$state" | tail -n +2 | while read -r a; do [ -n "$a" ] && ! grep -qF "$a" "$log" && say "ALERT $a"; done
    # the watchdog keeps the scheduler alive; the guardian keeps the watchdog alive
    wpid="$(cat "$dir/watchdog.pid" 2>/dev/null || true)"
    if [ "${live:-0}" -gt 0 ] && { [ -z "$wpid" ] || ! kill -0 "$wpid" 2>/dev/null; }; then
        say "watchdog (pid ${wpid:-none}) is gone with $live item(s) left: starting it"
        setsid -f nohup bash compression_bench/scripts/scheduler_watchdog.sh "$queue" "${sched_args[@]}" \
            < /dev/null >> "$dir/watchdog.out" 2>&1
    fi
    # disk
    free="$(df --output=avail -BG "$repo" | tail -1 | tr -dc 0-9)"
    dfree="$(df --output=avail -BG "$docker_root" 2>/dev/null | tail -1 | tr -dc 0-9)"
    now="$(date +%s)"
    if [ "$free" -lt 150 ] && [ $((now - last_harvest)) -ge 1200 ]; then
        last_harvest="$now"
        freed="$("$COMPRESSION_BENCH_PY" compression_bench/scripts/harvest_runs.py --all 2>&1 | tail -1)"
        say "disk ${free} GiB free: harvest_runs --all: $freed"
    fi
    if [ "$free" -lt 60 ] && [ "$case_cleaned" -eq 0 ]; then
        case_cleaned=1
        n="$(find compression_output/.case_studies -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.npz' \) | wc -l)"
        find compression_output/.case_studies -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.npz' \) -delete
        say "ALERT disk ${free} GiB free: deleted $n measured case-study checkpoint weight files; now $(df --output=avail -BG "$repo" | tail -1 | tr -dc 0-9) GiB free"
    fi
    if [ -n "$dfree" ] && [ "$dfree" -lt 62 ] && ! tail -50 "$log" | grep -q "ALERT Docker root"; then
        say "ALERT Docker root ${dfree} GiB free: a launch needs 60"
    fi
    say "ok: $live $agent item(s) left, repository disk ${free} GiB, Docker root ${dfree:-?} GiB"
    if [ "${live:-0}" -eq 0 ]; then say "no $agent item left; guardian stops"; exit 0; fi
    sleep 120
done
