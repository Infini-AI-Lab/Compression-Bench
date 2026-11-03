#!/usr/bin/env bash
# Keep the scheduler alive until its sweep is finished.
#
#   setsid -f nohup bash compression_bench/scripts/scheduler_watchdog.sh compression_bench/evolve_space/scheduler/queue.json &
#
# Every 60 s: if the scheduler's pid file names a dead process and no report.json says the sweep is finished,
# start it again (the queue file is its state, so it picks up where it was). Uses pid files only.
set -u
queue="$1"
shift
dir="$(dirname "$queue")"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$repo/compression_bench/scripts/env.sh" >/dev/null 2>&1
echo $$ > "$dir/watchdog.pid"
while true; do
    if [ -f "$dir/report.json" ] && [ "$dir/report.json" -nt "$queue" ]; then
        echo "$(date -u +%FT%TZ) WATCHDOG sweep finished; stopping" >> "$dir/scheduler.log"; exit 0
    fi
    pid="$(cat "$dir/scheduler.pid" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) ALERT WATCHDOG restarts the scheduler (pid ${pid:-none} is gone)" >> "$dir/scheduler.log"
        (cd "$repo" && setsid -f nohup "$COMPRESSION_BENCH_PY" compression_bench/scripts/scheduler.py --queue "$queue" --max-live 4 \
            "$@" \
            < /dev/null >> "$dir/scheduler.out" 2>&1)
        sleep 10
    fi
    sleep 60
done
