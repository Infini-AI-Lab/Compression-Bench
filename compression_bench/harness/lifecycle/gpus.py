"""Which GPU a run may use, and holding it while the run lives.

A card is free when, over several samples, (1) no compute process is on it,
(2) almost none of its memory is in use, and (3) no other compression-bench run
holds its claim. A claim is an exclusive `flock` on
`/tmp/compression-bench-card-claims/<index>.lock`, held by the driver's own process, so
a crashed driver releases it automatically; the JSON beside it says who holds it.

Container GPU reservations (`HostConfig.DeviceRequests`) are REPORTED, not
treated as busy by themselves: on shared hosts long-idle dev containers
(`sleep infinity`, some requesting all GPUs) reserve every card while using
none, and what decides contention is a process or memory on the card.

Exclusions:
  COMPRESSION_BENCH_EXCLUDED_GPUS       never used, whether picked or named
  COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS  never picked automatically, usable when named
                                (e.g. a card pinned to another system via CDI)

Nothing here ever signals another process or touches another container.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from .utils import utc_now

CLAIMS = Path("/tmp/compression-bench-card-claims")
#: Above this much used memory a card is busy even without a listed process
#: (a process in another PID namespace can be invisible to --query-compute-apps).
BUSY_MEMORY_MIB = 1024
SAMPLES = 3
SAMPLE_INTERVAL_S = 5.0


def _indices(var: str) -> frozenset[int]:
    raw = os.environ.get(var, "").replace(",", " ").split()
    return frozenset(int(x) for x in raw if x.isdigit())


def excluded() -> frozenset[int]:
    return _indices("COMPRESSION_BENCH_EXCLUDED_GPUS")


def auto_excluded() -> frozenset[int]:
    return _indices("COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS")


def sample() -> dict[int, dict]:
    """One reading: per card, its uuid, used memory and compute pids."""
    cards = {}
    got = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, timeout=60, check=False)
    for line in got.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            cards[int(parts[0])] = {"uuid": parts[1],
                                    "memory_mib": int(float(parts[2] or 0)),
                                    "pids": []}
    got = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=60, check=False)
    by_uuid = {card["uuid"]: index for index, card in cards.items()}
    for line in got.stdout.splitlines():
        uuid, _, pid = (p.strip() for p in line.partition(","))
        if uuid in by_uuid:
            cards[by_uuid[uuid]]["pids"].append(pid)
    return cards


def busy_cards(samples: int = SAMPLES, interval: float = SAMPLE_INTERVAL_S,
               *, read=sample, sleep=time.sleep) -> tuple[set[int], dict[int, dict]]:
    """Cards busy in ANY of `samples` readings: a card reads idle for a moment
    while a neighbour's engine starts, so one quiet reading is not enough."""
    busy: set[int] = set()
    last: dict[int, dict] = {}
    for n in range(max(1, samples)):
        if n:
            sleep(interval)
        last = read()
        for index, card in last.items():
            if card["pids"] or card["memory_mib"] > BUSY_MEMORY_MIB:
                busy.add(index)
    return busy, last


def reservations(uuids: dict[int, str]) -> dict[int, list[str]]:
    """Running containers whose DeviceRequests name each card (Count=-1 means
    all cards). Reported as a warning, never decisive on its own."""
    names = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True,
                           timeout=60, check=False).stdout.split()
    held: dict[int, list[str]] = {}
    if not names:
        return held
    got = subprocess.run(["docker", "inspect", "-f",
                          "{{.Name}}\t{{json .HostConfig.DeviceRequests}}", *names],
                         capture_output=True, text=True, timeout=120, check=False)
    by_uuid = {uuid: index for index, uuid in uuids.items()}
    for line in got.stdout.splitlines():
        name, _, raw = line.partition("\t")
        name = name.lstrip("/")
        try:
            requests = json.loads(raw) or []
        except json.JSONDecodeError:
            continue
        for request in requests:
            ids = request.get("DeviceIDs") or []
            if request.get("Count") == -1 and not ids:
                for index in uuids:
                    held.setdefault(index, []).append(name)
            for device in ids:
                index = int(device) if str(device).isdigit() else by_uuid.get(device)
                if index is not None:
                    held.setdefault(index, []).append(name)
    return {index: sorted(set(who)) for index, who in held.items()}


class Claim:
    """An exclusive, process-lifetime hold on one card."""

    def __init__(self, index: int, *, run_id: str, root: Path = CLAIMS):
        self.index = int(index)
        self.run_id = run_id
        self.root = Path(root)
        self._handle = None

    @property
    def lock_path(self) -> Path:
        return self.root / f"{self.index}.lock"

    @property
    def info_path(self) -> Path:
        return self.root / f"{self.index}.json"

    def acquire(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self._handle = handle
        self.info_path.write_text(json.dumps({
            "gpu": self.index, "run_id": self.run_id, "pid": os.getpid(),
            "claimed_utc": utc_now()}) + "\n")
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            info = json.loads(self.info_path.read_text())
            if info.get("run_id") == self.run_id:
                self.info_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        fcntl.flock(self._handle, fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def held_by_claims(indices, root: Path = CLAIMS) -> set[int]:
    """Cards whose claim lock some live process holds right now."""
    held = set()
    for index in indices:
        path = Path(root) / f"{index}.lock"
        if not path.exists():
            continue
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                held.add(index)
    return held


def claim_holder(index: int, root: Path = CLAIMS) -> str | None:
    """The run id written beside a card's claim (who holds it), if any."""
    try:
        return json.loads((Path(root) / f"{index}.json").read_text()).get("run_id")
    except (OSError, json.JSONDecodeError):
        return None


def why_busy(index: int, *, samples: int = SAMPLES, interval: float = SAMPLE_INTERVAL_S,
             read=sample, sleep=time.sleep, claims_root: Path = CLAIMS) -> str:
    """Empty when the named card may be used, else the reason it may not."""
    if index in excluded():
        return f"GPU {index} is excluded by COMPRESSION_BENCH_EXCLUDED_GPUS"
    busy, last = busy_cards(samples, interval, read=read, sleep=sleep)
    if index not in last:
        return f"no GPU {index} on this host"
    if index in busy:
        card = last[index]
        return (f"GPU {index} is in use (compute pids {card['pids'] or 'none listed'}, "
                f"{card['memory_mib']} MiB used)")
    if index in held_by_claims([index], claims_root):
        return f"GPU {index} is claimed by another compression-bench run"
    return ""


def pick(*, samples: int = SAMPLES, interval: float = SAMPLE_INTERVAL_S,
         read=sample, sleep=time.sleep, claims_root: Path = CLAIMS) -> int | None:
    """The lowest-numbered usable card that is not auto-excluded, or None."""
    busy, last = busy_cards(samples, interval, read=read, sleep=sleep)
    claimed = held_by_claims(last, claims_root)
    skip = excluded() | auto_excluded()
    for index in sorted(last):
        if index in skip or index in busy or index in claimed:
            continue
        return index
    return None


def acquire_idle(index: int, *, run_id: str, patience_s: float = 900, give_up_s: float = 2 * 3600,
                 poll_s: float = 60, read=sample, sleep=time.sleep, clock=time.monotonic,
                 claims_root: Path = CLAIMS, log=lambda message: None) -> tuple[int, "Claim | None"] | None:
    """A card with nothing on it for a hidden verdict: the run's own when it is idle, else, once
    it has stayed busy for `patience_s`, any other idle unclaimed card (claimed here until the
    caller releases it), else None after `give_up_s`. A verdict measures the submission, so a
    foreign process holding memory on the card (vLLM then fails its start-up memory check) is
    waited out or avoided, never scored. Nothing is signalled."""
    start = clock()
    while True:
        busy, last = busy_cards(read=read, sleep=sleep)
        if index in last and index not in busy:
            # Idle is not enough: another verdict or probe may have just claimed the card and not loaded yet.
            claim = Claim(index, run_id=f"{run_id}-verdict", root=claims_root)
            if claim.acquire():
                return index, claim
            if claim_holder(index, claims_root) == run_id:
                return index, None             # the run's own driver holds its card: the verdict is its own
        waited = clock() - start
        if waited >= patience_s:
            other = pick(read=read, sleep=sleep, claims_root=claims_root)
            if other is not None:
                claim = Claim(other, run_id=f"{run_id}-verdict", root=claims_root)
                if claim.acquire():
                    log(f"GPU {index} stayed busy for {waited:.0f}s; the verdict runs on idle GPU {other}")
                    return other, claim
        if waited >= give_up_s:
            return None
        card = last.get(index) or {}
        log(f"GPU {index} is busy (pids {card.get('pids') or 'none listed'}, {card.get('memory_mib')} MiB): "
            f"the verdict waits for an idle card")
        sleep(poll_s)


def main(argv=None) -> int:
    """`python -m lifecycle.gpus pick | check N | reservations N` for the launcher."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"] and len(argv) == 2:
        why = why_busy(int(argv[1]))
        print(why or "free")
        return 1 if why else 0
    if argv[:1] == ["reservations"] and len(argv) == 2:
        uuids = {i: c["uuid"] for i, c in sample().items()}
        print(", ".join(reservations(uuids).get(int(argv[1]), [])) or "none")
        return 0
    found = pick()
    if found is None:
        print("no free GPU", file=sys.stderr)
        return 1
    print(found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
