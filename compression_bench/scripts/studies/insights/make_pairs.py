"""Local-vs-public comparison pairs: where an agent's own local validation and the Checker's public reward
disagree on which of two candidates is better.

    python3 make_pairs.py SPEC.json [--out pairs.json]

SPEC.json lists the runs and the candidate pairs read from their trajectories:

    {"runs":  {"<key>": {"agent": "...", "task": "<task id>", "run_dir": "<run directory, relative to the repo or absolute>"}},
     "pairs": [{"run": "<key>", "A": "changed candidate", "B": "reference candidate",
                "validator": "natural | bf16_sampled | ref_model_sampled | candidate_sampled | weights_only",
                "change_type": "U | R | C", "reason": "what the change does",
                "metric": "the local metric as the agent stated it", "unit": "nats/token | CE nats ... | PPL ...",
                "local_A": float | null, "local_B": float | null,
                "reference": float | null,   # BF16 (or near-BF16) CE for CE metrics; the relative change in % when local_A is null
                "steps": "trajectory steps the local numbers come from",
                "public_A": ["<utc_start prefix>", <public reward>], "public_B": [...], "note": ""}]}

Each candidate's public reward is looked up in the run's public-checks/checks.jsonl (full checks only) by its
utc_start prefix and must match the stated reward within 0.006. local_rel_change_pct = A/B - 1 (for CE with a
BF16/near-BF16 reference: (A-ref)/(B-ref) - 1; for PPL: ratio - 1; null for CE without reference, the sign then from
local_delta). public implied KL = ln(10/public_reward). reversed = sign(local change) != sign(public implied-KL change).
"""
import argparse
import collections
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def full_checks(run_dir):
    out = []
    for line in open(run_dir / "public-checks/checks.jsonl"):
        d = json.loads(line)
        r = d["response"]
        if d["request"].get("action") == "full-check" and r.get("public_reward") is not None:
            out.append((d["utc_start"], r["public_reward"], r.get("scoring_version")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", default="pairs.json")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    runs = spec["runs"]
    checks = {k: full_checks(REPO / r["run_dir"]) for k, r in runs.items()}

    def pub(k, prefix, expect):
        m = [c for c in checks[k] if c[0].startswith(prefix)]
        assert len(m) == 1, (k, prefix, m)
        u, v, sv = m[0]
        assert abs(v - expect) < 0.006, (k, prefix, v, expect)
        return v, u, sv

    pairs = []
    for i, s in enumerate(spec["pairs"], 1):
        k = s["run"]
        lA, lB, ref, unit = s.get("local_A"), s.get("local_B"), s.get("reference"), s["unit"]
        vA, uA, svA = pub(k, *s["public_A"])
        vB, uB, svB = pub(k, *s["public_B"])
        if lA is None:
            lrel = ref  # the relative change is stated directly
        elif ref is not None:
            lrel = ((lA - ref) / (lB - ref) - 1) * 100
        elif unit.startswith("CE nats"):
            lrel = None
        else:
            lrel = (lA / lB - 1) * 100
        pubrel = (math.log(10 / vA) / math.log(10 / vB) - 1) * 100
        ldelta = (lA - lB) if lA is not None else None
        lsign = lrel if lrel is not None else ldelta
        rev = (lsign < 0) != (pubrel < 0)
        run = runs[k]
        pairs.append(dict(id=i, run=f"{run['task']} / {run['agent']}", run_dir=run["run_dir"],
                          trajectory_hint="lineage TRAJECTORY/<run>/steps.jsonl.gz",
                          candidate_A=s["A"], candidate_B=s["B"], validator_text=s["validator"], change_type=s["change_type"],
                          change_type_reason=s["reason"], local_metric=s["metric"], local_unit=unit,
                          local_A=lA, local_B=lB, local_reference=ref, local_delta_A_minus_B=ldelta,
                          local_rel_change_pct=None if lrel is None else round(lrel, 2),
                          public_reward_A=vA, public_reward_B=vB, public_utc_A=uA, public_utc_B=uB, public_scoring_version=svA,
                          public_implied_KL_rel_change_pct=round(pubrel, 2), reversed=bool(rev),
                          local_source=f"steps.jsonl.gz {s['steps']}",
                          public_source=f"{run['run_dir']}/public-checks/checks.jsonl utc_start {uA} (A), {uB} (B)",
                          note=s.get("note", "")))
    cnt = collections.defaultdict(lambda: {"pairs": 0, "reversed": 0})
    for p in pairs:
        for key in [("all",), (p["validator_text"],), (p["validator_text"], "U" if p["change_type"] == "U" else "R_or_C"),
                    ("change_type", p["change_type"])]:
            kk = "/".join(key)
            cnt[kk]["pairs"] += 1
            cnt[kk]["reversed"] += p["reversed"]
    big = [p["id"] for p in pairs if p["reversed"] and abs(p["public_implied_KL_rel_change_pct"]) >= 10]
    out = dict(description=__doc__.strip(), spec=str(a.spec),
               summary_counts={k: v for k, v in sorted(cnt.items())}, large_reversals_ids_abs_public_ge_10pct=big, pairs=pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in sorted(cnt.items()):
        print(k, v)
    print("large reversals", big)
    for p in pairs:
        print(p["id"], p["run"], p["validator_text"], p["change_type"], p["local_rel_change_pct"], p["local_delta_A_minus_B"],
              p["public_implied_KL_rel_change_pct"], "REV" if p["reversed"] else "")


if __name__ == "__main__":
    main()
