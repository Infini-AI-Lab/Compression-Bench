#!/usr/bin/env python3
"""The public Checker's evaluator: the shared pipeline (evaluate.py) on public inputs.

    evaluate_public.py <submission> --output <result.json> [--package-only] [--profile formal|debug-9b]

`server.py` runs it once per request. The profile (`evaluator-assets/
evaluation_profiles.json`) selects the inputs: `formal` scores the submission on
the public perplexity blocks and decode prompts (no correctness gate, which runs
only in the hidden verdict); `debug-9b` measures the debug checkpoint unscored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate as shared
from inference_runtime import review_metadata

ASSETS_DIR = Path(__file__).parent / "evaluator-assets"
PROFILE_CONFIG = ASSETS_DIR / "evaluation_profiles.json"
CheckError = shared.SubmissionError


def load_profile(name="formal", config_path=PROFILE_CONFIG):
    """Read task-owned settings; relative resource paths are relative to the config."""
    config_path = Path(config_path)
    profiles = json.loads(config_path.read_text(encoding="utf-8"))
    if name not in profiles:
        raise ValueError(f"this task has no {name!r} Checker profile")
    profile = dict(profiles[name])
    for key in ("tokenizer", "decode_assets", "ppl_assets"):
        path = Path(profile[key])
        profile[key] = path if path.is_absolute() else config_path.parent / path
    return profile

DEFAULT_PROFILE = load_profile()


def evaluate(submission, *, package_only=False, timeout=600, profile=None):
    """Select public/debug resources, call the shared pipeline, and format the response."""
    profile = DEFAULT_PROFILE if profile is None else profile
    measured = shared.evaluate_submission(submission, profile, public=True,
                                          package_only=package_only, timeout=timeout)
    result = {"valid": True, "public_only": True, **measured["check"], **review_metadata(),
              "stage_seconds": {row["stage"]: row["seconds"]
                                for row in (measured.get("timing") or {}).get("stages", [])}}
    if not profile.get("score"):
        result.update(debug_only=True, is_official_score=False)
    if package_only:
        return result
    quality, speed = measured["ppl"], measured["decode"]
    result.update(model_memory=speed.get("model_memory"),
                  model_memory_status="measured_pending_independent_audit")
    # The agent gets the measured perplexity and reward, never how it was measured:
    # no per-block losses, protocol id, split, token counts or reward-formula text,
    # which would describe the hidden benchmark.
    result.update(generated_text=measured["smoke"]["generated_text"],
                  public_ppl=quality["ppl"], decode=speed)
    if profile.get("score"):
        # The correctness gate runs only in the hidden verdict; nothing of it is public.
        score = measured["score"]
        result.update(public_reward=score["reward"], baseline_status="measured")
        for key in ("scoring_version", "reward_scale", "ppl_fidelity", "bf16_ppl",
                    "ppl_increase_percent", "decode_latency_seconds", "decode_in_reward",
                    "correctness_gate"):
            result[key] = score[key]
        result.update(public_bf16_ppl=score["bf16_ppl"])
    else:
        latency = speed["median_round_mean_seconds"]
        result.update(decode_latency_seconds=latency, ppl_latency_product=quality["ppl"] * latency)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-only", action="store_true",
                        help="check package files and size without loading the model")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--profile", default="formal", help="profile name in the task-owned configuration")
    parser.add_argument("--config", type=Path, default=PROFILE_CONFIG)
    args = parser.parse_args()
    result = {"valid": False, "public_only": True, **review_metadata()}
    try:
        profile = load_profile(args.profile, args.config)
        if not profile.get("score"):
            result.update(debug_only=True, is_official_score=False)
        result = evaluate(args.submission, package_only=args.package_only,
                          timeout=args.timeout, profile=profile)
    except Exception as exc:
        result.update(error=str(exc), **review_metadata())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
