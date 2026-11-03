#!/usr/bin/env python3
"""Refresh the OpenRouter price snapshot open-weight agents are priced by.

    openrouter_prices.py [--from FILE] [--alias CLI_MODEL=OPENROUTER_ID ...]

Fetches https://openrouter.ai/api/v1/models (public, no key) or reads a saved
copy (`--from`), keeps every open-weight vendor's models (`pricing.OPEN_VENDORS`;
OpenAI's only for gpt-oss) and rewrites `compression_bench/harness/lifecycle/
openrouter_prices.json`: per-token USD for prompt, completion and cached input,
the fetch time, and the aliases from a CLI's model id to OpenRouter's (kept, and
extended by `--alias`). Closed models are priced by their own CLI, never here.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from lifecycle import pricing  # noqa: E402

SOURCE = "https://openrouter.ai/api/v1/models"
FIELDS = ("prompt", "completion", "input_cache_read", "input_cache_write")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="a saved copy of the models list")
    parser.add_argument("--alias", action="append", default=[], metavar="CLI_MODEL=OPENROUTER_ID")
    args = parser.parse_args(argv)
    if args.source:
        listing = json.loads(args.source.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            listing = json.load(response)
    vendors = set(pricing.OPEN_VENDORS.values())
    models = {}
    for entry in listing.get("data", []):
        model = entry.get("id", "")
        vendor, _, name = model.partition("/")
        if vendor not in vendors or (vendor == "openai" and not name.startswith("gpt-oss")):
            continue
        rates = {k: entry["pricing"][k] for k in FIELDS if (entry.get("pricing") or {}).get(k)}
        if "prompt" in rates and "completion" in rates:
            models[model] = rates
    old = pricing.snapshot()
    aliases = dict(old.get("aliases") or {})
    for pair in args.alias:
        cli_model, _, target = pair.partition("=")
        aliases[cli_model] = target
    missing = sorted(t for t in aliases.values() if t not in models)
    if missing:
        raise SystemExit(f"aliased models absent from OpenRouter's list: {missing}")
    table = {"source": SOURCE, "units": "USD per token",
             "fetched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "aliases": dict(sorted(aliases.items())), "models": dict(sorted(models.items()))}
    pricing.PRICES.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(f"{len(models)} open-weight models priced, {len(aliases)} aliases -> {pricing.PRICES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
