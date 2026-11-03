#!/usr/bin/env python3
"""Download a task's pinned checkpoint into the HF cache when model_for_task cannot find it.

    python3 compression_bench/scripts/deploy/fetch_model.py compression_tasks/<task>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))
from lifecycle.utils import read_contract  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

model = read_contract(sys.argv[1])["model"]
print(snapshot_download(model["id"], revision=model["revision"], max_workers=8), flush=True)
