"""Released quantized models evaluated as baselines on the tasks ("released" baselines), and the GPTQ baseline's spec.

Each entry is a published checkpoint, pinned to a commit, with the files it ships to the
package unchanged and the runtime that serves its format natively on the GPU (llama.cpp for
GGUF). `NO_FIT` names the tasks that have no release.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

NEMOTRON_IQ2_XXS = [f"UD-IQ2_XXS/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ2_XXS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]

RELEASES = {
    "nemotron3_5_super_vl_120b_a12b_32g": {},
    "nemotron3_5_super_vl_120b_a12b_48g": {
        "unsloth-UD-IQ2_XXS": {"repo": "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
                               "revision": "75a5829055f69a0359085ae6bbd0a81b60bc05c6",
                               "format": "gguf", "engine": "llamacpp", "files": NEMOTRON_IQ2_XXS},
    },
    "qwen3_5_122b_a10b_32g": {
        "unsloth-UD-IQ1_M": {"repo": "unsloth/Qwen3.5-122B-A10B-GGUF", "revision": "51eab4d59d53f573fb9206cb3ce613f1d0aa392b",
                             "format": "gguf", "engine": "llamacpp", "files": ["Qwen3.5-122B-A10B-UD-IQ1_M.gguf"]},
    },
    "qwen3_5_122b_a10b_48g": {
        "unsloth-UD-IQ3_S": {"repo": "unsloth/Qwen3.5-122B-A10B-GGUF", "revision": "51eab4d59d53f573fb9206cb3ce613f1d0aa392b",
                             "format": "gguf", "engine": "llamacpp", "files": ["Qwen3.5-122B-A10B-UD-IQ3_S.gguf"]},
    },
    "qwen3_8_27b_8g": {
        "ista-GSQ-RCO-IQ2_XS": {"repo": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "revision": "d562806dbafae37109975e970aae91b43e73b440",
                                "format": "gguf", "engine": "llamacpp", "files": ["Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf"]},
    },
    "qwen3_8_27b_12g": {
        "ista-GSQ-RCO-IQ3_S": {"repo": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", "revision": "d562806dbafae37109975e970aae91b43e73b440",
                               "format": "gguf", "engine": "llamacpp", "files": ["Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"]},
    },
}
NO_FIT = {"nemotron3_5_super_vl_120b_a12b_32g": "no released quantization of this checkpoint fits 32 GiB"}


def short(task: str) -> str:
    return re.sub(r"_standard_1xH200NVL$", "", Path(task).name)


def task_dir(task: str) -> Path:
    return REPO / "compression_tasks" / f"{short(task)}_standard_1xH200NVL"


def release(task: str, name: str) -> dict:
    if name == "gptq":
        return gptq(task)
    found = RELEASES[short(task)].get(name)
    if found is None:
        raise SystemExit(f"no release {name!r} for {short(task)}; known: {sorted(RELEASES[short(task)])}")
    return {"name": name, **found}


def source_of(task: str) -> dict:
    """The BF16 checkpoint the task's contract pins."""
    text = (task_dir(task) / "contract.yaml").read_text()
    block = text.split("\nmodel:\n", 1)[1]
    repo = re.search(r"^\s+id:\s*\"?([^\"\n]+)\"?", block, re.M).group(1).strip()
    revision = re.search(r"^\s+revision:\s*\"?([0-9a-f]{40})", block, re.M).group(1)
    return {"repo": repo, "revision": revision}


def contract_capacity(task: str) -> float:
    """The task's capacity limit in GiB, from its contract."""
    return float(re.search(r"^capacity_gib:\s*([\d.]+)", (task_dir(task) / "contract.yaml").read_text(), re.M).group(1))


#: The repo's GPTQ baseline (gptq_baseline.py), regenerated with its recorded recipe and run through the same
#: pipeline as the releases: its packed weights served by the vLLM plugin baseline_latency.py installs.
GPTQ_ROOT = REPO / "compression_bench" / "evolve_space" / "baselines" / "released" / "gptq"
#: baseline_latency.py's GPTQ records: the recipe and inference.py each regeneration reuses.
LATENCY_GPTQ = REPO / "evaluation_progress" / "baselines" / "latency" / "gptq"


def gptq(task: str) -> dict:
    return {"name": "gptq", "format": "gptq-packed", "engine": "vllm-gptq",
            "recipe": "compression_bench/scripts/baselines/gptq_baseline.py, as recorded in "
                      f"{(LATENCY_GPTQ / short(task) / 'quantize-spec.json').relative_to(REPO)}",
            "model_dir": str((GPTQ_ROOT / short(task) / "model").relative_to(REPO)),
            "inference": str((LATENCY_GPTQ / short(task) / "inference.py").relative_to(REPO))}
