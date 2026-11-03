"""Fetch only the sampled BF16 tensors of the two MoE source checkpoints, at the revisions the task
contracts pin, with HTTP range requests (compression_bench/scripts/baselines/released_identity.py helpers);
no shard is downloaded whole. Qwen3.5-122B-A10B stores each layer's experts fused ([256, 2048, 3072]
gate_up, [256, 3072, 1024] down): only the sampled experts' contiguous slices are fetched. Embedding and
lm_head: four contiguous 1024-row slices. Qwen3.8-27B is read from the local BF16 copy instead.

    python3 fetch_sampled_tensors.py [--out DIR] [--models nemotron,qwen122]
                                                      # default DIR: compression_bench/evolve_space/analysis/quant_error/tensors

Writes DIR/<model>/<tensor name>[.e<expert>|.r<row0>].npy (float32 is NOT used: raw bf16 as uint16) and
DIR/<model>/MANIFEST.json (name, shape, dtype, shard, byte range, revision).
"""
import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import QUANT_ERROR_DIR, REPO  # noqa: E402
sys.path.insert(0, str(REPO / "compression_bench" / "scripts" / "baselines"))
import released_identity as ri  # noqa: E402

HF = "https://huggingface.co"
MODELS = {
    "nemotron": dict(repo="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                     rev="1f8953f7f0935358e3605150d22bc15df062061c",
                     contract="compression_tasks/nemotron3_5_super_vl_120b_a12b_48g_standard_1xH200NVL/contract.yaml"),
    "qwen122": dict(repo="Qwen/Qwen3.5-122B-A10B", rev="dc4d348443bc740c68e2d77492492c11606384d5",
                    contract="compression_tasks/qwen3_5_122b_a10b_48g_standard_1xH200NVL/contract.yaml"),
}
ROW_SLICES = (0, 32768, 65536, 98304)   # 1024-row slices of embedding / lm_head (both vocabularies > 99328)


def plan(model: str) -> list[dict]:
    """(tensor, slice) pairs to fetch. Same depths/experts as grid_mse.py for Nemotron."""
    out = []
    if model == "nemotron":
        P = "language_model.backbone.layers"
        for L in (1, 12, 45, 72, 87):
            for e in (0, 64, 128, 192, 256, 320, 384, 448):
                for p in ("up", "down"):
                    out.append(dict(name=f"{P}.{L}.mixer.experts.{e}.{p}_proj.weight", cls=f"routed_{p}", layer=L, expert=e))
            for p in ("up", "down"):
                out.append(dict(name=f"{P}.{L}.mixer.shared_experts.{p}_proj.weight", cls=f"shared_{p}", layer=L))
            for p in ("fc1", "fc2"):
                out.append(dict(name=f"{P}.{L}.mixer.{p}_latent_proj.weight", cls=f"latent_{p}", layer=L))
        for L in (0, 44, 86):
            out.append(dict(name=f"{P}.{L}.mixer.in_proj.weight", cls="mamba_in", layer=L))
            out.append(dict(name=f"{P}.{L}.mixer.out_proj.weight", cls="mamba_out", layer=L))
        for L in (7, 47, 78):
            for p in ("q", "k", "v", "o"):
                out.append(dict(name=f"{P}.{L}.mixer.{p}_proj.weight", cls=f"attn_{p}", layer=L))
        for r in ROW_SLICES:
            out.append(dict(name="language_model.backbone.embeddings.weight", cls="embed", row0=r, rows=1024))
            out.append(dict(name="language_model.lm_head.weight", cls="lm_head", row0=r, rows=1024))
    else:
        P = "model.language_model.layers"
        for L in (0, 12, 24, 36, 47):
            for e in (0, 32, 64, 96, 128, 160, 192, 224):
                out.append(dict(name=f"{P}.{L}.mlp.experts.gate_up_proj", cls="routed_gate_up", layer=L, expert=e))
                out.append(dict(name=f"{P}.{L}.mlp.experts.down_proj", cls="routed_down", layer=L, expert=e))
            for p in ("gate", "up", "down"):
                out.append(dict(name=f"{P}.{L}.mlp.shared_expert.{p}_proj.weight", cls=f"shared_{p}", layer=L))
        for L in (0, 12, 24, 36):
            for p in ("in_proj_qkv", "in_proj_z", "out_proj"):
                out.append(dict(name=f"{P}.{L}.linear_attn.{p}.weight", cls=f"lin_{p}", layer=L))
        for L in (3, 23, 47):
            for p in ("q", "k", "v", "o"):
                out.append(dict(name=f"{P}.{L}.self_attn.{p}_proj.weight", cls=f"attn_{p}", layer=L))
        for r in ROW_SLICES:
            out.append(dict(name="model.language_model.embed_tokens.weight", cls="embed", row0=r, rows=1024))
            out.append(dict(name="lm_head.weight", cls="lm_head", row0=r, rows=1024))
    return out


def fname(item: dict) -> str:
    s = item["name"]
    if s.endswith(("experts.gate_up_proj", "experts.down_proj")):      # fused [E, out, in]: one expert's slice
        s += f".e{item['expert']}"
    if "row0" in item:
        s += f".r{item['row0']}"
    return s + ".npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(QUANT_ERROR_DIR / "tensors"))
    ap.add_argument("--models", default="nemotron,qwen122")
    a = ap.parse_args()
    for model in a.models.split(","):
        m = MODELS[model]
        d = Path(a.out) / model
        d.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{HF}/{m['repo']}/resolve/{m['rev']}/model.safetensors.index.json", timeout=120) as r:
            wm = json.load(r)["weight_map"]
        items = plan(model)
        headers = {}
        for shard in sorted({wm[i["name"]] for i in items}):
            headers[shard] = ri.safetensors_header(f"{HF}/{m['repo']}/resolve/{m['rev']}/{shard}")

        def fetch(item):
            shard = wm[item["name"]]
            t = headers[shard][item["name"]]
            assert t["dtype"] == "BF16", t
            shape = list(t["shape"])
            start, nbytes = t["data_start"], t["bytes"]
            if item["name"].endswith(("experts.gate_up_proj", "experts.down_proj")):   # fused [E, out, in]
                per = nbytes // shape[0]
                start, nbytes, shape = start + item["expert"] * per, per, shape[1:]
            if "row0" in item:
                per = nbytes // shape[0]
                start, nbytes, shape = start + item["row0"] * per, per * item["rows"], [item["rows"]] + shape[1:]
            path = d / fname(item)
            if not path.exists():
                raw = ri.get(f"{HF}/{m['repo']}/resolve/{m['rev']}/{shard}", start, start + nbytes)
                np.save(path, np.frombuffer(raw, np.uint16).reshape(shape))
            return dict(item, file=path.name, shape=shape, dtype="BF16", shard=shard, byte_range=[start, start + nbytes])

        with ThreadPoolExecutor(8) as ex:
            manifest = list(ex.map(fetch, items))
        json.dump(dict(repo=m["repo"], revision=m["rev"], contract=m["contract"], tensors=manifest),
                  open(d / "MANIFEST.json", "w"), indent=1)
        print(model, len(manifest), "tensors,", sum(x["byte_range"][1] - x["byte_range"][0] for x in manifest) / 2**30, "GiB")


if __name__ == "__main__":
    main()
