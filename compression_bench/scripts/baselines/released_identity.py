#!/usr/bin/env python3
"""Is a released quantized model made from the task's own BF16 checkpoint?

    python3 released_identity.py <task> --release <name> [--out FILE]

Reads, over HTTP range requests and without downloading any weights, the headers of the
release's files (GGUF) at its pinned revision and of the task's
BF16 checkpoint at the revision its contract pins. Every tensor the release stores
unquantized (F32/F16/BF16: norms, SSM parameters, biases, convolutions, routers) is
compared value by value with the BF16 tensor of the same layer and size; a converter's
fixed transform (`+1` on a shifted RMSNorm, `-exp(A_log)`) is tried and named. Identity
holds when every compared tensor equals a BF16 tensor exactly. Also recorded: the
release's own metadata (base model, quantization types), its file sizes and LFS
digests at the revision, and the BF16 shards' LFS digests at both revisions.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import released_models  # noqa: E402

HF = "https://huggingface.co"
CHUNK = 4 << 20

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
        11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
        19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
        27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
FLOAT = {"F32": np.float32, "F16": np.float16, "BF16": "bf16", "F64": np.float64}
ST_FLOAT = {"F32": "F32", "F16": "F16", "BF16": "BF16"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_LOCATION: dict[str, tuple[str, float]] = {}
_LOCK = threading.Lock()


def located(url: str, *, fresh: bool = False) -> str:
    """The storage URL a resolve link redirects to, asked of the Hub once per file (its rate limit)."""
    with _LOCK:
        hit = _LOCATION.get(url)
        if hit and not fresh and time.time() - hit[1] < 1800:
            return hit[0]
        opener = urllib.request.build_opener(_NoRedirect)
        for attempt in range(8):
            try:
                with opener.open(urllib.request.Request(url, method="HEAD"), timeout=120):
                    target = url             # served directly (a small, non-LFS file)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    target = urllib.parse.urljoin(url, exc.headers["Location"])
                    break
                if exc.code != 429 or attempt == 7:
                    raise
                time.sleep(float(exc.headers.get("Retry-After") or 30))
        _LOCATION[url] = (target, time.time())
        return target


def get(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of a URL, retried; a rate limit waits as long as the server asks."""
    for attempt in range(10):
        try:
            req = urllib.request.Request(located(url, fresh=attempt > 0),
                                         headers={"Range": f"bytes={start}-{end - 1}",
                                                  "User-Agent": "compression-bench-identity/1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start:
                raise OSError(f"short read {len(data)} of {end - start}")
            return data
        except urllib.error.HTTPError as exc:
            if attempt == 9:
                raise
            time.sleep(float(exc.headers.get("Retry-After") or 10) if exc.code == 429 else 2 * (attempt + 1))
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == 9:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


class Remote:
    """Sequential reader over one URL, fetched in chunks."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.base, self.buf = 0, b""

    def read(self, n: int) -> bytes:
        out = b""
        while n:
            lo = self.pos - self.base
            if not 0 <= lo < len(self.buf):
                self.base = self.pos
                self.buf = get(self.url, self.pos, min(self.size, self.pos + max(CHUNK, n)))
                lo = 0
            piece = self.buf[lo:lo + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return out

    def unpack(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))


def gguf_string(f: Remote) -> str:
    (n,) = f.unpack("Q")
    return f.read(n).decode("utf-8", errors="replace")


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_value(f: Remote, kind: int, keep_arrays: bool):
    if kind == 8:
        return gguf_string(f)
    if kind in SCALAR:
        return f.unpack(SCALAR[kind])[0]
    if kind == 9:
        (sub,), (n,) = f.unpack("I"), f.unpack("Q")
        if sub in SCALAR and not keep_arrays:
            f.read(n * struct.calcsize(SCALAR[sub]))
            return f"<array {n}>"
        values = [gguf_value(f, sub, keep_arrays) for _ in range(n)]
        return values if keep_arrays or n <= 16 else f"<array {n}>"
    raise ValueError(f"unknown GGUF value type {kind}")


def gguf_header(url: str, size: int) -> dict:
    f = Remote(url, size)
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file: " + url)
    (version,), (n_tensors,), (n_kv,) = f.unpack("I"), f.unpack("Q"), f.unpack("Q")
    meta = {}
    for _ in range(n_kv):
        key = gguf_string(f)
        (kind,) = f.unpack("I")
        meta[key] = gguf_value(f, kind, keep_arrays=False)
    tensors = []
    for _ in range(n_tensors):
        name = gguf_string(f)
        (dims,) = f.unpack("I")
        shape = list(f.unpack("Q" * dims))
        (kind,), (offset,) = f.unpack("I"), f.unpack("Q")
        tensors.append({"name": name, "shape": shape, "type": GGML.get(kind, f"type{kind}"), "offset": offset})
    align = int(meta.get("general.alignment", 32))
    start = (f.pos + align - 1) // align * align
    ends = sorted(t["offset"] for t in tensors) + [size - start]
    following = {o: ends[i + 1] for i, o in enumerate(ends[:-1])}
    for t in tensors:
        t["bytes"] = following[t["offset"]] - t["offset"]
        t["data_start"] = start + t["offset"]
    return {"version": version, "metadata": meta, "tensors": tensors, "data_start": start}


def safetensors_header(url: str) -> dict:
    (n,) = struct.unpack("<Q", get(url, 0, 8))
    header = json.loads(get(url, 8, 8 + n))
    header.pop("__metadata__", None)
    for name, t in header.items():
        t["data_start"] = 8 + n + t["data_offsets"][0]
        t["bytes"] = t["data_offsets"][1] - t["data_offsets"][0]
    return header


def tree(repo: str, rev: str) -> dict:
    out, url = {}, f"{HF}/api/models/{repo}/tree/{rev}?recursive=true"
    while url:
        with urllib.request.urlopen(url, timeout=120) as r:
            for f in json.load(r):
                if f["type"] == "file":
                    out[f["path"]] = {"size": f.get("size"), "sha256": (f.get("lfs") or {}).get("oid"),
                                      "git_oid": f.get("oid")}
            link, url = r.headers.get("Link"), None
            if link and 'rel="next"' in link:
                url = link.split(";")[0].strip("<>")
    return out


def as_array(data: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(data, {"F32": np.float32, "F16": np.float16, "F64": np.float64}[dtype]).astype(np.float32)


TRANSFORMS = {"identity": lambda x: x, "plus_one": lambda x: x + np.float32(1),
              "minus_exp": lambda x: -np.exp(x.astype(np.float32))}
#: A transcendental transform is computed by the converter's own float32 exp: equal within this many ulps.
EXP_ULPS = 2


def ulps(a: np.ndarray, b: np.ndarray) -> int:
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def part_of(name: str) -> str:
    return "vision" if re.search(r"vision|radio_model|mm_projector|mlp1\.|^v\.|mmproj", name) else "language model"


def layer_of(name: str) -> int | None:
    m = re.search(r"(?:^|\.)(?:blk|layers)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def compare(release_tensors: dict, source_tensors: dict, max_numel: int) -> dict:
    """Each small unquantized release tensor against the same-layer, same-size BF16 tensors."""
    def fetch(t):
        return as_array(get(t["url"], t["data_start"], t["data_start"] + t["bytes"]), t["dtype"])

    small_src = {n: t for n, t in source_tensors.items() if t["numel"] <= max_numel}
    jobs = [(n, t) for n, t in release_tensors.items() if t["numel"] <= max_numel]

    def candidates(name, t):
        """The BF16 tensor of the same name; else those of the same layer and size; else, for a
        converter's extra block (llama.cpp's `nextn` block after the last layer), the MTP tensors."""
        if name in small_src:
            return [name]
        same = [n for n, s in small_src.items() if layer_of(n) == layer_of(name) and s["numel"] == t["numel"]]
        return same or [n for n, s in small_src.items() if ".mtp." in n and s["numel"] == t["numel"]]

    pairs = {name: candidates(name, t) for name, t in jobs}
    needed = {n for found in pairs.values() for n in found}
    with ThreadPoolExecutor(4) as pool:
        src_values = dict(zip(sorted(needed), pool.map(lambda n: fetch(small_src[n]), sorted(needed))))
        rel_values = dict(zip([n for n, _ in jobs], pool.map(lambda j: fetch(j[1]), jobs)))
    rows = []
    for name, t in jobs:
        value = rel_values[name].reshape(-1)
        best = None
        for sname in pairs[name]:
            for tname, fn in TRANSFORMS.items():
                ref = fn(src_values[sname].reshape(-1))
                if t["dtype"] == "F16":
                    ref = ref.astype(np.float16).astype(np.float32)
                elif t["dtype"] == "BF16":
                    ref = as_array((ref.view(np.uint32) >> 16).astype(np.uint16).tobytes(), "BF16")
                tolerance = EXP_ULPS if tname == "minus_exp" else 0
                # In place, or in the converter's own order of heads (the same values, reordered).
                for order, a, b in (("in place", value, ref), ("reordered", np.sort(value), np.sort(ref))):
                    gap = ulps(a, b)
                    exact = gap <= tolerance
                    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
                    key = (exact, order == "in place", -diff)
                    if best is None or key > best["_key"]:
                        best = {"source": sname, "transform": tname, "order": order, "exact": exact,
                                "max_abs_diff": diff, "max_ulps": gap, "_key": key}
                    if exact:
                        break
        best = best or {"source": None, "transform": None, "order": None, "exact": False, "max_abs_diff": None}
        best.pop("_key", None)
        rows.append({"release": name, "dtype": t["dtype"], "numel": t["numel"], "part": part_of(name), **best})
    exact = [r for r in rows if r["exact"]]
    text = [r for r in rows if r["part"] == "language model"]
    return {"compared": len(rows), "exact": len(exact),
            "language_model": {"compared": len(text), "exact": sum(r["exact"] for r in text)},
            "vision": {"compared": len(rows) - len(text), "exact": len(exact) - sum(r["exact"] for r in text),
                       "note": "the vision tower is not part of what the task evaluates"},
            "transforms": {f"{k} {o}": n for k in TRANSFORMS for o in ("in place", "reordered")
                           if (n := sum(1 for r in exact if r["transform"] == k and r["order"] == o))},
            "not_exact": [r for r in rows if not r["exact"]], "rows": rows}


def source_tensors(repo: str, rev: str, files: dict) -> dict:
    index_url = f"{HF}/{repo}/resolve/{rev}/model.safetensors.index.json"
    with urllib.request.urlopen(index_url, timeout=120) as r:
        shards = sorted(set(json.load(r)["weight_map"].values()))
    out = {}
    for shard in shards:
        url = f"{HF}/{repo}/resolve/{rev}/{shard}"
        for name, t in safetensors_header(url).items():
            if t["dtype"] in ST_FLOAT:
                out[name] = {"url": url, "dtype": t["dtype"], "data_start": t["data_start"], "bytes": t["bytes"],
                             "numel": int(np.prod(t["shape"])) if t["shape"] else 1, "shape": t["shape"]}
    return out


def release_tensors(spec: dict, files: dict) -> tuple[dict, dict]:
    repo, rev = spec["repo"], spec["revision"]
    out, info = {}, {"files": {}}
    for path in spec["files"]:
        url = f"{HF}/{repo}/resolve/{rev}/{path}"
        info["files"][path] = files[path]
        if spec["format"] == "gguf":
            header = gguf_header(url, files[path]["size"])
            meta = header["metadata"]
            keep = {k: v for k, v in meta.items() if not k.startswith("tokenizer.") or k == "tokenizer.ggml.model"}
            info["files"][path].update(gguf_version=header["version"], metadata=keep,
                                       tensor_count=len(header["tensors"]))
            for t in header["tensors"]:
                numel = int(np.prod(t["shape"])) if t["shape"] else 1
                info.setdefault("types", {}).setdefault(t["type"], [0, 0])
                info["types"][t["type"]][0] += 1
                info["types"][t["type"]][1] += t["bytes"]
                if t["name"] in ("token_embd.weight", "output.weight"):
                    info[t["name"]] = {"type": t["type"], "shape": t["shape"], "bytes": t["bytes"]}
                if t["type"] in ("F32", "F16", "BF16"):
                    out[t["name"]] = {"url": url, "dtype": t["type"], "data_start": t["data_start"],
                                      "bytes": numel * {"F32": 4, "F16": 2, "BF16": 2}[t["type"]], "numel": numel}
        elif path.endswith(".safetensors"):
            for name, t in safetensors_header(url).items():
                info.setdefault("types", {}).setdefault(t["dtype"], [0, 0])
                info["types"][t["dtype"]][0] += 1
                info["types"][t["dtype"]][1] += t["bytes"]
                if t["dtype"] in ST_FLOAT and not re.search(r"\.(suh|svh|mcg|mul1|trellis)$", name):
                    out[name] = {"url": url, "dtype": t["dtype"], "data_start": t["data_start"], "bytes": t["bytes"],
                                 "numel": int(np.prod(t["shape"])) if t["shape"] else 1}
    return out, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("--release", required=True)
    parser.add_argument("--max-numel", type=int, default=1 << 20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = released_models.release(args.task, args.release)
    source = released_models.source_of(args.task)
    rel_files = tree(spec["repo"], spec["revision"])
    missing = [p for p in spec["files"] if p not in rel_files]
    if missing:
        raise SystemExit(f"release files absent at {spec['revision']}: {missing}")
    src_files = tree(source["repo"], source["revision"])
    rel, info = release_tensors(spec, rel_files)
    src = source_tensors(source["repo"], source["revision"], src_files)
    result = {"schema": "compression-bench-released-identity-v1", "task": args.task, "release": spec,
              "source": {**source, "weight_shards": {p: f for p, f in src_files.items() if p.endswith(".safetensors")}},
              "release_info": info, "unquantized_release_tensors": len(rel),
              "comparison": compare(rel, src, args.max_numel)}
    c = result["comparison"]
    lm = c["language_model"]
    result["identity"] = {"verdict": "same checkpoint" if lm["compared"] and lm["exact"] == lm["compared"] else "not established",
                          "basis": "every unquantized language-model tensor equals the pinned BF16 tensor",
                          "compared": lm["compared"], "exact": lm["exact"], "vision": c["vision"]}
    text = json.dumps(result, indent=1, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({"task": args.task, "release": args.release, **result["identity"],
                      "transforms": c["transforms"], "types": info.get("types"),
                      "token_embd": info.get("token_embd.weight"), "output": info.get("output.weight"),
                      "not_exact": c["not_exact"][:10]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
