#!/usr/bin/env python3
"""CPU-only comparison of a reproduced model's tensors with the selected submission's, tensor by tensor.

    source_reproduction_audit.py [--original DIR] [--reproduced DIR] [--out FILE]

Defaults are the container mounts controlled_moe_launcher.py uses (/original, /reproduced,
/record/moe_source_reproduction_audit.json). Both directories must hold the same tensor names; each tensor's
shape, dtype and SHA-256 of its raw bytes are recorded with exact_equal per tensor.
"""
import argparse
import json
from pathlib import Path

from safetensors import safe_open

from study_tensors import safetensors_index, tensor_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--original', default='/original')
    p.add_argument('--reproduced', default='/reproduced')
    p.add_argument('--out', default='/record/moe_source_reproduction_audit.json')
    a = p.parse_args()
    orig, rep = safetensors_index(a.original), safetensors_index(a.reproduced)
    assert orig.keys() == rep.keys()
    rows = []
    for k in sorted(orig):
        with safe_open(orig[k], framework='pt') as fa, safe_open(rep[k], framework='pt') as fb:
            x, y = fa.get_tensor(k), fb.get_tensor(k)
            assert x.shape == y.shape and x.dtype == y.dtype
            ha, hb = tensor_sha256(x), tensor_sha256(y)
            rows.append(dict(key=k, shape=list(x.shape), dtype=str(x.dtype), original_sha256=ha, reproduced_sha256=hb,
                             exact_equal=ha == hb))
    Path(a.out).write_text(json.dumps(dict(all_exact_equal=all(r['exact_equal'] for r in rows), tensor_count=len(rows),
                                           exact_equal_count=sum(r['exact_equal'] for r in rows), tensors=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
