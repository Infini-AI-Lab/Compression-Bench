#!/usr/bin/env python3
"""CPU-only semantic check that the compensation ablation's B represents different weights from A, not merely
different codes on zero-valued grids.

Run by moe_gptq_study.py in the platform image with A at /a, B at /b and the study records at /record. Samples the
first expert's first eight output rows of each of the 96 routed-expert code tensors and counts codes that differ on
a positive scale. Writes /record/representation_audit.json.
"""
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

from study_tensors import safetensors_index

sys.path.insert(0, '/a/model')
from qserv.packing import unpack_int3  # noqa: E402

torch.set_num_threads(4)
a, b = safetensors_index('/a/model'), safetensors_index('/b/model')
assert a.keys() == b.keys()
rows = []
for name in sorted(a):
    if '.mlp.experts.' not in name or name.endswith('_scale'):
        continue
    group = 64 if name.endswith('.down_proj') else 256
    with safe_open(a[name], framework='pt') as fa, safe_open(b[name], framework='pt') as fb:
        aa = fa.get_slice(name)[:1, :8]
        bb = fb.get_slice(name)[:1, :8]
    with safe_open(a[name + '_scale'], framework='pt') as fs:
        scale = fs.get_slice(name + '_scale')[:1, :8].float()
    k = scale.shape[-1] * group
    qa, qb = unpack_int3(aa, k), unpack_int3(bb, k)
    positive = scale.repeat_interleave(group, dim=-1) > 0
    count = int(((qa != qb) & positive).sum())
    rows.append({'key': name, 'sample': 'first expert, first eight output rows, all input columns',
                 'positive_grid_codes_tested': int(positive.sum()),
                 'different_represented_weights': count})
assert len(rows) == 96
assert sum(r['different_represented_weights'] for r in rows) > 0
Path('/record/representation_audit.json').write_text(json.dumps({
    'passed': True, 'sampled_modules': len(rows),
    'sample_selection_fixed': True,
    'different_represented_weights': sum(r['different_represented_weights'] for r in rows),
    'not_merely_different_zero_scale_codes': True, 'modules': rows}, indent=2) + '\n')
