#!/usr/bin/env python3
"""Build B of the compensation ablation: the source routed-expert weights rounded to A's exact selected FP8 scale
grids (same channel order), without GPTQ error-compensation updates. No calibration or verification inputs.

Run by moe_gptq_study.py in the platform image with A (the decision ablation's moe_A) at /selected, B at /alternate,
the BF16 checkpoint at /input/model, the decision ablation's records at /prior and this study's records at /record.
Every non-expert tensor and every scale tensor is A's. Writes build.json and tensor_audit.json to /record.
"""
import gc
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from study_tensors import safetensors_index, tensor_sha256 as digest

sys.path.insert(0, '/selected/model')
from qserv.packing import pack_int3, unpack_int3  # noqa: E402


def fixed_grid_round(w, scales, group):
    """Same representable seven-level grid; no sequential correction updates."""
    e, out, inn = w.shape
    s = scales.float()[..., None]
    assert scales.shape == (e, out, inn // group)
    wf = w.float().reshape(e, out, inn // group, group)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    codes = torch.clamp(torch.round(wf / safe) + 3, 0, 6)
    codes = torch.where(s > 0, codes, torch.full_like(codes, 3))
    return codes.reshape(e, out, inn).to(torch.uint8)


def main():
    torch.set_num_threads(8)
    selected = Path('/selected/model')
    target = Path('/alternate/model')
    target.mkdir(parents=True, exist_ok=True)
    assert not list(target.glob('*.safetensors')), 'never overwrite an artifact'
    index = json.loads(Path('/input/model/model.safetensors.index.json').read_text())['weight_map']
    prior = json.loads(Path('/prior/moe_build.json').read_text())
    controls = {x['layer']: x for x in prior['controls']}
    selected_index = safetensors_index(selected)
    audit = []
    started = time.monotonic()
    changed_modules = 0
    for file in sorted(selected.glob('*.safetensors')):
        output = {}
        with safe_open(file, framework='pt') as f:
            for name in f.keys():
                a = f.get_tensor(name)
                changed = '.mlp.experts.' in name and not name.endswith('_scale')
                row = {'key': name, 'shape': list(a.shape), 'dtype': str(a.dtype),
                       'A_sha256': digest(a), 'changed_codes_only': changed}
                if changed:
                    layer = int(name.split('.')[2])
                    down = name.endswith('.down_proj')
                    assert down or name.endswith('.gate_up_proj')
                    group = 64 if down else 256
                    source_name = name.replace('model.layers.', 'model.language_model.layers.', 1)
                    with safe_open(Path('/input/model') / index[source_name], framework='pt') as sf:
                        original = sf.get_tensor(source_name)
                        if down:
                            assert digest(original) == controls[layer]['source_W2_sha256']
                        order = torch.tensor(controls[layer]['A_channel_order'], device='cuda')
                        full = torch.cat((order, order + len(order)))
                        scale_name = name + '_scale'
                        with safe_open(selected_index[scale_name], framework='pt') as sf_scale:
                            scales = sf_scale.get_tensor(scale_name)
                        parts = []
                        zeros = 0
                        for start in range(0, len(original), 8):
                            w = original[start:start + 8].cuda()
                            w = w[:, :, order] if down else w[:, full, :]
                            s = scales[start:start + 8].cuda()
                            zeros += int((s.float() == 0).sum())
                            codes = fixed_grid_round(w, s, group)
                            packed = pack_int3(codes)
                            if start == 0:
                                # An independent scalar expression checks packing and rounding.
                                recovered = unpack_int3(packed[:1, :2], w.shape[-1])
                                ss = s[:1, :2].float().repeat_interleave(group, dim=-1)
                                expected = ((w[:1, :2].float() / torch.where(ss > 0, ss, 1)).round() + 3).clamp(0, 6)
                                expected = torch.where(ss > 0, expected, 3).to(torch.uint8)
                                assert torch.equal(recovered, expected)
                            parts.append(packed.cpu())
                            del w, s, codes, packed
                        b = torch.cat(parts)
                        del original, parts, scales
                    changed_modules += 1
                    row.update(group_size=group, zero_scale_groups=zeros,
                               source_key=source_name, scale_values_unchanged=True,
                               channel_order_unchanged=True, rounding_check_passed=True)
                else:
                    b = a
                assert b.shape == a.shape and b.dtype == a.dtype
                row['B_sha256'] = digest(b)
                if not changed:
                    assert row['A_sha256'] == row['B_sha256']
                output[name] = b
                audit.append(row)
            save_file(output, str(target / file.name))
        print(file.name, 'completed', round(time.monotonic() - started, 1), flush=True)
        del output
        gc.collect()
        torch.cuda.empty_cache()
    assert changed_modules == 96
    Path('/record/tensor_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    Path('/record/build.json').write_text(json.dumps({
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seconds': time.monotonic() - started, 'changed_modules': changed_modules,
        'all_nonexpert_and_scale_tensors_identical': True,
        'source_down_weights_match_prior_control_hashes': True,
        'intervention': 'Round source weights to identical selected grids; no GPTQ error compensation.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
