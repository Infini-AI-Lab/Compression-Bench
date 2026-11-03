#!/usr/bin/env python3
"""CPU-only tensor control audit of the MoE pair before fresh verification; no quality scoring.

Run by controlled_parallel.py in the platform image with moe_A at /a, moe_B at /b and the study records at /record.
Every tensor must keep its shape and dtype; every non-expert tensor must be bit-identical; the routed experts'
gate/up tensors must differ only by the intermediate-row ordering recorded in /record/moe_build.json.
Writes /record/moe_tensor_audit.json.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from study_tensors import safetensors_index, tensor_sha256


def main():
    ia=safetensors_index('/a/model');ib=safetensors_index('/b/model');assert ia.keys()==ib.keys()
    info=json.loads(Path('/record/moe_build.json').read_text());perm={r['layer']:torch.tensor(r['A_channel_order']) for r in info['controls']}
    audit=[]
    for k in ia:
        with safe_open(ia[k],framework='pt') as fa,safe_open(ib[k],framework='pt') as fb:
            ta,tb=fa.get_tensor(k),fb.get_tensor(k);assert ta.shape==tb.shape and ta.dtype==tb.dtype,k
            ha,hb=tensor_sha256(ta),tensor_sha256(tb);expert='.mlp.experts.' in k
            if not expert:assert ha==hb,('unintended parameter change',k)
            row={'key':k,'shape':list(ta.shape),'dtype':str(ta.dtype),'A_sha256':ha,'B_sha256':hb,'expert_tensor':expert}
            if '.mlp.experts.gate_up_proj' in k:
                li=int(k.split('.')[2]);p=perm[li];inv=torch.argsort(p)
                # Only the intermediate row permutation differs in gate/up.
                full=torch.cat([inv,inv+len(inv)])
                av=ta.view(torch.uint8) if ta.dtype==torch.float8_e4m3fn else ta
                bv=tb.view(torch.uint8) if tb.dtype==torch.float8_e4m3fn else tb
                assert torch.equal(av[:,full],bv),('gate/up values changed beyond ordering',k)
                row['identical_after_original_coordinate_restore']=True
            audit.append(row)
    Path('/record/moe_tensor_audit.json').write_text(json.dumps({'status':'passed','tensor_controls':audit,'expert_ids_and_router_weights_unchanged':True,'all_shapes_and_dtypes_unchanged':True},indent=2)+'\n')


if __name__=='__main__':main()
