#!/usr/bin/env python3
"""Build dense_B, the dense pair's ablated artifact: the selected submission's 3-bit MLPs re-quantized with per-group
MSE range search disabled; no evaluation inputs.

Run by controlled_dense_launcher.py in the platform image with the selected run's work/ at /archive/work (its
quantize.py under build/ and the cached Hessians under hess-calib/), the BF16 checkpoint at /input/model, dense_A
at /selected, dense_B at /alternate, scratch at /scratch and the study records at /record. Before building, the
selected setting is reproduced on the first MLP and must match dense_A bit for bit. Every other tensor is copied
from dense_A. Writes dense_build.json, dense_tensor_audit.json and dense_reproduction_check.json to /record.
"""
import gc
import json
from pathlib import Path
import sys
import time

from study_tensors import safetensors_index, tensor_sha256 as digest


def dense():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path[:0]=['/archive/work/build','/selected']
    import quantize as q
    from qpack.format import pack_ints
    torch.manual_seed(0);torch.set_num_threads(8)
    index=json.loads(Path('/input/model/model.safetensors.index.json').read_text())['weight_map']
    model=Path('/selected/model');target=Path('/alternate/model')
    changed={};records=[];start=time.monotonic()
    location=safetensors_index(model)
    def load_source(name):
        with safe_open(Path('/input/model')/index[name],framework='pt') as h:return h.get_tensor(name)
    def original(name):
        with safe_open(location[name],framework='pt') as h:return h.get_tensor(name)
    for li in range(64):
        for suffix,parts,hkey in q.layer_targets(li,'linear_attention' if li%4!=3 else 'full_attention')[1]:
            if not suffix.startswith('mlp.'):continue
            base=f'model.layers.{li}.{suffix}'
            cfg=json.loads((model/'config.json').read_text())['quantization_config']['layers'][suffix]
            bits,group=cfg['bits'],cfg['group'];assert bits==3 and group==64 and cfg.get('act_order') is True
            W=torch.cat([load_source(f'model.language_model.layers.{li}.{p}').float() for p in parts]).cuda()
            H=torch.load(f'/archive/work/hess-calib/H_{li}_{hkey}.pt',map_location='cuda',weights_only=True)
            if li==0 and suffix=='mlp.gate_up_proj':
                Q,S,Z,P=q.gptq(W,H,bits,group,percdamp=.01,mse=True,act_order=True)
                lo,hi=pack_ints(Q,bits)
                rebuilt={'.wlo':lo,'.scales':S.half(),'.zeros':Z.to(torch.uint8),'.wperm':P.to(torch.int64)}
                if hi is not None:rebuilt['.whi']=hi
                equal={k:torch.equal(v.cpu(),original(base+k)) for k,v in rebuilt.items()}
                assert all(equal.values()),('selected reproduction mismatch',equal)
                Path('/record/dense_reproduction_check.json').write_text(json.dumps({'tensor':base,'exact_equal':equal,'source_weights_sha256':digest(W),'hessian_sha256':digest(H)},indent=2))
                del Q,S,Z,P,lo,hi,rebuilt;gc.collect();torch.cuda.empty_cache()
            Q,S,Z,P=q.gptq(W,H,bits,group,percdamp=.01,mse=False,act_order=True)
            lo,hi=pack_ints(Q,bits)
            tensors={base+'.wlo':lo.cpu(),base+'.scales':S.half().cpu(),base+'.zeros':Z.to(torch.uint8).cpu(),base+'.wperm':P.to(torch.int64).cpu()}
            if hi is not None:tensors[base+'.whi']=hi.cpu()
            assert torch.equal(tensors[base+'.wperm'],original(base+'.wperm'))
            f=Path('/scratch')/(base+'.safetensors');save_file(tensors,str(f))
            for key in tensors:changed[key]=f
            records.append({'tensor':base,'bits':bits,'group':group,'act_order':True,'percdamp':.01,'selected_mse':True,'ablated_mse':False,'hessian_sha256':digest(H),'source_sha256':digest(W)})
            print(f'dense {li} {suffix}: {time.monotonic()-start:.1f}s',flush=True)
            del W,H,Q,S,Z,P,lo,hi,tensors;gc.collect();torch.cuda.empty_cache()
    controls=[]
    for src in sorted(model.glob('*.safetensors')):
        output={}
        with safe_open(src,framework='pt') as h:
            for key in h.keys():
                if key in changed:
                    with safe_open(changed[key],framework='pt') as r:output[key]=r.get_tensor(key)
                else:output[key]=h.get_tensor(key)
                t=h.get_tensor(key)
                assert output[key].shape==t.shape and output[key].dtype==t.dtype
                if key not in changed:assert torch.equal(output[key],t)
                controls.append({'key':key,'changed_by_intervention':key in changed,'shape':list(t.shape),'dtype':str(t.dtype),'A_sha256':digest(t),'B_sha256':digest(output[key])})
            save_file(output,str(target/src.name))
        del output;gc.collect()
    Path('/record/dense_tensor_audit.json').write_text(json.dumps(controls,indent=2))
    Path('/record/dense_build.json').write_text(json.dumps({'status':'built','seconds':time.monotonic()-start,'changed_modules':records,'total_changed_keys':len(changed),'unchanged_parameter_shapes_and_dtypes':True},indent=2))


if __name__=='__main__':
    dense()
