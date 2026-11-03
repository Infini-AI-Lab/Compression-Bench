#!/usr/bin/env python3
"""Build moe_A (the selected model, reproduced) and moe_B (no expert-intermediate activation ordering) in one pass of
the selected run's own encoder; B shares A's calibration stream.

Run by controlled_moe_launcher.py in the platform image with the selected run's work/ at /archive/work (encoder at
tools/encode.py, calibration tokens under data/), the BF16 checkpoint at /input/model, moe_A at /selected, moe_B at
/alternate and the study records at /record. The encoder's source is edited before it runs (no local validation, no
BF16 reference pass, a second down-projection quantization without the ordering for B); the edited source is
recorded as /record/moe_encoder_instrumented.py. Expert IDs, routing and precision stay fixed: this is NOT a precision-shuffle test. Writes moe_build.json
and moe_packing_check.json to /record.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from study_tensors import tensor_sha256

sys.path[:0]=['/selected/model','/archive/work/tools']
source=Path('/archive/work/tools/encode.py').read_text()
# Avoid local-validation selection or a redundant reference forward. Calibration
# propagation always uses A, so B never changes a later layer's Hessian.
source=source.replace('val_chunks = [self.val_ids[i:i + bs] for i in range(0, len(self.val_ids), bs)]','val_chunks = []')
a=source.index('        # ---------- final norm + lm_head + val CE (quantized stream)')
b=source.index('        self.flush(force=True)',a)
source=source[:a]+'''        # The only final shared parameter still to write is the int8 LM head.
        c,s,deq=rtq_rowwise(self.lm_head_bf16)
        self.put("lm_head.weight",c.cpu())
        self.put("lm_head.weight_scale",s.cpu())
        ce_q=ce_ref=None
''' +source[b:]
needle='''            codes2, s2 = gptq(w2, H2[:, perm2][:, :, perm2], GROUP2, tag=f"L{li} w2")
            del H2'''
replacement='''            codes2, s2 = gptq(w2, H2[:, perm2][:, :, perm2], GROUP2, tag=f"L{li} w2")
            # Ablate only the documented intermediate-space activation ordering.
            # A and B use exactly the same original-coordinate H2 and source W2.
            w2_original=layer.mlp.experts.down_proj.data.to(DEV,torch.bfloat16)
            codes2_b,s2_b=gptq(w2_original,H2,GROUP2,tag=f"L{li} w2 no-order")
            self.alt[p+"mlp.experts.gate_up_proj"]=pack_int3(codes13).cpu()
            self.alt[p+"mlp.experts.gate_up_proj_scale"]=s13.to(FP8).cpu()
            self.alt[p+"mlp.experts.down_proj"]=pack_int3(codes2_b).cpu()
            self.alt[p+"mlp.experts.down_proj_scale"]=s2_b.to(FP8).cpu()
            self.interventions.append({"layer":li,"expert_ids_unchanged":True,
              "num_experts":self.ne,"all_expert_bits":3,"group13":GROUP13,"group2":GROUP2,
              "A_channel_order":perm2.cpu().tolist(),"B_channel_order":"identity",
              "shared_original_H2_sha256":digest(H2),"source_W2_sha256":digest(w2_original),
              "A_calibration_stream_only":True})
            del codes2_b,s2_b,w2_original
            del H2'''
assert source.count(needle)==1
source=source.replace(needle,replacement)
Path('/record/moe_encoder_instrumented.py').write_text(source)
ns={'__name__':'controlled_moe_encoder','__file__':'/archive/work/tools/encode.py'}
exec(compile(source,'/record/moe_encoder_instrumented.py','exec'),ns)

ns['digest']=tensor_sha256
# The source encoder's all-expert integer packer creates >70GiB of temporary int64
# buffers. Chunk expert batches without changing a single output bit.
pack_original=ns['pack_int3']
def pack_bounded(codes):
    return torch.cat([pack_original(codes[i:i+8]) for i in range(0,codes.shape[0],8)],dim=0)
probe=torch.randint(0,7,(10,16,3072),dtype=torch.uint8,device='cuda')
assert torch.equal(pack_original(probe),pack_bounded(probe))
Path('/record/moe_packing_check.json').write_text(json.dumps({'bit_exact':True,'chunk_experts':8,'changed_model_format':False}))
del probe
ns['pack_int3']=pack_bounded
Encoder=ns['Encoder'];put_original=Encoder.put;flush_original=Encoder.flush

def flush_b(self,force=False):
    if not self.bout:return
    total=sum(t.numel()*t.element_size() for t in self.bout.values())
    if total>3.5*2**30 or force:
        fn=f'weights-{self.bi:05d}.safetensors'
        save_file(self.bout,str(Path('/alternate/model')/fn));self.bfiles.append(fn);self.bi+=1;self.bout={};gc.collect()

def put(self,name,tensor):
    bt=self.alt.pop(name,tensor)
    assert bt.shape==tensor.shape and bt.dtype==tensor.dtype,(name,bt.shape,tensor.shape,bt.dtype,tensor.dtype)
    self.bout[name]=bt.detach().cpu().contiguous()
    flush_b(self)
    put_original(self,name,tensor)
Encoder.put=put
args=SimpleNamespace(src='/input/model',out='/selected',cal='/archive/work/data/cal_mix3.npy',val='/archive/work/data/val.npy',chunk_seqs=8,act_order=1,group13=256,group2=64,limit_layers=None,reference=0)
torch.set_num_threads(8);torch.manual_seed(0);start=time.monotonic()
enc=Encoder(args);enc.alt={};enc.bout={};enc.bi=0;enc.bfiles=[];enc.interventions=[]
with torch.no_grad():enc.run()
flush_b(enc,True)
assert not enc.alt
assert enc.bfiles==enc.shard_names
Path('/alternate/model/shards.json').write_text(json.dumps({'files':enc.bfiles}))
Path('/alternate/model/config.json').write_bytes(Path('/selected/model/config.json').read_bytes())
Path('/record/moe_build.json').write_text(json.dumps({'status':'built','seconds':time.monotonic()-start,'decision':'expert-intermediate activation ordering','not_precision_allocation':True,'controls':enc.interventions,'calibration_sha256':hashlib.sha256(Path(args.cal).read_bytes()).hexdigest(),'source_encoder_sha256':hashlib.sha256(Path('/archive/work/tools/encode.py').read_bytes()).hexdigest()},indent=2))
print('MOE PAIR BUILT',time.monotonic()-start,flush=True)
