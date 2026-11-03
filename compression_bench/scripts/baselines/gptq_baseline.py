#!/usr/bin/env python3
"""The GPTQ compression baseline of the compression-bench tasks.

GPTQ (Frantar et al., ICLR 2023) adapted to each task's model with one fixed recipe
and no per-task search (gptq_core.py has the algorithm, gptq_codec.py the format):

  calibration  128 random 2048-token C4 segments, the official draw (seed 0) from
               c4-train.00000-of-01024, each model's own tokenizer;
  GPTQ         official defaults: 1% dampening, blocks of 128 columns, natural order,
               asymmetric min-max grid, groups of 128 input columns, sequential over
               blocks (block i is calibrated on the outputs of quantized blocks 0..i-1;
               all of a block's layers take their Hessians from one forward pass);
  bits         the largest uniform b in (4, 3, 2) for every linear layer of the decoder
               blocks such that the package fits the task's limit; embeddings and LM
               head BF16 if it still fits, else 8-bit, else 4-bit (embedding by
               round-to-nearest on the same grid, LM head by GPTQ on the final-norm
               outputs of the quantized model);
  MoE          every routed expert is its own linear layer, its Hessian from the
               calibration tokens the model's router sends it; an expert no token
               reaches is rounded to nearest (H = I). Routers, shared-expert gates,
               Gated-DeltaNet a/b projections, convolutions, SSM parameters and norms
               stay BF16.

Deviations from the official code, each forced: those of gptq_core.py (a column no calibration
input reaches keeps its weights; float16 scales), the MoE rule above (the GPTQ paper predates
mixture-of-experts), groups of 128 at every bit width, and calibration forwards of 8 sequences
at a time (the math is unchanged).

The checkpoint is measured with its packed weights served natively by vLLM
(gptq_packed_runtime.py): baseline_latency.py --kind gptq for decode latency, PPL and
accuracy, released_baseline.py --release gptq for the task's full hidden verdict.

    gptq_baseline.py plan <task>                      # the recipe and its package accounting
    gptq_baseline.py quantize <task> --gpu N          # writes .../baselines/gptq/<task>/model
    gptq_baseline.py validate <task> --gpu N          # layer-by-layer path vs the model's own forward

Every card used is held with the harness's own claim for as long as it is used.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import gptq_codec  # noqa: E402
import gptq_core  # noqa: E402

REPO = HERE.parents[3]
OUT_ROOT = REPO / "compression_bench" / "evolve_space" / "baselines" / "gptq"
C4 = OUT_ROOT / "data" / "c4-train.00000-of-01024.json.gz"
C4_SHA256 = "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"
C4_SOURCE = "allenai/c4 en/c4-train.00000-of-01024.json.gz @ 1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
IMAGE = "vllm/vllm-openai:v0.32.0"

#: Per architecture: where its text model lives in the checkpoint and which tensors GPTQ quantizes.
LAYOUTS = {
    "qwen3_5": {
        "config_key": "text_config",
        "text": r"^(model\.language_model\.|lm_head\.weight$)",
        "quantize": (r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|"
                     r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"),
        "embed": "model.language_model.embed_tokens.weight", "head": "lm_head.weight",
        "final_norm": "model.language_model.norm.weight", "layers": "model.language_model.layers.{}.",
    },
    "qwen3_5_moe": {
        "config_key": "text_config",
        "text": r"^(model\.language_model\.|lm_head\.weight$)",
        "quantize": (r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj\.weight|"
                     r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight|"
                     r"mlp\.shared_expert\.(gate|up|down)_proj\.weight|mlp\.experts\.(gate_up_proj|down_proj))$"),
        "embed": "model.language_model.embed_tokens.weight", "head": "lm_head.weight",
        "final_norm": "model.language_model.norm.weight", "layers": "model.language_model.layers.{}.",
    },
    "nemotron_h": {
        "config_key": "llm_config",
        "text": r"^language_model\.(backbone\.|lm_head\.weight$)",
        "quantize": (r"^language_model\.backbone\.layers\.\d+\.mixer\.((q|k|v|o)_proj|in_proj|out_proj|"
                     r"fc1_latent_proj|fc2_latent_proj|shared_experts\.(up|down)_proj|"
                     r"experts\.\d+\.(up|down)_proj)\.weight$"),
        "embed": "language_model.backbone.embeddings.weight", "head": "language_model.lm_head.weight",
        "final_norm": "language_model.backbone.norm_f.weight", "layers": "language_model.backbone.layers.{}.",
        # The task's starter builds vLLM's Nemotron-H Omni class, which reads the processor configs.
        "runtime_files": ["preprocessor_config.json", "processor_config.json"],
    },
}
SMALL_FILES = ("config.json", "generation_config.json")


# =========================================================================== host side

def task_dir(task: str) -> Path:
    path = Path(task)
    if not path.is_dir():
        path = REPO / "compression_tasks" / (task if task.endswith("_1xH200NVL") else f"{task}_standard_1xH200NVL")
    if not (path / "contract.yaml").is_file():
        raise SystemExit(f"no task at {path}")
    return path.resolve()


def contract_of(task: Path) -> dict:
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    return render.load_contract(task)


def model_dir_of(task: Path) -> Path:
    out = subprocess.check_output([sys.executable, str(REPO / "compression_bench" / "scripts" / "model_for_task.py"),
                                   str(task)], text=True).strip()
    return Path(out)


def architecture(model: Path) -> str:
    config = json.loads((model / "config.json").read_text())
    for key in ("text_config", "llm_config"):
        if key in config:
            return config[key]["model_type"].removesuffix("_text")
    return config["model_type"]


def tokenizer_files(contract: dict) -> list[str]:
    return list(contract["evaluation"]["official_tokenizer_files"])


def small_files(model: Path, contract: dict) -> dict[str, int]:
    names = list(SMALL_FILES) + tokenizer_files(contract) + LAYOUTS[architecture(model)].get("runtime_files", [])
    return {name: (model / name).stat().st_size for name in names if (model / name).is_file()}


def plan(task: Path) -> dict:
    contract = contract_of(task)
    model = model_dir_of(task)
    arch = architecture(model)
    layout = LAYOUTS[arch]
    tensors = gptq_core.checkpoint_tensors(model)
    recipe = gptq_core.choose_recipe(tensors, layout, contract["capacity_gib"], small_files(model, contract))
    return {"task": contract["task_id"], "task_dir": str(task), "model_id": contract["model"]["id"],
            "model_revision": contract["model"]["revision"], "model_dir": str(model), "arch": arch,
            "capacity_gib": contract["capacity_gib"], "block_bits": recipe["block_bits"],
            "head_bits": recipe["head_bits"], "planned": recipe["accounting"], "tried": recipe["tried"]}


def out_dir(task: Path) -> Path:
    return OUT_ROOT / task.name.removesuffix("_standard_1xH200NVL")


class Cards:
    """Hold the harness's claim on every named card while the work runs."""

    def __init__(self, gpus: str, run_id: str):
        sys.path.insert(0, str(REPO / "compression_bench" / "harness"))
        from lifecycle import gpus as lifecycle_gpus
        self.gpus = lifecycle_gpus
        self.indices = [int(g) for g in gpus.split(",")]
        self.run_id = run_id
        self.claims = []

    def __enter__(self):
        for index in self.indices:
            why = self.gpus.why_busy(index)
            if why:
                self.release()
                raise SystemExit(why)
            claim = self.gpus.Claim(index, run_id=self.run_id)
            if not claim.acquire():
                self.release()
                raise SystemExit(f"GPU {index} is claimed by another run")
            self.claims.append(claim)
        return self

    def release(self):
        for claim in self.claims:
            claim.release()
        self.claims = []

    def __exit__(self, *_):
        self.release()


def mounts_for(*paths: Path) -> list[str]:
    """Bind each path read-only at its own path (a snapshot's repository, for its blob links)."""
    seen, args = set(), []
    for path in paths:
        path = Path(path)
        root = path.parents[1] if path.parent.name == "snapshots" else path
        real = Path(os.path.realpath(root))
        roots = {root, real}
        if path.is_dir():
            roots.update(p.resolve().parent for p in path.glob("*.safetensors") if p.is_symlink())
        for p in sorted(roots):
            if p not in seen:
                seen.add(p)
                args += ["-v", f"{p}:{p}:ro"]
    return args


def docker(gpus: str, name: str, mounts: list[str], args: list[str], log: Path | None = None) -> int:
    command = ["docker", "run", "--rm", "--name", name, "--gpus", f'"device={gpus}"', "--ipc=host",
               "--network=none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
               "-v", f"{REPO}:{REPO}:ro", *mounts, "--entrypoint", "python3", IMAGE,
               str(HERE), *args]
    print(" ".join(command), flush=True)
    if log is None:
        return subprocess.run(command, check=False).returncode
    with log.open("a") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def cmd_plan(args) -> int:
    print(json.dumps(plan(task_dir(args.task)), indent=1))
    return 0


def cmd_quantize(args) -> int:
    task = task_dir(args.task)
    spec = plan(task)
    out = (Path(args.output_root).resolve() / task.name.removesuffix("_standard_1xH200NVL")
           if getattr(args, "output_root", None) else out_dir(task))
    model_out = out / "model"
    if model_out.exists() and any(model_out.iterdir()):
        raise SystemExit(f"{model_out} exists; remove it first to re-quantize")
    model_out.mkdir(parents=True, exist_ok=True)
    contract = contract_of(task)
    spec.update(out=str(out), model_out=str(model_out), c4=str(C4), batch=args.batch,
                tokenizer_dir=str(task / "assets" / "tokenizer"), tokenizer_files=tokenizer_files(contract),
                expert_chunk=args.expert_chunk)
    if gptq_core.hashlib.sha256(C4.read_bytes()).hexdigest() != C4_SHA256:
        raise SystemExit(f"{C4} is not the pinned C4 shard")
    (out / "quantize-spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"gptq-baseline-{task.name.removesuffix('_standard_1xH200NVL')}-{stamp.lower()}".replace("_", "-")
    mounts = mounts_for(Path(spec["model_dir"])) + ["-v", f"{out}:{out}"]
    started = time.time()
    with Cards(args.gpu, f"gptq-baseline-{spec['task']}"):
        code = docker(args.gpu, name, mounts, ["--inside", "quantize", str(out / "quantize-spec.json")],
                      log=out / "quantize.log")
    report = out / "quantize-report.json"
    if report.is_file():
        data = json.loads(report.read_text())
        data["host_wall_seconds"] = round(time.time() - started, 1)
        report.write_text(json.dumps(data, indent=1) + "\n")
    print(f"quantize exit {code}; log {out / 'quantize.log'}")
    return code


def cmd_validate(args) -> int:
    task = task_dir(args.task)
    spec = plan(task)
    out = out_dir(task)
    out.mkdir(parents=True, exist_ok=True)
    spec.update(out=str(out), batch=args.batch, layers=args.layers)
    (out / "validate-spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"gptq-validate-{task.name.removesuffix('_standard_1xH200NVL')}-{stamp.lower()}".replace("_", "-")
    with Cards(args.gpu, f"gptq-baseline-validate-{spec['task']}"):
        return docker(args.gpu, name, mounts_for(Path(spec["model_dir"])) + ["-v", f"{out}:{out}"],
                      ["--inside", "validate", str(out / "validate-spec.json")], log=out / "validate.log")


# =========================================================================== container side

def decode_config(value):
    """Transformers' JSON tags for non-finite floats ({"__float__": "Infinity"})."""
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float({"Infinity": "inf", "-Infinity": "-inf", "NaN": "nan"}[value["__float__"]])
        return {k: decode_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_config(v) for v in value]
    return value


class Checkpoint:
    """Reads single tensors of a sharded safetensors checkpoint."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open
        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self.handles[shard].get_tensor(name)

    def names(self, prefix):
        return [n for n in self.map if n.startswith(prefix)]


class Capture:
    """Hessians H = 2/n sum(x x^T) (float32) of a block's quantized layers during one pass."""

    def __init__(self, device):
        self.device = device
        self.sums, self.counts, self.hooks = {}, {}, []
        self.batch_sequences = 0
        self.active = True

    def linear(self, key, module):
        columns = module.weight.shape[1]
        self.sums[key] = torch.zeros((1, columns, columns), dtype=torch.float32, device=self.device)
        self.counts[key] = torch.zeros(1, dtype=torch.float64)

        def hook(mod, inputs, output):
            if not self.active:
                return
            x = inputs[0].reshape(-1, columns).float()
            self.sums[key][0].addmm_(x.t(), x)
            self.counts[key] += self.batch_sequences       # one official add_batch sample per sequence
        self.hooks.append(module.register_forward_hook(hook))

    def experts(self, key, experts: int, columns: int):
        self.sums[key] = torch.zeros((experts, columns, columns), dtype=torch.float32, device=self.device)
        self.counts[key] = torch.zeros(experts, dtype=torch.float64)

    def add_expert(self, key, expert: int, x):
        xf = x.float()
        self.sums[key][expert].addmm_(xf.t(), xf)

    def add_expert_sequences(self, keys, top_k_index, experts: int, batch: int):
        present = torch.zeros((batch, experts), dtype=torch.bool, device=top_k_index.device)
        present.scatter_(1, top_k_index.reshape(batch, -1), True)
        per_expert = present.sum(dim=0).double().cpu()
        for key in keys:
            self.counts[key] += per_expert

    def hessian(self, key):
        """2/n * sum (n = sequences reaching the layer); an expert no token reaches gets H = 0 (all dead)."""
        H = self.sums.pop(key)
        n = self.counts.pop(key)
        scale = torch.where(n > 0, 2.0 / n.clamp_min(1), torch.zeros_like(n)).to(device=H.device, dtype=H.dtype)
        H *= scale[:, None, None]
        return H, n

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def qwen_experts_forward(capture, prefix):
    """Qwen3_5MoeExperts' eager forward, capturing each expert's gate_up and down Hessians."""
    def forward(self, hidden_states, top_k_index, top_k_weights):
        E = self.num_experts
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=E + 1).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        if capture is not None and capture.active:
            capture.add_expert_sequences([prefix + "gate_up_proj", prefix + "down_proj"], top_k_index, E,
                                         capture.batch_sequences)
        for expert_idx in hit.view(-1).tolist():
            if expert_idx == E:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            inter = self.act_fn(gate) * up
            if capture is not None and capture.active:
                capture.add_expert(prefix + "gate_up_proj", expert_idx, current)
                capture.add_expert(prefix + "down_proj", expert_idx, inter)
            out = torch.nn.functional.linear(inter, self.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final
    return forward


def nemotron_experts_forward(capture, prefix):
    """NemotronHExperts' eager forward, capturing each expert's up and down Hessians."""
    def forward(self, hidden_states, top_k_index, top_k_weights):
        E = self.num_experts
        final = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=E + 1).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        if capture is not None and capture.active:
            capture.add_expert_sequences([prefix + "up_proj", prefix + "down_proj"], top_k_index, E,
                                         capture.batch_sequences)
        for expert_idx in hit.tolist():
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            inter = self.act_fn(torch.nn.functional.linear(current, self.up_proj[expert_idx]))
            if capture is not None and capture.active:
                capture.add_expert(prefix + "up_proj", expert_idx, current)
                capture.add_expert(prefix + "down_proj", expert_idx, inter)
            out = torch.nn.functional.linear(inter, self.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final.to(hidden_states.dtype)
    return forward


class Adapter:
    """One architecture's decoder blocks, built one at a time from the checkpoint."""

    kept_linears: tuple[str, ...] = ()

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.layout = LAYOUTS[spec["arch"]]
        self.ckpt = Checkpoint(spec["model_dir"])
        full = json.loads((Path(spec["model_dir"]) / "config.json").read_text())
        self.text_config = decode_config(full[self.layout["config_key"]])
        self.masks = {}

    def block_prefix(self, i):
        return self.layout["layers"].format(i)

    def state_dict(self, i):
        prefix = self.block_prefix(i)
        return {name[len(prefix):]: self.ckpt.get(name) for name in self.ckpt.names(prefix)}

    def build(self, i):
        with torch.device("meta"):
            layer = self.block_class(self.config, i)
        layer.load_state_dict(self.state_dict(i), strict=True, assign=True)
        layer = layer.to(self.device)
        layer.eval()
        for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
            if tensor.is_meta:
                raise RuntimeError(f"block {i}: {name} was not loaded")
        return layer

    def targets(self, layer, i):
        """[(checkpoint name, module or (experts module, attr), kind)] of every tensor GPTQ quantizes."""
        prefix = self.block_prefix(i)
        found, quant = [], re.compile(self.layout["quantize"])
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                full = f"{prefix}{name}.weight"
                if quant.search(full):
                    found.append((full, module, "linear"))
                elif not name.endswith(self.kept_linears):
                    raise RuntimeError(f"unexpected linear layer {full}")
        found += self.expert_targets(layer, i)
        return found

    def expert_targets(self, layer, i):
        return []


class QwenAdapter(Adapter):
    kept_linears = ("in_proj_a", "in_proj_b", "shared_expert_gate")

    def __init__(self, spec, device):
        super().__init__(spec, device)
        if spec["arch"] == "qwen3_5_moe":
            from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe as C, modeling_qwen3_5_moe as M
            self.config = C.Qwen3_5MoeTextConfig(**self.text_config)
            self.block_class, rotary = M.Qwen3_5MoeDecoderLayer, M.Qwen3_5MoeTextRotaryEmbedding
        else:
            from transformers.models.qwen3_5 import configuration_qwen3_5 as C, modeling_qwen3_5 as M
            self.config = C.Qwen3_5TextConfig(**self.text_config)
            self.block_class, rotary = M.Qwen3_5DecoderLayer, M.Qwen3_5TextRotaryEmbedding
        self.config._attn_implementation = "sdpa"
        self.M = M
        self.rotary = rotary(config=self.config).to(device)
        self.num_layers = self.config.num_hidden_layers
        self.final_norm_class = M.Qwen3_5MoeRMSNorm if spec["arch"] == "qwen3_5_moe" else M.Qwen3_5RMSNorm

    def context(self, x):
        """Qwen3_5(Moe)TextModel.forward's position ids, masks and rotary embeddings for a batch."""
        key = tuple(x.shape)
        if key not in self.masks:
            B, T = x.shape[:2]
            position_ids = torch.arange(T, device=x.device).view(1, 1, -1).expand(4, B, -1)
            text_position_ids, rope_ids = position_ids[0], position_ids[1:]
            kwargs = {"config": self.config, "inputs_embeds": x, "attention_mask": None,
                      "past_key_values": None, "position_ids": text_position_ids}
            masks = {"full_attention": self.M.create_causal_mask(**kwargs),
                     "linear_attention": self.M.create_recurrent_attention_mask(**kwargs)}
            self.masks[key] = (masks, text_position_ids, self.rotary(x, rope_ids))
        return self.masks[key]

    def forward(self, layer, i, x):
        masks, text_position_ids, position_embeddings = self.context(x)
        return layer(x, position_embeddings=position_embeddings,
                     attention_mask=masks[self.config.layer_types[i]], position_ids=text_position_ids,
                     past_key_values=None, use_cache=False)

    def expert_targets(self, layer, i):
        if self.spec["arch"] != "qwen3_5_moe":
            return []
        prefix = self.block_prefix(i) + "mlp.experts."
        experts = layer.mlp.experts
        return [(prefix + "gate_up_proj", (experts, "gate_up_proj"), "experts"),
                (prefix + "down_proj", (experts, "down_proj"), "experts")]

    def install_capture(self, layer, i, capture):
        if self.spec["arch"] == "qwen3_5_moe":
            experts = layer.mlp.experts
            prefix = self.block_prefix(i) + "mlp.experts."
            experts.forward = qwen_experts_forward(capture, prefix).__get__(experts)

    def split_name(self, name):
        return None


class NemotronAdapter(Adapter):
    kept_linears = ()

    def __init__(self, spec, device):
        super().__init__(spec, device)
        from transformers.models.nemotron_h import configuration_nemotron_h as C, modeling_nemotron_h as M
        self.config = C.NemotronHConfig(**self.text_config)
        self.config._attn_implementation = "sdpa"
        self.M = M
        self.block_class = M.NemotronHBlock
        self.num_layers = self.config.num_hidden_layers
        self.final_norm_class = M.NemotronHRMSNorm

    def state_dict(self, i):
        prefix = self.block_prefix(i)
        experts = re.compile(r"^mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")
        out, stacked = {}, {"up": {}, "down": {}}
        for name in self.ckpt.names(prefix):
            short = name[len(prefix):]
            hit = experts.match(short)
            if hit:
                stacked[hit.group(2)][int(hit.group(1))] = self.ckpt.get(name)
            else:
                out[short] = self.ckpt.get(name)
        for proj, tensors in stacked.items():
            if tensors:
                if sorted(tensors) != list(range(len(tensors))):
                    raise RuntimeError(f"block {i}: missing {proj}_proj experts")
                out[f"mixer.experts.{proj}_proj"] = torch.stack([tensors[e] for e in range(len(tensors))])
        return out

    def context(self, x, block_type):
        key = (tuple(x.shape), block_type)
        if key not in self.masks:
            T = x.shape[1]
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            kwargs = {"config": self.config, "inputs_embeds": x, "attention_mask": None,
                      "past_key_values": None, "position_ids": position_ids}
            masks = {"full_attention": self.M.create_causal_mask(**kwargs),
                     "linear_attention": self.M.create_recurrent_attention_mask(**kwargs)}
            self.masks[key] = (masks.get(block_type), position_ids)
        return self.masks[key]

    def forward(self, layer, i, x):
        mask, position_ids = self.context(x, layer.block_type)
        return layer(x, attention_mask=mask, position_ids=position_ids, past_key_values=None, use_cache=False)

    def expert_targets(self, layer, i):
        if layer.block_type != "moe":
            return []
        prefix = self.block_prefix(i) + "mixer.experts."
        experts = layer.mixer.experts
        return [(prefix + "up_proj", (experts, "up_proj"), "experts"),
                (prefix + "down_proj", (experts, "down_proj"), "experts")]

    def install_capture(self, layer, i, capture):
        if layer.block_type == "moe":
            experts = layer.mixer.experts
            prefix = self.block_prefix(i) + "mixer.experts."
            experts.forward = nemotron_experts_forward(capture, prefix).__get__(experts)

    def split_name(self, name):
        """The stacked expert tensor -> its per-expert checkpoint names."""
        hit = re.match(r"^(.*\.mixer\.experts\.)(up|down)_proj$", name)
        return f"{hit.group(1)}{{e}}.{hit.group(2)}_proj.weight" if hit else None


def adapter_for(spec, device):
    return NemotronAdapter(spec, device) if spec["arch"] == "nemotron_h" else QwenAdapter(spec, device)


def quantize_target(W, H, bits, chunk, check=True):
    """GPTQ over a stack [E, out, in] in chunks of experts -> (Q bf16 on device, codes, scales, zeros, dead)."""
    E = W.shape[0]
    outs = {"Q": [], "codes": [], "scales": [], "zeros": [], "dead": []}
    for start in range(0, E, chunk):
        sl = slice(start, start + chunk)
        Q, codes, scales, zeros, dead = gptq_core.gptq(W[sl], H[sl].clone(), bits)
        Qb = Q.to(torch.bfloat16)
        if check:
            # The stored triplet must decode to exactly the weights GPTQ produced.
            decoded = gptq_codec.dequantize_rows(codes, scales, zeros, gptq_core.GROUP_SIZE).to(torch.bfloat16)
            if not torch.equal(decoded, Qb):
                raise RuntimeError("codec round trip differs from GPTQ's weights")
        del Q
        for k, v in (("Q", Qb), ("codes", codes), ("scales", scales), ("zeros", zeros), ("dead", dead)):
            outs[k].append(v)
    return {k: torch.cat(v) for k, v in outs.items()}


def packed_entry(name, shape, bits, result, split=None):
    tensors = {name + gptq_codec.SUFFIXES["codes"]: gptq_codec.pack(result["codes"], bits).cpu(),
               name + gptq_codec.SUFFIXES["scales"]: result["scales"].reshape(*shape[:-1], -1).cpu().contiguous(),
               name + gptq_codec.SUFFIXES["zeros"]: gptq_codec.pack(result["zeros"], bits).cpu()}
    entry = {"name": name, "shape": list(shape), "bits": bits, "group_size": gptq_core.GROUP_SIZE}
    if split:
        entry["split"] = split
    return entry, tensors


def inside_quantize(spec_path: str) -> None:
    global torch
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False          # as the official gptq.py
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads(Path(spec_path).read_text())
    out, model_out = Path(spec["out"]), Path(spec["model_out"])
    device = torch.device("cuda")
    started = time.time()
    log = lambda *a: print(f"[{time.time() - started:8.1f}s]", *a, flush=True)  # noqa: E731
    adapter = adapter_for(spec, device)
    layout, bits, head_bits = adapter.layout, spec["block_bits"], spec["head_bits"]
    report = {"task": spec["task"], "arch": spec["arch"], "status": "running", "gpu": torch.cuda.get_device_name(),
              "recipe": {"method": "GPTQ (Frantar et al., ICLR 2023), official defaults",
                         "block_bits": bits, "group_size": gptq_core.GROUP_SIZE, "block_size": gptq_core.BLOCK_SIZE,
                         "percdamp": gptq_core.PERCDAMP, "act_order": False, "true_sequential": False,
                         "symmetric": False, "scale_dtype": "float16",
                         "embed_head": "bf16" if head_bits is None else f"{head_bits}-bit",
                         "embedding": "bf16" if head_bits is None else f"round-to-nearest {head_bits}-bit, groups of 128",
                         "lm_head": "bf16" if head_bits is None else f"GPTQ {head_bits}-bit, groups of 128",
                         "calibration": f"{gptq_core.NSAMPLES} x {gptq_core.SEQLEN} tokens, C4, seed {gptq_core.SEED}",
                         "selection_rule": "largest uniform b in (4,3,2) whose package fits; embed/head bf16, else 8, else 4",
                         "tried": spec["tried"]},
              "layers": []}

    # ---- calibration: the official draw
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_dir"])
    texts = gptq_core.c4_documents(spec["c4"])
    windows, draws = gptq_core.sample_c4(texts, lambda text: tokenizer(text).input_ids)
    del texts
    report["calibration"] = {"source": C4_SOURCE, "sha256": C4_SHA256, "documents_drawn": draws,
                             "token_sha256": gptq_core.token_hash(windows), "samples": len(windows),
                             "seqlen": gptq_core.SEQLEN, "tokenizer": spec["tokenizer_dir"]}
    log("calibration", report["calibration"]["token_sha256"])
    calib = torch.tensor(windows, dtype=torch.long)

    meta = {"format": gptq_codec.FORMAT, "method": "GPTQ", "group_size": gptq_core.GROUP_SIZE,
            "packing": gptq_codec.__doc__.split("\n\n")[1], "tensors": []}
    shards = {}

    def save(tensors, filename):
        save_file(tensors, str(model_out / filename), metadata={"format": "pt"})
        for name in tensors:
            shards[name] = filename

    # ---- embedding (round-to-nearest when quantized: a lookup table)
    embed = adapter.ckpt.get(layout["embed"]).to(device)
    if head_bits is not None:
        Q, codes, scales, zeros = gptq_core.rtn(embed, head_bits)
        Qb = Q.to(torch.bfloat16)
        decoded = gptq_codec.dequantize_rows(codes, scales, zeros, gptq_core.GROUP_SIZE).to(torch.bfloat16)
        if not torch.equal(decoded, Qb):
            raise RuntimeError("embedding codec round trip differs")
        entry, tensors = packed_entry(layout["embed"], embed.shape, head_bits,
                                      {"codes": codes, "scales": scales, "zeros": zeros})
        meta["tensors"].append(entry)
        save(tensors, "gptq-embed.safetensors")
        embed = Qb
        del Q, codes, scales, zeros
    hidden = embed.shape[1]
    nsamples, seqlen, batch = calib.shape[0], calib.shape[1], spec["batch"]
    inps = torch.empty((nsamples, seqlen, hidden), dtype=torch.bfloat16, device=device)
    outs = torch.empty_like(inps)
    for s in range(0, nsamples, batch):
        inps[s:s + batch] = F.embedding(calib[s:s + batch].to(device), embed)
    del embed
    torch.cuda.empty_cache()
    log("embeddings ready", tuple(inps.shape))

    dead_total, unhit_total = 0, 0
    for i in range(adapter.num_layers):
        t0 = time.time()
        layer = adapter.build(i)
        targets = adapter.targets(layer, i)
        capture = Capture(device)
        for name, module, kind in targets:
            if kind == "linear":
                capture.linear(name, module)
            else:
                experts, attr = module
                param = getattr(experts, attr)
                capture.experts(name, param.shape[0], param.shape[2])
        adapter.install_capture(layer, i, capture)
        with torch.no_grad():
            for s in range(0, nsamples, batch):
                capture.batch_sequences = inps[s:s + batch].shape[0]
                adapter.forward(layer, i, inps[s:s + batch])
        capture.remove()
        capture.active = False
        t1 = time.time()
        layer_info = {"layer": i, "type": getattr(layer, "block_type", None), "tensors": []}
        layer_tensors = {}
        for name, module, kind in targets:
            H, n = capture.hessian(name)
            if kind == "linear":
                W = module.weight.data.unsqueeze(0)
            else:
                experts, attr = module
                W = getattr(experts, attr).data
            result = quantize_target(W, H, bits, spec["expert_chunk"] if kind == "experts" else 1)
            del H
            if kind == "linear":
                module.weight.data = result["Q"][0]
            else:
                getattr(experts, attr).data = result["Q"]
            shape = list(W.shape[1:]) if kind == "linear" else list(W.shape)
            entry, tensors = packed_entry(name, shape, bits, result, split=adapter.split_name(name))
            meta["tensors"].append(entry)
            layer_tensors.update(tensors)
            dead = int(result["dead"].sum())
            unhit = int((n == 0).sum())
            dead_total += dead
            unhit_total += unhit if kind == "experts" else 0
            info = {"name": name, "shape": shape, "dead_columns": dead}
            if kind == "experts":
                info.update(experts_without_tokens=unhit, sequences_per_expert_min=int(n.min()),
                            sequences_per_expert_median=float(n.median()))
            layer_info["tensors"].append(info)
            del result, W
        t2 = time.time()
        with torch.no_grad():
            for s in range(0, nsamples, batch):
                outs[s:s + batch] = adapter.forward(layer, i, inps[s:s + batch])
        save(layer_tensors, f"gptq-layer-{i:03d}.safetensors")
        inps, outs = outs, inps
        # The capturing forward bound to the experts module is a reference cycle (module ->
        # bound method -> module): break it, or every MoE block's weights stay on the card.
        for module in layer.modules():
            module.__dict__.pop("forward", None)
        del layer, targets, capture, layer_tensors
        gc.collect()
        torch.cuda.empty_cache()
        layer_info.update(seconds=round(time.time() - t0, 1), calibrate_seconds=round(t1 - t0, 1),
                          gptq_seconds=round(t2 - t1, 1), output_rms=float(inps.float().pow(2).mean().sqrt()),
                          gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
        report["layers"].append(layer_info)
        log(f"layer {i}/{adapter.num_layers} {layer_info['type']} {layer_info['seconds']}s "
            f"(calib {layer_info['calibrate_seconds']}s, gptq {layer_info['gptq_seconds']}s) rms {layer_info['output_rms']:.3f} "
            f"dead {sum(t['dead_columns'] for t in layer_info['tensors'])} "
            f"unhit {sum(t.get('experts_without_tokens', 0) for t in layer_info['tensors'])} "
            f"gpu {layer_info['gpu_allocated_gib']} GiB")
        (out / "quantize-report.json").write_text(json.dumps(report, indent=1) + "\n")

    # ---- LM head: GPTQ on the final-norm outputs of the quantized model
    if head_bits is not None:
        norm = adapter.final_norm_class(hidden, eps=getattr(adapter.config, "rms_norm_eps", None)
                                        or adapter.config.layer_norm_epsilon).to(device)
        norm.weight.data = adapter.ckpt.get(layout["final_norm"]).to(device)
        head = adapter.ckpt.get(layout["head"]).to(device)
        H = torch.zeros((1, hidden, hidden), dtype=torch.float32, device=device)
        with torch.no_grad():
            for s in range(0, nsamples, batch):
                y = norm(inps[s:s + batch]).reshape(-1, hidden).float()
                H[0].addmm_(y.t(), y)
        H *= 2.0 / nsamples
        result = quantize_target(head.unsqueeze(0), H, head_bits, 1)
        entry, tensors = packed_entry(layout["head"], list(head.shape), head_bits, result)
        meta["tensors"].append(entry)
        save(tensors, "gptq-head.safetensors")
        report["lm_head"] = {"bits": head_bits, "dead_columns": int(result["dead"].sum())}
        del head, H, result
        log("lm_head quantized")
    report["dead_columns"], report["experts_without_tokens"] = dead_total, unhit_total
    finish_checkpoint(spec, adapter, meta, shards, report)
    report["status"] = "complete"
    report["wall_seconds"] = round(time.time() - started, 1)
    (out / "quantize-report.json").write_text(json.dumps(report, indent=1) + "\n")
    log("done", json.dumps(report["package"]))


def finish_checkpoint(spec, adapter, meta, shards, report):
    """gptq_meta.json, the index, config/tokenizer files and links to the original shards that
    hold the tensors left unquantized; then the package accounting from the written files."""
    model, model_out = Path(spec["model_dir"]), Path(spec["model_out"])
    layout = adapter.layout
    tensors = gptq_core.checkpoint_tensors(model)
    text = re.compile(layout["text"])
    replaced = set()
    for entry in meta["tensors"]:
        replaced.update(gptq_codec.original_names(entry))
    kept = {name: info for name, info in tensors.items() if text.search(name) and name not in replaced}
    quant = re.compile(layout["quantize"])
    missing = [n for n in tensors if text.search(n) and quant.search(n) and n not in replaced]
    if missing:
        raise RuntimeError(f"{len(missing)} tensors that should be quantized were not: {missing[:5]}")
    weight_map = dict(shards)
    for name, (_, _, shard) in kept.items():
        weight_map[name] = shard
    for shard in sorted({info[2] for info in kept.values()}):
        (model_out / shard).symlink_to(os.path.realpath(model / shard))
    meta["recipe"] = report["recipe"]
    meta["source"] = {"model_id": spec["model_id"], "revision": spec["model_revision"]}
    (model_out / "gptq_meta.json").write_text(json.dumps(meta) + "\n")
    (model_out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": gptq_codec.FORMAT}, "weight_map": weight_map}, indent=1) + "\n")
    for name in list(SMALL_FILES) + spec["tokenizer_files"] + layout.get("runtime_files", []):
        if (model / name).is_file():
            shutil.copy2(model / name, model_out / name)
    # The package: the GPTQ shards as written, the kept text tensors at their stored size
    # (plus the header a shard holding them would carry), every small file.
    gptq_files = {p.name: p.stat().st_size for p in model_out.glob("gptq-*.safetensors")}
    kept_bytes = sum(gptq_core.tensor_bytes(d, s) for d, s, _ in kept.values())
    offset, header = 0, {}
    for name, (dtype, shape, _) in sorted(kept.items()):
        size = gptq_core.tensor_bytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    kept_header = 8 + len(json.dumps(header, separators=(",", ":")))
    files = {p.name: p.stat().st_size for p in model_out.iterdir()
             if p.is_file() and not p.is_symlink() and not p.name.endswith(".safetensors")}
    embed_head = {n: kept[n] for n in (layout["embed"], layout["head"]) if n in kept}
    total = sum(gptq_files.values()) + kept_bytes + kept_header + sum(files.values())
    report["package"] = {
        "gptq_shards_bytes": sum(gptq_files.values()), "kept_text_tensors": len(kept),
        "kept_text_tensor_bytes": kept_bytes, "kept_header_bytes": kept_header,
        "embed_head_bf16_bytes": sum(gptq_core.tensor_bytes(d, s) for d, s, _ in embed_head.values()),
        "small_files": files, "total_bytes": total, "total_gib": total / 2 ** 30,
        "limit_gib": spec["capacity_gib"], "planned_gib": spec["planned"]["total_gib"],
        "excluded": "vision tower, MTP layers and every other tensor outside the text model",
    }
    if total > spec["capacity_gib"] * 2 ** 30:
        raise RuntimeError(f"package {total / 2**30:.4f} GiB exceeds {spec['capacity_gib']} GiB")


def inside_validate(spec_path: str) -> None:
    """The layer-by-layer path against the model's own forward, and each capturing expert
    forward against the module's own forward, on real weights."""
    global torch
    import torch
    spec = json.loads(Path(spec_path).read_text())
    device = torch.device("cuda")
    adapter = adapter_for(spec, device)
    layout = adapter.layout
    torch.manual_seed(0)
    layers = spec.get("layers") or 4
    T, B = 256, 2
    ids = torch.randint(0, 100000, (B, T), device=device)
    embed = adapter.ckpt.get(layout["embed"]).to(device)
    x = torch.nn.functional.embedding(ids, embed)
    results = {}
    # (1) a truncated copy of the model's own text model vs this script's per-block calls
    if spec["arch"] == "nemotron_h":
        from transformers.models.nemotron_h import modeling_nemotron_h as M
        cfg = type(adapter.config)(**{**adapter.text_config, "layers_block_type":
                                      adapter.config.layers_block_type[:layers]})
        cfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = M.NemotronHModel(cfg)
    else:
        M = adapter.M
        cfg = type(adapter.config)(**{**adapter.text_config, "num_hidden_layers": layers,
                                      "layer_types": adapter.config.layer_types[:layers]})
        cfg._attn_implementation = "sdpa"
        cls = M.Qwen3_5MoeTextModel if spec["arch"] == "qwen3_5_moe" else M.Qwen3_5TextModel
        with torch.device("meta"):
            model = cls(cfg)
    state = {}
    for i in range(layers):
        for k, v in adapter.state_dict(i).items():
            state[f"layers.{i}.{k}"] = v
    state["norm_f.weight" if spec["arch"] == "nemotron_h" else "norm.weight"] = adapter.ckpt.get(layout["final_norm"])
    state["embeddings.weight" if spec["arch"] == "nemotron_h" else "embed_tokens.weight"] = adapter.ckpt.get(layout["embed"])
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if spec["arch"] != "nemotron_h":
        model.rotary_emb = adapter.rotary
    model = model.to(device).eval()
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    with torch.no_grad():
        reference = model(input_ids=ids).last_hidden_state
        h = x
        for i in range(layers):
            h = adapter.forward(model.layers[i], i, h)
        norm = model.norm_f if spec["arch"] == "nemotron_h" else model.norm
        mine = norm(h)
    results["text_model_vs_blocks"] = {"layers": layers, "max_abs_diff": float((reference - mine).abs().max()),
                                       "reference_rms": float(reference.float().pow(2).mean().sqrt()),
                                       "missing": missing, "unexpected": unexpected, "meta": still_meta}
    # (2) each capturing expert forward vs the module's own (eager) forward
    for i in range(layers):
        layer = model.layers[i]
        experts = getattr(getattr(layer, "mlp", None), "experts", None) or getattr(getattr(layer, "mixer", None), "experts", None)
        if experts is None:
            continue
        router = layer.mlp.gate if hasattr(layer, "mlp") else layer.mixer.gate
        hs = torch.randn(B * T, experts.gate_up_proj.shape[2] if hasattr(experts, "gate_up_proj") else adapter.config.hidden_size,
                         device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            _, w, idx = router(hs)
            hs_in = layer.mixer.fc1_latent_proj(hs) if spec["arch"] == "nemotron_h" else hs
            default_impl = experts.config._experts_implementation
            default = type(experts).forward(experts, hs_in, idx, w)
            experts.config._experts_implementation = "eager"
            own = type(experts).forward(experts, hs_in, idx, w)
            experts.config._experts_implementation = default_impl
            capture = Capture(device)
            prefix = "x."
            names = ["gate_up_proj", "down_proj"] if spec["arch"] != "nemotron_h" else ["up_proj", "down_proj"]
            for n_ in names:
                p = getattr(experts, n_)
                capture.experts(prefix + n_, p.shape[0], p.shape[2])
            capture.batch_sequences = B
            fwd = (qwen_experts_forward if spec["arch"] != "nemotron_h" else nemotron_experts_forward)(capture, prefix)
            mine = fwd(experts, hs_in, idx, w)
        results[f"experts_layer_{i}"] = {"max_abs_diff_vs_eager": float((own.float() - mine.float()).abs().max()),
                                         "max_abs_diff_vs_default": float((default.float() - mine.float()).abs().max()),
                                         "default_implementation": str(default_impl),
                                         "own_rms": float(own.float().pow(2).mean().sqrt()),
                                         "hessians_captured": {k: float(v.abs().sum()) for k, v in capture.sums.items()}}
    print(json.dumps(results, indent=1), flush=True)
    (Path(spec["out"]) / "validate-result.json").write_text(json.dumps(results, indent=1) + "\n")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--inside"]:
        {"quantize": inside_quantize, "validate": inside_validate}[argv[1]](argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("task")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("quantize")
    p.add_argument("task")
    p.add_argument("--gpu", required=True)
    p.add_argument("--output-root", type=Path, help="write regenerated weights separately from the recorded baseline")
    p.add_argument("--batch", type=int, default=8, help="calibration sequences per forward (math unchanged)")
    p.add_argument("--expert-chunk", type=int, default=32, help="experts quantized together (math unchanged)")
    p.set_defaults(func=cmd_quantize)
    p = sub.add_parser("validate")
    p.add_argument("task")
    p.add_argument("--gpu", required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--layers", type=int, default=4)
    p.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
