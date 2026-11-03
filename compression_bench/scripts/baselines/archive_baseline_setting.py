#!/usr/bin/env python3
"""Archive a baseline_latency.py record's implementation and provenance, or reuse a verified identical BF16 run.

    archive_baseline_setting.py bf16|gptq <setting> [--reuse-target <setting>]

A GPTQ record is checked against the quantization it measured (gptq_baseline.py's quantize-report.json):
the recipe, the calibration draw and the package's tensor bytes must match. `--reuse-target` copies a
measured BF16 record to another setting of the same checkpoint once every benchmark input, decode setting
and quality reference of the two tasks is shown identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'evaluation_progress/baselines/latency'
GPTQ_OUT = REPO / 'compression_bench/evolve_space/baselines/gptq'   # gptq_baseline.py OUT_ROOT


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def archive(kind, setting):
    record = ROOT / kind / setting
    config = read(record / 'configuration.json')
    work = REPO / config['work_dir']
    runtime = record / 'runtime'
    runtime.mkdir(exist_ok=True)
    for source in (work / 'submission').glob('*.py'):
        shutil.copy2(source, runtime / source.name)
    for source in (work / 'evaluator').glob('*.py'):
        shutil.copy2(source, runtime / source.name)
    shutil.copy2(work / 'runner.py', runtime / 'runner.py')
    for source in (work / 'evaluator').glob('*.dist-info'):
        shutil.copytree(source, runtime / source.name, dirs_exist_ok=True)
    assets = runtime / 'evaluator-assets'
    assets.mkdir(exist_ok=True)
    for name in ('decode_config.json', 'task_policy.json'):
        shutil.copy2(work / 'evaluator/evaluator-assets' / name, assets / name)
    assert digest(runtime / 'inference.py') == config['inference_sha256']
    for name in ('gptq_codec.py', 'gptq_packed_runtime.py'):
        if kind == 'gptq':
            assert digest(runtime / name) == config[name + '_sha256']
    task = REPO / 'compression_tasks' / config['task']
    hashes = {}
    for base, names in [('tests/evaluator-assets', ('decode_inputs.json', 'decode_config.json')),
                        ('assets/hidden', ('ppl_inputs.json', 'ppl_config.json', 'accuracy_inputs.json', 'accuracy_config.json'))]:
        for name in names:
            hashes[f'{base}/{name}'] = digest(task / base / name)
    provenance = {'benchmark_input_sha256': hashes,
                  'runtime_sha256': {p.name: digest(p) for p in runtime.glob('*.py')}}
    if kind == 'gptq':
        model = Path(config['model_dir'])
        # The original metadata can survive a temporary regenerated checkpoint;
        # use it only when its content hash proves it is the executed format.
        metadata_source = model
        if not (metadata_source / 'gptq_meta.json').exists():
            metadata_source = GPTQ_OUT / setting / 'model'
        for name, key in [('gptq_meta.json', 'packed_format_sha256'), ('config.json', 'source_config_sha256')]:
            source = metadata_source / name
            assert digest(source) == config[key], name
            shutil.copy2(source, record / ('model-' + name))
        if (model / 'model.safetensors.index.json').exists():
            shutil.copy2(model / 'model.safetensors.index.json', record / 'model.safetensors.index.json')
        for name in ('quantize-spec.json', 'quantize-report.json', 'compact-kept-report.json'):
            source = model.parent / name
            if source.exists():
                shutil.copy2(source, record / name)
        original = GPTQ_OUT / setting
        current, previous = read(record / 'quantize-report.json'), read(original / 'quantize-report.json')
        # Search/accounting entries can differ by metadata file lengths. The
        # scientific recipe, draw, quantized bytes and kept tensor bytes must match.
        recipe_keys = ('block_bits', 'group_size', 'block_size', 'percdamp', 'act_order',
                       'true_sequential', 'symmetric', 'scale_dtype', 'embed_head',
                       'embedding', 'lm_head', 'calibration', 'selection_rule')
        for key in recipe_keys:
            assert current['recipe'][key] == previous['recipe'][key], key
        assert current['calibration'] == previous['calibration']
        for key in ('gptq_shards_bytes', 'kept_text_tensors', 'kept_text_tensor_bytes',
                    'embed_head_bf16_bytes', 'limit_gib'):
            assert current['package'][key] == previous['package'][key], key
        assert current['package']['total_gib'] <= current['package']['limit_gib']
        provenance['original_recipe_verified'] = True
        provenance['original_package_tensor_bytes_verified'] = True
        provenance['original_quantize_report_sha256'] = digest(original / 'quantize-report.json')
        provenance['calibration'] = current['calibration']
    write(record / 'provenance.json', provenance)
    print(f'Archived {kind}/{setting}')


def reuse(source, target):
    src, dst = ROOT / 'bf16' / source, ROOT / 'bf16' / target
    summary, config = read(src / 'SUMMARY.json'), read(src / 'configuration.json')
    assert summary['status'] == 'measured'
    assert summary['ppl_verification']['status'] == summary['gate_verification']['status'] == 'measured'
    tasks = [REPO / 'compression_tasks' / (s + '_standard_1xH200NVL') for s in (source, target)]
    proof = {'source': f'bf16/{source}', 'execution_settings': config,
             'note': 'Same physical run; no independent target samples were collected.', 'identical_sha256': {}}
    for base, name in [('tests/evaluator-assets', 'decode_inputs.json'),
                       ('assets/hidden', 'ppl_inputs.json'), ('assets/hidden', 'accuracy_inputs.json'),
                       ('', 'assets/model_facts.json')]:
        a, b = [t / base / name for t in tasks]
        assert digest(a) == digest(b), name
        proof['identical_sha256'][str(Path(base) / name)] = digest(a)
    configs = [read(t / 'tests/evaluator-assets/decode_config.json') for t in tasks]
    for key in ('measurement_protocol', 'vllm_version', 'input_tokens', 'output_tokens',
                'requests_per_round', 'rounds', 'max_model_len', 'max_num_seqs', 'gpu_memory_utilization'):
        assert configs[0][key] == configs[1][key], key
    for name in ('ppl_config.json', 'accuracy_config.json'):
        assert read(tasks[0] / 'assets/hidden' / name) == read(tasks[1] / 'assets/hidden' / name), name
    shutil.copytree(src, dst, dirs_exist_ok=True)
    config.update(task=target + '_standard_1xH200NVL', reused_from=f'bf16/{source}',
                  reuse_proof='reuse-proof.json')
    summary['reused_from'] = f'bf16/{source}'
    write(dst / 'configuration.json', config)
    write(dst / 'SUMMARY.json', summary)
    write(dst / 'reuse-proof.json', proof)
    print(f'Reused bf16/{source} for {target}, after input/configuration checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('bf16', 'gptq'))
    parser.add_argument('setting')
    parser.add_argument('--reuse-target')
    args = parser.parse_args()
    archive(args.kind, args.setting)
    if args.reuse_target:
        assert args.kind == 'bf16'
        reuse(args.setting, args.reuse_target)
